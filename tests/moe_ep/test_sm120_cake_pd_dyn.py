# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental (local only): kernel_sched closure and in-kernel dynamic EPLB on the W4A8, NVFP4 and
MXFP4 (W4A4, ``mxfp4``) Cake backends, with redundant experts.

Geometry: DeepSeek-V4-Flash routed experts (H4096 / I2048 / top-6), 256 logical experts plus
``CAKE_SM120_PD_DYN_REDUNDANT`` (default 16) replicas, i.e. 68 physical slots per rank at EP4.  Replicas
carry byte copies of their logical expert's weights (and, for NVFP4, its per-expert global scales).
Routing is skewed onto 16 hot logical experts (4 per rank); each hot expert has one replica on the next
rank.  The static ids follow the round-robin replica rule (``cake_dyneplb.rr_topk``).

For every case (``CAKE_SM120_PD_DYN_CASES``, ``backend:return_codec[:variant]``) three layers on the same
physical weights must give identical bytes for the same inputs:

* the default closure, static ids;
* the kernel_sched closure, static ids;
* the kernel_sched closure with ``dynamic_eplb=True`` after ``set_dyn_placement`` (``min_rows=0``: the
  in-kernel planner re-splits every call), static ids as input.

The dyn layer's counters must show one plan per call and no fallback.

    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_sm120_cake_pd_dyn.py -m "gpu_4 and arch_sm120" -v
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

from .test_moe_ep_sm120_nvfp4_cake_mega import _expert_planes, expert_scales
from .test_moe_ep_sm120_w4a8_cake_mega import (
    DEFAULT_SWIGLU_LIMIT,
    SHAPES,
    init_distributed,
    local_weights,
    table_spec,
)

LOGICAL = 256
HOT_PER_RANK = 4
REDUNDANT = int(os.environ.get("CAKE_SM120_PD_DYN_REDUNDANT", "16"))
ROWS = tuple(
    int(r) for r in os.environ.get("CAKE_SM120_PD_DYN_ROWS", "512,1024,4096").split(",")
)
CASES = os.environ.get(
    "CAKE_SM120_PD_DYN_CASES",
    "w4a8:bf16,w4a8:int8,nvfp4:int8,w4a8:int8:fp8shared,nvfp4:int8:fp8shared,"
    "mxfp4:bf16,mxfp4:int8,mxfp4:int8:fp8shared",
)
SEED = 20261109
# dyn_stats words (generator DYN_ST_*): epochs, gated, blob mismatch, CTA deadline, CTA bad row,
# rank fallback, bad route, plans, max planner iterations, cap hit, map miss
ST_EPOCHS, ST_GATED, ST_BLOB, ST_DEADLINE, ST_ROW, ST_FALLBACK, ST_BAD = range(7)
ST_PLANS, ST_MAP_MISS = 7, 10


def _cases() -> list[tuple[str, str, str]]:
    out = []
    for item in CASES.split(","):
        if item.strip():
            backend, codec, *rest = item.strip().split(":")
            out.append((backend, codec, rest[0] if rest else "noshared"))
    return out


def placement(world_size: int) -> list[int]:
    """physical_to_logical: rank r hosts logical [r * 64, (r + 1) * 64) plus replicas of hot experts."""

    per = LOGICAL // world_size
    hot = [r * per + h for r in range(world_size) for h in range(HOT_PER_RANK)]
    extra = REDUNDANT // world_size
    p2l = []
    for r in range(world_size):
        p2l += list(range(r * per, (r + 1) * per))
        donor = (r + 1) % world_size
        donors = [e for e in hot if e // per == donor]
        p2l += [donors[i % len(donors)] for i in range(extra)]
    return p2l


def logical_routing(world_size: int, rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[world, rows, K] logical ids (skewed onto the hot experts) and weights; same on every rank."""

    g = torch.Generator(device="cpu").manual_seed(seed)
    per = LOGICAL // world_size
    logits = torch.randn(world_size, rows, LOGICAL, generator=g)
    for r in range(world_size):
        logits[..., r * per : r * per + HOT_PER_RANK] += 2.5
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), 6, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return ids.to(torch.int64), weights.float()


def nvfp4_local(rank: int, world_size: int, p2l: list[int], shape, shared: bool):
    from flashinfer.moe_ep import MoEWeightPack

    local = shape.experts // world_size
    logical = p2l[rank * local : (rank + 1) * local]
    parts = [_expert_planes(e, SEED, shape) for e in logical]
    w13, w13_scale, w2, w2_scale = (torch.stack([p[i] for p in parts]) for i in range(4))
    routed = (w13, w2, w13_scale.view(torch.float8_e4m3fn), w2_scale.view(torch.float8_e4m3fn))
    a1, a2, nc = expert_scales(torch.tensor(logical, device="cuda"))
    scales = (a1.contiguous(), a2.contiguous(), nc.contiguous())
    if not shared:
        return MoEWeightPack(*routed), scales
    w8 = local_weights(rank, world_size, SEED, shared=True, shape=dataclasses.replace(shape, experts=world_size))
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        CakePrequantizedMoEWeightsWithSharedExpert,
    )

    return (
        CakePrequantizedMoEWeightsWithSharedExpert(
            *routed,
            shared_w13=w8.shared_w13,
            shared_w2=w8.shared_w2,
            shared_w13_scale=w8.shared_w13_scale,
            shared_w2_scale=w8.shared_w2_scale,
        ),
        scales,
    )


def make_layer(rank, world_size, backend, codec, variant, weights, scales, shape, *, ks, dyn):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    common = dict(
        intermediate_size=shape.intermediate,
        top_k=shape.top_k,
        shared_expert="fp8_in_kernel" if variant == "fp8shared" else None,
        transport_table=table_spec(),
        kernel_sched=ks,
        dynamic_eplb=dyn,
    )
    if backend == "mxfp4":
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake import (
            Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig,
        )

        cfg = Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig(
            swiglu_limit=DEFAULT_SWIGLU_LIMIT, return_codec=codec, **common
        )
    elif backend == "nvfp4":
        cfg = Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(
            gate_up_clamp=DEFAULT_SWIGLU_LIMIT,
            combine_dtype=codec,
            fc1_alpha=scales[0],
            fc2_alpha=scales[1],
            fc1_norm_const=scales[2],
            **common,
        )
    else:
        cfg = Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
            swiglu_limit=DEFAULT_SWIGLU_LIMIT, return_codec=codec, **common
        )
    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank, device=torch.cuda.current_device()),
        fleet_params=FleetParams(
            num_experts=shape.experts, max_tokens_per_rank=max(ROWS), token_hidden_size=shape.hidden
        ),
        weights=weights,
        backend=MegaConfig(megakernel=cfg, quantize_input=True, preprocess_weights=True),
    )


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
@pytest.mark.parametrize("case", _cases(), ids=lambda c: "-".join(c))
def test_ks_and_dyn_eplb_match_static(case) -> None:
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_dyneplb import (
        Placement,
        rr_topk,
    )

    backend, codec, variant = case
    rank, world_size = init_distributed()
    if world_size not in (4, 8, 16) or REDUNDANT % world_size:
        pytest.skip("dyn EPLB is built for EP 4 / 8 / 16 with REDUNDANT % world == 0")
    p2l = placement(world_size)
    shape = dataclasses.replace(SHAPES["v4flash"], experts=len(p2l))
    shared = variant == "fp8shared"
    if backend == "nvfp4":
        weights, scales = nvfp4_local(rank, world_size, p2l, shape, shared)
    else:
        weights = local_weights(rank, world_size, SEED, shared=shared, shape=shape, physical_to_logical=p2l)
        scales = None
    pl = Placement.from_lists(p2l, world_size)

    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_local_split import (
        LocalSplit,
        local_split_reference,
    )

    split = LocalSplit(p2l, world_size, rank, torch.device("cuda", torch.cuda.current_device()), mode="own")
    cases, ls_cases = [], []
    for i, rows in enumerate(ROWS):
        ids_l, w = logical_routing(world_size, rows, SEED + i)
        phys = rr_topk(ids_l, pl)
        g = torch.Generator(device="cuda").manual_seed(SEED + 17 * i + rank)
        h = torch.randn(rows, shape.hidden, dtype=torch.bfloat16, device="cuda", generator=g)
        cases.append(
            MoEEpTensors(
                hidden_states=h,
                topk_ids=phys[rank].cuda().contiguous(),
                topk_weights=w[rank].cuda().contiguous(),
            )
        )
        # source-local split (cake_local_split, "own"): device kernel == host reference, ids keep their logical expert
        ls_ids = split(ids_l[rank].cuda())
        ref, _ = local_split_reference(ids_l[rank], split.tables, rank, tile=split.tile)
        assert torch.equal(ls_ids.cpu(), torch.as_tensor(ref).view_as(ls_ids).cpu())
        assert torch.equal(torch.tensor(p2l)[ls_ids.cpu()], ids_l[rank])
        ls_cases.append(dataclasses.replace(cases[-1], topk_ids=ls_ids.contiguous()))

    outs = {}
    stats = None
    for arm, ks, dyn in (("default", False, False), ("ks", True, False), ("ksdyn", True, True)):
        layer = make_layer(rank, world_size, backend, codec, variant, weights, scales, shape, ks=ks, dyn=dyn)
        layer.warmup(cases[0])
        if dyn:
            layer._kernel.set_dyn_placement(
                layer._workspace, p2l, logical_experts=LOGICAL, min_rows=0, wait_ns=1_000_000
            )
            s0 = layer._kernel.dyn_stats(layer._workspace)
        outs[arm] = [layer(t).clone() for t in cases]
        if arm == "ks":
            outs["ksls"] = [layer(t).clone() for t in ls_cases]
        torch.cuda.synchronize()
        if dyn:
            s1 = layer._kernel.dyn_stats(layer._workspace)
            stats = [b - a for a, b in zip(s0, s1, strict=True)]
        assert layer._kernel.protocol_error(layer._workspace) == 0, arm
        layer.destroy()
        dist.barrier()

    for arm in ("ks", "ksdyn", "ksls"):
        for rows, a, b in zip(ROWS, outs["default"], outs[arm], strict=True):
            assert torch.equal(a, b), f"{arm} differs from the default closure at rows={rows}"
    assert stats[ST_EPOCHS] == len(ROWS), stats
    assert stats[ST_PLANS] == len(ROWS), stats
    for k in (ST_GATED, ST_BLOB, ST_DEADLINE, ST_ROW, ST_FALLBACK, ST_BAD, ST_MAP_MISS):
        assert stats[k] == 0, (k, stats)


@pytest.mark.parametrize("act_codec", (0, 1, 2))
@pytest.mark.parametrize("return_codec", (0, 1))
def test_ks_closure_resolves_dyn_builds(act_codec, return_codec) -> None:
    """CPU: the kernel_sched closure lists MOE_DYN_EPLB and resolves dyn builds of both codecs at EP 4 / 8 / 16."""

    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_geometry import (
        GeometryError,
        KernelParams,
        describe_kernel,
    )
    from flashinfer.moe_ep.kernel_src.sm120.cake_mxfp4_megamoe_ks import DEVICE_PARAMETERS, read_manifest

    assert DEVICE_PARAMETERS[-1] == "MOE_DYN_EPLB"
    assert set(read_manifest()["device"]["parameters"]) == {*DEVICE_PARAMETERS, "MOE_INSTANCE"}
    for ranks in (4, 8, 16):
        for shared in (False, True):
            base = dict(ranks=ranks, experts=256 + 2 * ranks * 2, top_k=6, max_rows=4096, hidden=4096,
                        intermediate=2048, shared_expert=shared, return_codec=return_codec, act_codec=act_codec)
            off = describe_kernel(KernelParams(**base, kernel_sched=True))
            on = describe_kernel(KernelParams(**base, kernel_sched=True, dyn_eplb=1))
            names_off = [a["name"] for a in off["abi"]["args"]]
            names_on = [a["name"] for a in on["abi"]["args"]]
            assert "dyn_placement" not in names_off
            assert {"dyn_placement", "dyn_scratch", "dyn_stats"} <= set(names_on)
    with pytest.raises(GeometryError):
        KernelParams(ranks=4, experts=256, top_k=6, max_rows=4096, hidden=4096, intermediate=2048,
                     shared_expert=False, dyn_eplb=1)
