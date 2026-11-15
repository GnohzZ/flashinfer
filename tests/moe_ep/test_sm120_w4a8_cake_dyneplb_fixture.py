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

"""Per-batch EPLB token split on a real EPLB fixture reproduces the logical run bit for bit
(experimental, dynamic-EPLB P1a; opt-in).

``CAKE_SM120_MEGAMOE_EPLB_FIXTURE_DIR`` names an EPLB fixture
(``megamoe-unified/tools/eplb/make_eplb_fixture.py``: physical-id routing, replica
weights byte-copied from the logical fixture, ``meta.json`` ``eplb.phy2log``); its
``eplb.source_fixture`` (or ``CAKE_SM120_MEGAMOE_LOGICAL_FIXTURE_DIR``) is the logical
fixture.  For every row count the logical routing (``phy2log[fixture ids]``) of all
ranks gives the batch's per-source counts; :func:`plan_quotas` + :class:`SplitMapper`
map this rank's routes.  The split runs (plain, diagonal, interleaved, both) on the physical layer, the
fixture's own round-robin run and the logical run must give identical bytes:

    CAKE_SM120_MEGAMOE_EPLB_FIXTURE_DIR=.../real-ep8-p288 torchrun --nproc_per_node=8 \
        -m pytest tests/moe_ep/test_sm120_w4a8_cake_dyneplb_fixture.py -m "gpu_4 and arch_sm120" -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.moe_ep.test_moe_ep_sm120_w4a8_cake_mega import (
    HIDDEN,
    INTERMEDIATE,
    TOP_K,
    init_distributed,
    table_spec,
)
from tests.moe_ep.test_sm120_w4a8_cake_fixture import _canonical_from_kernel_layout

EPLB_ENV = "CAKE_SM120_MEGAMOE_EPLB_FIXTURE_DIR"
LOGICAL_ENV = "CAKE_SM120_MEGAMOE_LOGICAL_FIXTURE_DIR"
CAPACITY = int(os.environ.get("CAKE_SM120_MEGAMOE_FIXTURE_CAPACITY", "8192"))
ROWS = tuple(
    int(r)
    for r in os.environ.get(
        "CAKE_SM120_MEGAMOE_DYNEPLB_ROWS", "512,1000,2048,8192"
    ).split(",")
)


def _make_layer(root: Path, rank: int, world_size: int, experts: int):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    fx = torch.load(root / f"rank{rank}_weights.pt", map_location="cpu")
    c = _canonical_from_kernel_layout(fx, device, experts // world_size)
    weights = MoEWeightPack(
        c["w13"],
        c["w2"],
        c["w13_scale"].view(torch.float8_e8m0fnu),
        c["w2_scale"].view(torch.float8_e8m0fnu),
    )
    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, device=device.index
        ),
        fleet_params=FleetParams(
            num_experts=experts,
            max_tokens_per_rank=CAPACITY,
            token_hidden_size=HIDDEN,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOP_K,
                shared_expert=None,
                transport_table=table_spec(),
            ),
            quantize_input=False,
            preprocess_weights=True,
        ),
    )


def _run(layer, cases):
    import torch.distributed as dist

    layer.warmup(cases[0])
    outs = [layer(t).clone() for t in cases]
    torch.cuda.synchronize()
    assert layer._kernel.protocol_error(layer._workspace) == 0
    layer.destroy()
    dist.barrier()
    return outs


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_dynamic_split_matches_logical_bitwise() -> None:
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_dyneplb import (
        Placement,
        SplitMapper,
        padded_rows_per_rank,
        plan_quotas,
        rr_topk,
    )

    root = os.environ.get(EPLB_ENV)
    if not root:
        pytest.skip(f"set {EPLB_ENV} to an EPLB fixture directory")
    root = Path(root)
    rank, world_size = init_distributed()
    meta = json.loads((root / "meta.json").read_text())
    eplb = meta["eplb"]
    if int(eplb["ranks"]) != world_size:
        pytest.skip(f"fixture is EP{eplb['ranks']}, launched EP{world_size}")
    logical_root = Path(os.environ.get(LOGICAL_ENV, eplb["source_fixture"]))
    placement = Placement.from_lists(eplb["phy2log"], world_size)
    n_log, n_phys = placement.logical, placement.physical
    assert n_log == int(eplb["logical_experts"]) and n_phys == int(
        eplb["physical_experts"]
    )

    inputs = torch.load(root / "inputs.pt", map_location="cpu")
    p2l = torch.tensor(eplb["phy2log"], dtype=torch.int64)
    fixture_phys = inputs["topk_idx"].long()  # [ranks, rows, k], fixture rr rule
    logical_all = p2l[fixture_phys]
    device = torch.device("cuda", torch.cuda.current_device())
    mapper = SplitMapper(placement, device)

    modes = {
        "split": {},
        "diag": {"diagonal": True},
        "ilv": {"interleave": True},
        "dilv": {"diagonal": True, "interleave": True},
    }
    cases = {m: [] for m in ("logical", "rr", *modes)}
    for rows in ROWS:
        tile = 64 if rows <= 512 else 128
        tk = logical_all[:, :rows]
        counts = [
            torch.bincount(tk[s].flatten(), minlength=n_log).tolist()
            for s in range(world_size)
        ]
        quotas = plan_quotas(counts, placement, tile)
        own = tk[rank].to(device)
        ids = {
            "logical": own,
            "rr": rr_topk(tk, placement)[rank].to(device),
            **{
                m: mapper(own, mapper.tables(counts, quotas, rank, **kw))
                for m, kw in modes.items()
            },
        }
        assert torch.equal(ids["rr"].cpu(), fixture_phys[rank, :rows])
        for m in modes:
            assert torch.equal(p2l[ids[m].cpu()], tk[rank])
        if rank == 0:
            full = torch.stack(
                [
                    mapper(tk[s].to(device), mapper.tables(counts, quotas, s)).cpu()
                    for s in range(world_size)
                ]
            )
            assert torch.bincount(full.flatten(), minlength=n_phys).tolist() == quotas
            print(
                f"rows={rows} padded max rr/split = "
                f"{max(padded_rows_per_rank(fixture_phys[:, :rows], n_phys, world_size, tile))}/"
                f"{max(padded_rows_per_rank(full, n_phys, world_size, tile))}"
            )
        x_fp8 = inputs["x_fp8"][rank, :rows].to(device).contiguous()
        x_sf = inputs["x_sf"][rank, :rows].to(device).contiguous()
        w = inputs["topk_weights"][rank, :rows].float().to(device).contiguous()
        for m, t in ids.items():
            cases[m].append(
                MoEEpTensors(
                    hidden_states=x_fp8,
                    topk_ids=t.to(torch.int64).contiguous(),
                    topk_weights=w,
                    scales=x_sf,
                )
            )

    physical = {}
    layer = _make_layer(root, rank, world_size, n_phys)
    layer.warmup(cases["rr"][0])
    for m in ("rr", *modes):
        physical[m] = [layer(t).clone() for t in cases[m]]
    torch.cuda.synchronize()
    assert layer._kernel.protocol_error(layer._workspace) == 0
    layer.destroy()
    import torch.distributed as dist

    dist.barrier()
    logical = _run(
        _make_layer(logical_root, rank, world_size, n_log), cases["logical"]
    )
    for m in ("rr", *modes):
        for rows, a, b in zip(ROWS, physical[m], logical, strict=True):
            assert torch.equal(a, b), f"{m} differs from logical at rows={rows}"
    dist.barrier()
