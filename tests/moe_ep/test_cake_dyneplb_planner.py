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

"""CPU tests of the per-batch EPLB planner and tile-aligned token split (cake_dyneplb)."""

from __future__ import annotations

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_dyneplb import (
    Placement,
    SplitMapper,
    padded_rows_per_rank,
    plan_quotas,
    rr_topk,
    split_topk,
)

E, K = 256, 6


def make_placement(ranks: int, redundant: int, seed: int, *, same_rank_dup: bool):
    """Home block per rank + ``redundant // ranks`` replicas of random experts per rank;
    optionally one replica lands on a rank that already hosts the expert."""
    g = torch.Generator().manual_seed(seed)
    home = E // ranks
    extra = redundant // ranks
    p2l: list[int] = []
    for r in range(ranks):
        own = list(range(r * home, (r + 1) * home))
        others = [e for e in torch.randperm(E, generator=g).tolist() if e not in own]
        reps = others[:extra]
        if same_rank_dup and r == 0:
            reps[0] = own[0]
        p2l += sorted(own + reps)
    return Placement.from_lists(p2l, ranks)


def skewed_topk(ranks: int, rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(E, generator=g) ** 6 + 1e-3
    out = torch.empty((ranks, rows, K), dtype=torch.int64)
    for s in range(ranks):
        out[s] = torch.multinomial(w.expand(rows, E), K, generator=g)
    return out


CASES = [
    (4, 512, 128, False),
    (8, 512, 64, True),
    (8, 2048, 128, False),
    (8, 1000, 128, True),
]


@pytest.mark.parametrize("ranks,rows,tile,dup", CASES)
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("interleave", [False, True])
def test_split_maps_back_and_meets_quotas(ranks, rows, tile, dup, diagonal, interleave):
    P = make_placement(ranks, 32, seed=ranks + rows, same_rank_dup=dup)
    topk = skewed_topk(ranks, rows, seed=rows)
    topk[0, :3, -1] = -1  # padding slots stay untouched
    counts = [
        torch.bincount(topk[s][topk[s] >= 0], minlength=E).tolist()
        for s in range(ranks)
    ]
    quotas = plan_quotas(counts, P, tile)
    n = torch.tensor(counts).sum(0)
    # quotas: per expert sum == routes, all tile-aligned except at most one
    for e in range(E):
        q = [quotas[p] for p in P.replicas[e]]
        assert sum(q) == int(n[e])
        assert sum(1 for x in q if x % tile) <= 1
        # merged same-rank duplicates: only the first replica on a rank gets rows
        reps = {}
        for p in P.replicas[e]:
            reps.setdefault(p // P.local, []).append(p)
        for ps in reps.values():
            assert all(quotas[p] == 0 for p in ps[1:])
    phys = split_topk(topk, quotas, P, diagonal=diagonal, interleave=interleave)
    p2l = torch.tensor(P.physical_to_logical)
    assert torch.equal(torch.where(phys >= 0, p2l[phys.clamp_min(0)], phys), topk)
    got = torch.bincount(phys[phys >= 0], minlength=P.physical)
    assert got.tolist() == quotas
    # per-rank device mapping == the reference on every rank
    mapper = SplitMapper(P, "cpu")
    for s in range(ranks):
        out = mapper(
            topk[s],
            mapper.tables(counts, quotas, s, diagonal=diagonal, interleave=interleave),
        )
        assert torch.equal(out, phys[s])
    # the split is never worse than round-robin on the tile-padded critical rank
    rr = rr_topk(topk, P)
    assert max(padded_rows_per_rank(phys, P.physical, ranks, tile)) <= max(
        padded_rows_per_rank(rr, P.physical, ranks, tile)
    )


def test_diagonal_keeps_local_routes():
    ranks, rows, tile = 8, 2048, 128
    P = make_placement(ranks, 32, seed=3, same_rank_dup=False)
    topk = skewed_topk(ranks, rows, seed=5)
    counts = [torch.bincount(topk[s].flatten(), minlength=E).tolist() for s in range(ranks)]
    quotas = plan_quotas(counts, P, tile)
    plain = split_topk(topk, quotas, P)
    diag = split_topk(topk, quotas, P, diagonal=True)

    def local_routes(phys):
        return sum(
            int((phys[s] // P.local == s).sum()) for s in range(ranks)
        )

    assert local_routes(diag) >= local_routes(plain)
    assert torch.bincount(diag.flatten(), minlength=P.physical).tolist() == quotas


def test_plan_is_deterministic_and_rank_independent():
    ranks = 8
    P = make_placement(ranks, 32, seed=11, same_rank_dup=True)
    topk = skewed_topk(ranks, 777, seed=13)
    counts = [torch.bincount(topk[s].flatten(), minlength=E).tolist() for s in range(ranks)]
    q1 = plan_quotas(counts, P, 128)
    q2 = plan_quotas(torch.tensor(counts), P, 128)
    # only the per-expert totals matter
    q3 = plan_quotas([list(map(sum, zip(*counts)))], P, 128)
    assert q1 == q2 == q3


def test_single_replica_experts_unchanged():
    ranks = 4
    P = Placement.from_lists(list(range(E)), ranks)  # no redundancy
    topk = skewed_topk(ranks, 300, seed=1)
    counts = [torch.bincount(topk[s].flatten(), minlength=E).tolist() for s in range(ranks)]
    quotas = plan_quotas(counts, P, 128)
    assert torch.equal(split_topk(topk, quotas, P), topk)


def test_interleave_spreads_sources_over_replicas():
    ranks, rows, tile = 8, 2048, 128
    P = make_placement(ranks, 32, seed=3, same_rank_dup=False)
    topk = skewed_topk(ranks, rows, seed=5)
    counts = [torch.bincount(topk[s].flatten(), minlength=E).tolist() for s in range(ranks)]
    quotas = plan_quotas(counts, P, tile)

    def max_pair(phys):
        owner = phys // P.local
        return max(
            int((owner[s] == r).sum())
            for s in range(ranks)
            for r in range(ranks)
            if r != s
        )

    plain = split_topk(topk, quotas, P)
    ilv = split_topk(topk, quotas, P, interleave=True)
    assert max_pair(ilv) <= max_pair(plain)
    assert torch.bincount(ilv.flatten(), minlength=P.physical).tolist() == quotas


def _blob_constants(ranks: int, experts: int) -> dict[str, int]:
    """The switch-on build's DYN_BLOB_* layout (generator formulas) for a host-only blob test."""
    mh_max = 512 // ranks
    C = {"PHYSICAL_RANKS": ranks, "ROUTED_EXPERTS": experts, "LOCAL_EXPERTS": experts // ranks, "DYN_MH_MAX": mh_max}
    C["DYN_BLOB_REP"] = 8 + experts
    C["DYN_BLOB_MHIDX"] = C["DYN_BLOB_REP"] + ranks * experts
    C["DYN_BLOB_MHREP"] = C["DYN_BLOB_MHIDX"] + experts
    C["DYN_BLOB_SHOST"] = C["DYN_BLOB_MHREP"] + mh_max * ranks
    C["DYN_BLOB_CNT_LSA"] = C["DYN_BLOB_SHOST"] + experts
    C["DYN_BLOB_AFF"] = C["DYN_BLOB_CNT_LSA"] + ranks
    C["DYN_BLOB_MHMASK"] = C["DYN_BLOB_AFF"] + 1
    C["DYN_BLOB_WORDS"] = C["DYN_BLOB_MHMASK"] + mh_max
    return C


def test_blob_early_word_only_in_builds_that_have_it():
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_dyneplb import dyn_placement_blob

    ranks = 4
    P = make_placement(ranks, 32, seed=9, same_rank_dup=False)
    old = _blob_constants(ranks, P.physical)
    new = dict(old, DYN_BLOB_EARLY=old["DYN_BLOB_WORDS"], DYN_BLOB_WORDS=old["DYN_BLOB_WORDS"] + 1)
    p2l = list(P.physical_to_logical)
    assert len(dyn_placement_blob(old, p2l, logical_experts=E, world_size=ranks, early_ctas=24)) == old["DYN_BLOB_WORDS"]
    blob = dyn_placement_blob(new, p2l, logical_experts=E, world_size=ranks, early_ctas=24)
    assert len(blob) == new["DYN_BLOB_WORDS"] and blob[new["DYN_BLOB_EARLY"]] == 24


@pytest.mark.parametrize("affinity,word", [(True, 1), (False, 0), (0, 0), (2048, 2048)])
def test_blob_affinity_word_is_a_row_threshold(affinity, word):
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_dyneplb import dyn_placement_blob

    ranks = 8
    P = make_placement(ranks, 32, seed=7, same_rank_dup=False)
    C = _blob_constants(ranks, P.physical)
    blob = dyn_placement_blob(C, list(P.physical_to_logical), logical_experts=E, world_size=ranks, affinity=affinity)
    assert blob[C["DYN_BLOB_AFF"]] == word
    with pytest.raises(ValueError):
        dyn_placement_blob(C, list(P.physical_to_logical), logical_experts=E, world_size=ranks, affinity=-1)


def test_dyn_eplb_defaults_follow_the_measured_policy():
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_dyneplb import dyn_eplb_defaults

    assert dyn_eplb_defaults(4)["affinity"] == 0 and dyn_eplb_defaults(4)["min_rows"] == 512
    assert dyn_eplb_defaults(8)["affinity"] == 2048 and dyn_eplb_defaults(8)["min_rows"] == 512
    assert dyn_eplb_defaults(16)["affinity"] == 2048
