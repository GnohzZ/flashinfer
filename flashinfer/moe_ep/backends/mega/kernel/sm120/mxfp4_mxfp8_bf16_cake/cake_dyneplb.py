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

"""Per-batch EPLB token split over existing replicas (experimental, dynamic-EPLB P1a).

With static EPLB every logical expert ``e`` has one or more physical replicas (byte
copies of its weights) on fixed ranks.  The router picks a replica per (token, slot);
the static rule is round-robin, which ignores the batch's actual load.  Here the
choice is made per batch from the batch's per-source per-expert route counts
``counts[s][e]`` (what an all_gather of the router histograms gives every rank):

1. :func:`plan_quotas` -- a deterministic min-max greedy over tile-aligned pieces
   (dyneplb/phase0/balance_sim.py ``greedy`` with no new slots) assigns every
   physical replica a row quota.  All quotas of an expert are multiples of the tile
   except at most one (the remainder), so the total padded rows of an expert stay
   ``ceil(n / tile) * tile`` and the per-rank padded rows are what the kernel's tile
   cursor walks.  Replicas of one expert on the same rank are merged into the first
   one (lowest physical id); the others get quota 0.
2. :func:`split_topk` / :class:`SplitMapper` -- every route ``(s, row, k)`` of expert
   ``e`` gets the global ordinal ``sum_{s' < s} counts[s'][e] + (its position among
   source s's routes of e in (row, k) order)`` and goes to the replica whose quota
   prefix interval holds the ordinal (HALO-Q / MegaFlux App. A.2).  With
   ``diagonal=True`` a source first fills the quota of the replica on its own rank
   (#19646 quota repair) and only the rest is placed by ordinal.  With
   ``interleave=True`` the (remote) ordinal interleaves the sources one route at a
   time -- ``sum_{s'} min(m[s'][e], p) + #{s' < s : m[s'][e] > p}`` for the route at
   remote position ``p`` of source ``s`` (``m`` = remote routes per source) -- so
   every source spreads its routes over the replicas roughly in proportion to the
   quotas instead of sending them all to one replica (the source-ordered rule
   concentrates traffic on few source->owner links; measured slower at EP8).

Every rank computes the same plan from the same counts, so no plan exchange is
needed; each rank maps only its own routes.  Physical ids follow the kernel ABI
``owner * local + slot``; a replica's weights are byte copies of the logical
expert's, so the kernel output is bit-identical to the logical run.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

import torch

__all__ = [
    "Placement",
    "plan_quotas",
    "split_topk",
    "rr_topk",
    "padded_rows_per_rank",
    "SplitMapper",
]


@dataclass(frozen=True)
class Placement:
    """Static replica placement: ``physical_to_logical[p]`` for ``p < ranks * local``."""

    physical_to_logical: tuple[int, ...]
    ranks: int
    logical: int
    local: int = field(init=False)
    replicas: tuple[tuple[int, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        n_phys = len(self.physical_to_logical)
        if n_phys % self.ranks:
            raise ValueError(f"{n_phys} physical experts on {self.ranks} ranks")
        reps: list[list[int]] = [[] for _ in range(self.logical)]
        for p, e in enumerate(self.physical_to_logical):
            reps[e].append(p)
        if any(not r for r in reps):
            raise ValueError("every logical expert needs a physical replica")
        object.__setattr__(self, "local", n_phys // self.ranks)
        object.__setattr__(self, "replicas", tuple(tuple(r) for r in reps))

    @classmethod
    def from_lists(cls, physical_to_logical, ranks: int) -> "Placement":
        p2l = tuple(int(x) for x in physical_to_logical)
        return cls(p2l, ranks, max(p2l) + 1)

    @property
    def physical(self) -> int:
        return len(self.physical_to_logical)

    def host_ranks(self, e: int) -> list[tuple[int, int]]:
        """``[(rank, representative physical id)]`` of ``e``, by physical id."""
        out, seen = [], set()
        for p in self.replicas[e]:
            r = p // self.local
            if r not in seen:
                seen.add(r)
                out.append((r, p))
        return out


def _ceil(n: int, t: int) -> int:
    return (n + t - 1) // t * t


def _best_move(n_src: int, load_r: int, load_q: int, n_dst: int, t: int):
    """balance_sim.best_move: move x rows (tile-aligned < n_src, or all n_src) r -> q."""
    best = (max(load_r, load_q), 0)
    cands = set()
    if n_src > t:
        half = (load_r - load_q) // 2
        lo = max(t, half // t * t)
        for x in (lo, lo + t):
            if x < n_src:
                cands.add(x)
        top = (n_src - 1) // t * t
        if top >= t:
            cands.add(min(top, max(t, lo)))
    for x in sorted(cands):
        m = max(load_r - x, load_q + x)
        if m < best[0]:
            best = (m, x)
    nr = load_r - _ceil(n_src, t)
    nq = load_q + _ceil(n_dst + n_src, t) - _ceil(n_dst, t)
    if max(nr, nq) < best[0]:
        best = (max(nr, nq), n_src)
    return best


def plan_quotas(
    counts,
    placement: Placement,
    tile: int = 128,
    *,
    init: str = "tile",
    max_iter: int = 400,
    top_pieces: int = 12,
) -> list[int]:
    """Row quota per physical replica for one batch (deterministic, integer only).

    ``counts`` -- ``[ranks][logical]`` routes (any int array-like; only the column sums
    matter for the quotas).  ``init='tile'`` deals each expert's tiles round-robin over
    its host ranks (the last, partial tile to host ``(tiles - 1) % hosts``);
    ``init='even'`` splits rows evenly over the physical replicas like the Phase-0
    simulator (used to cross-check it).  Then the min-max greedy repeatedly moves
    tile-aligned rows (or a whole piece) of one of the heaviest rank's
    ``top_pieces`` largest pieces to a lighter rank hosting the same expert.
    Tie-breaks: heaviest rank = lowest index; pieces by (rows desc, expert asc);
    receivers by rank asc; a candidate replaces the best only if strictly better.
    """
    E, R, T = placement.logical, placement.ranks, tile
    if torch.is_tensor(counts):
        counts = counts.cpu().tolist()
    n = [0] * E
    for row in counts:
        for e in range(E):
            n[e] += int(row[e])
    rows: list[dict[int, int]] = [dict() for _ in range(R)]  # rank -> {e: rows}
    rep: dict[tuple[int, int], int] = {}
    for e in range(E):
        hosts = placement.host_ranks(e)
        for r, p in hosts:
            rows[r][e] = 0
            rep[(r, e)] = p
        if init == "even":
            reps = placement.replicas[e]
            k = len(reps)
            for j, p in enumerate(reps):
                rows[p // placement.local][e] += n[e] // k + (1 if j < n[e] % k else 0)
        elif init == "tile":
            k = len(hosts)
            tiles = (n[e] + T - 1) // T
            for j, (r, _) in enumerate(hosts):
                rows[r][e] = (tiles // k + (1 if j < tiles % k else 0)) * T
            if n[e] % T:
                r_last = hosts[(tiles - 1) % k][0]
                rows[r_last][e] -= T - n[e] % T
        else:
            raise ValueError(init)

    def load(r: int) -> int:
        return sum(_ceil(x, T) for x in rows[r].values())

    loads = [load(r) for r in range(R)]
    for _ in range(max_iter):
        r = max(range(R), key=lambda q: (loads[q], -q))
        cur = loads[r]
        best = None  # (new max of the pair, x, e, q)
        pieces = sorted(rows[r].items(), key=lambda kv: (-kv[1], kv[0]))[:top_pieces]
        for e, ne in pieces:
            if ne == 0:
                continue
            for q in range(R):
                if q == r or e not in rows[q] or loads[q] >= cur:
                    continue
                m, x = _best_move(ne, cur, loads[q], rows[q][e], T)
                if x and m < cur and (best is None or m < best[0]):
                    best = (m, x, e, q)
        if best is None:
            break
        _, x, e, q = best
        rows[r][e] -= x
        rows[q][e] += x
        loads[r], loads[q] = load(r), load(q)
    quotas = [0] * placement.physical
    for (r, e), p in rep.items():
        quotas[p] = rows[r][e]
    return quotas


def _mapping_tables(counts, quotas, placement: Placement, diagonal: bool):
    """Per (source s, expert e): local take, remote base ordinal; per e the replica
    prefix (over remaining quotas) in physical-id order.  Pure python, used by both
    the CPU reference and (as tensors) the device mapper."""
    E, R = placement.logical, placement.ranks
    take = [[0] * E for _ in range(R)]
    local_rep = [[-1] * E for _ in range(R)]
    remaining = list(quotas)
    if diagonal:
        for e in range(E):
            for r, p in placement.host_ranks(e):
                t = min(int(counts[r][e]), quotas[p])
                take[r][e] = t
                local_rep[r][e] = p
                remaining[p] -= t
    base = [[0] * E for _ in range(R)]
    for e in range(E):
        acc = 0
        for s in range(R):
            base[s][e] = acc
            acc += int(counts[s][e]) - take[s][e]
    return take, local_rep, base, remaining


def _counts_of(topk_logical: torch.Tensor, E: int) -> list[list[int]]:
    R = topk_logical.shape[0]
    out = []
    for s in range(R):
        t = topk_logical[s].reshape(-1).long()
        out.append(torch.bincount(t[t >= 0], minlength=E)[:E].tolist())
    return out


def split_topk(
    topk_logical: torch.Tensor,
    quotas,
    placement: Placement,
    *,
    diagonal: bool = False,
    interleave: bool = False,
    counts=None,
) -> torch.Tensor:
    """CPU reference mapping of ``[ranks, rows, k]`` logical ids to physical ids.
    Negative ids (padding) are kept.  Straight loop over routes in (s, row, k) order."""
    R, rows, K = topk_logical.shape
    E = placement.logical
    if counts is None:
        counts = _counts_of(topk_logical, E)
    take, local_rep, base, remaining = _mapping_tables(
        counts, list(quotas), placement, diagonal
    )
    prefix = []
    for e in range(E):
        acc, pre = 0, []
        for p in placement.replicas[e]:
            acc += remaining[p]
            pre.append((acc, p))
        prefix.append(pre)
    src = topk_logical.tolist()
    out = [[[0] * K for _ in range(rows)] for _ in range(R)]
    for s in range(R):
        seen = [0] * E
        for i in range(rows):
            for k in range(K):
                e = src[s][i][k]
                if e < 0:
                    out[s][i][k] = e
                    continue
                pos = seen[e]
                seen[e] += 1
                if pos < take[s][e]:
                    out[s][i][k] = local_rep[s][e]
                    continue
                p_rem = pos - take[s][e]
                if interleave:
                    o = sum(
                        min(int(counts[t][e]) - take[t][e], p_rem) for t in range(R)
                    ) + sum(
                        1 for t in range(s) if int(counts[t][e]) - take[t][e] > p_rem
                    )
                else:
                    o = base[s][e] + p_rem
                for end, p in prefix[e]:
                    if o < end:
                        out[s][i][k] = p
                        break
                else:  # pragma: no cover - quotas cover every route
                    raise AssertionError("ordinal beyond quotas")
    return torch.tensor(out, dtype=topk_logical.dtype)


def rr_topk(topk_logical: torch.Tensor, placement: Placement) -> torch.Tensor:
    """Static fixture rule (tools/eplb/make_eplb_fixture.py ``rr``): source s, row t
    -> replica ``(t + s) % n_replicas`` of the expert's list."""
    R, rows, K = topk_logical.shape
    width = max(len(v) for v in placement.replicas)
    table = torch.full((placement.logical, width), -1, dtype=torch.int64)
    nrep = torch.tensor([len(v) for v in placement.replicas], dtype=torch.int64)
    for e, v in enumerate(placement.replicas):
        table[e, : len(v)] = torch.tensor(v)
    t_idx = torch.arange(rows)[:, None].expand(rows, K)
    out = torch.empty_like(topk_logical)
    for s in range(R):
        e = topk_logical[s].long().clamp_min(0)
        phys = table[e, (t_idx + s) % nrep[e]]
        out[s] = torch.where(topk_logical[s] >= 0, phys, topk_logical[s].long()).to(
            topk_logical.dtype
        )
    return out


def padded_rows_per_rank(
    topk_physical: torch.Tensor, n_phys: int, ranks: int, tile: int
) -> list[int]:
    """Tile-padded rows per owner rank (what the kernel's tile cursor walks)."""
    t = topk_physical.reshape(-1).long()
    pe = torch.bincount(t[t >= 0], minlength=n_phys)[:n_phys]
    return ((pe + tile - 1) // tile * tile).view(ranks, -1).sum(1).tolist()


class SplitMapper:
    """Device mapping of one source rank's ``[rows, k]`` logical ids, given the plan.

    ``tables(counts, quotas)`` builds the small per-batch tables on the host (the P1b
    kernel computes them on the device); ``__call__`` maps this rank's routes with a
    stable sort by expert, so it matches :func:`split_topk` exactly.
    """

    def __init__(self, placement: Placement, device) -> None:
        self.placement = placement
        self.device = torch.device(device)
        E = placement.logical
        self.width = max(len(v) for v in placement.replicas)
        rep = torch.full((E, self.width), -1, dtype=torch.int64)
        for e, v in enumerate(placement.replicas):
            rep[e, : len(v)] = torch.tensor(v)
        self.rep = rep.to(self.device)

    def tables(
        self,
        counts,
        quotas,
        rank: int,
        *,
        diagonal: bool = False,
        interleave: bool = False,
    ):
        P, E = self.placement, self.placement.logical
        take, local_rep, base, remaining = _mapping_tables(
            counts, list(quotas), P, diagonal
        )
        ends = torch.zeros((E, self.width), dtype=torch.int64)
        for e, v in enumerate(P.replicas):
            acc = 0
            for j, p in enumerate(v):
                acc += remaining[p]
                ends[e, j] = acc
            ends[e, len(v) :] = acc  # never selected: ordinal < total
        t = lambda x: torch.tensor(x, dtype=torch.int64)  # noqa: E731
        remote = None
        if interleave:  # [E, ranks] remote routes per source
            remote = (t(counts) - t(take)).T.contiguous().to(self.device)
        return (
            t(take[rank]).to(self.device),
            t(local_rep[rank]).to(self.device),
            t(base[rank]).to(self.device),
            ends.to(self.device),
            remote,
            rank,
        )

    def __call__(self, topk: torch.Tensor, tables) -> torch.Tensor:
        take, local_rep, base, ends, remote, rank = tables
        flat = topk.reshape(-1)
        valid = flat >= 0
        e = flat.long().clamp_min(0)
        # position of each route among this source's routes of the same expert
        key = torch.where(valid, e, torch.full_like(e, self.placement.logical))
        order = torch.sort(key, stable=True).indices
        sk = key[order]
        n = flat.numel()
        idx = torch.arange(n, device=flat.device)
        start = torch.zeros(n, dtype=torch.int64, device=flat.device)
        boundary = torch.ones(n, dtype=torch.bool, device=flat.device)
        boundary[1:] = sk[1:] != sk[:-1]
        start = torch.cummax(torch.where(boundary, idx, start), 0).values
        pos = torch.empty_like(idx)
        pos[order] = idx - start
        tk = take[e]
        p_rem = pos - tk
        if remote is None:
            o = base[e] + p_rem
        else:
            m = remote[e]  # [n, ranks]
            pr = p_rem.clamp_min(0).unsqueeze(1)
            o = torch.minimum(m, pr).sum(1) + (m[:, :rank] > pr).sum(1)
        j = (ends[e] <= o.unsqueeze(1)).sum(1)
        phys = self.rep[e, j.clamp_max(self.width - 1)]
        phys = torch.where(pos < tk, local_rep[e], phys)
        out = torch.where(valid, phys, flat.long())
        return out.to(topk.dtype).view_as(topk)


# ----------------------------------------------------------------------------------------------
# In-kernel dynamic EPLB (switch-on build, ``dynamic_eplb=True``): the placement blob.
# ----------------------------------------------------------------------------------------------
DYN_BLOB_MAGIC = 0x44594E45
DYN_BLOB_VERSION = 2


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def dyn_eplb_defaults(world_size: int) -> dict[str, int]:
    """Measured defaults of :meth:`set_dyn_placement` (dyneplb/p2 FI paired bench, held-out placement, both variants,
    2026-10-15): the split beats SGLang static at every R >= 512 measured (EP4 / EP8), and a gated call costs the
    same as the static kernel, so ``min_rows`` stays 512; the affinity map wins at EP8 from R2048 on and loses
    0.3-1 % at EP4 and at EP8 R <= 1024, so ``affinity`` = 2048 rows at world_size >= 8, else off."""

    return {"min_rows": 512, "affinity": 2048 if world_size >= 8 else 0, "early_ctas": 0}


def dyn_placement_blob(
    C: dict[str, int],
    physical_to_logical: list[int],
    *,
    logical_experts: int,
    world_size: int,
    min_rows: int = 512,
    wait_ns: int = 50_000,
    count_lsa: Any = False,
    affinity: bool | int = True,
    early_ctas: int = 0,
) -> list[int]:
    """int32 words of the ``dyn_placement`` kernel argument (layout = the generator's DYN_BLOB_*).

    ``C`` are the instance constants of the switch-on build.  ``count_lsa``: per rank r a bitmask of the peers whose
    count rows go over LSA (pair reachable both ways), or a bool for all/none; ``affinity``: owner-affinity token mapping
    inside the dilv allotment, True / False, or an int = from this many collective rows on (0 = never);
    ``early_ctas``: protocol CTAs requested when the shared expert is fused (the CTAs above stream the shared-expert
    tiles under the dyn prologue; 0 = off; builds without DYN_BLOB_EARLY ignore it).  The kernel
    falls back to the input ids when the blob does not match its build (magic / version / size).
    """

    n_phys = len(physical_to_logical)
    P, E = int(C["PHYSICAL_RANKS"]), int(C["ROUTED_EXPERTS"])
    local = n_phys // world_size
    mh_max = int(C["DYN_MH_MAX"])
    if n_phys != world_size * int(C["LOCAL_EXPERTS"]) or n_phys > E:
        raise ValueError("dyn_placement_blob: placement does not match the kernel geometry")
    aff_rows = 1 if affinity is True else 0 if affinity is False else int(affinity)
    if not (0 <= wait_ns < 2**63) or min_rows < 0 or not (0 <= aff_rows < 2**31) or not (0 <= early_ctas < 2**31):
        raise ValueError("dyn_placement_blob: settings out of range")
    e_log = int(logical_experts)
    hosts = [sorted({p // local for p in range(n_phys) if physical_to_logical[p] == e}) for e in range(e_log)]
    if any(not h for h in hosts):
        raise ValueError("dyn_placement_blob: a logical expert has no physical replica")
    mh = [e for e in range(e_log) if len(hosts[e]) > 1]
    if len(mh) > mh_max:
        raise ValueError(f"dyn_placement_blob: {len(mh)} multi-host experts > DYN_MH_MAX {mh_max}")
    blob = [DYN_BLOB_MAGIC, DYN_BLOB_VERSION, e_log, n_phys, min_rows, _i32(wait_ns), _i32(wait_ns >> 32), len(mh)]
    blob += [int(v) for v in physical_to_logical]
    blob += [0] * (int(C["DYN_BLOB_REP"]) - len(blob))
    rep = [-1] * (P * E)
    for p in range(n_phys):
        r, e = p // local, physical_to_logical[p]
        if rep[r * E + e] < 0:
            rep[r * E + e] = p
    blob += rep
    mh_idx = [-1] * E
    for i, e in enumerate(mh):
        mh_idx[e] = i
    mh_rep = [-1] * (mh_max * P)
    mh_mask = [0] * mh_max
    for i, e in enumerate(mh):
        for r in hosts[e]:
            mh_rep[i * P + r] = rep[r * E + e]
            mh_mask[i] |= 1 << r
    shost = [-1] * E
    for e in range(e_log):
        if len(hosts[e]) == 1:
            shost[e] = hosts[e][0]
    if isinstance(count_lsa, (list, tuple)):
        masks = [int(m) for m in count_lsa]
    else:
        masks = [((1 << world_size) - 1) if count_lsa else 0] * world_size
    masks += [0] * (P - len(masks))
    blob += mh_idx + mh_rep + shost + masks + [aff_rows] + mh_mask
    if "DYN_BLOB_EARLY" in C:
        blob += [int(early_ctas)]
    layout = {"DYN_BLOB_MHIDX": int(C["DYN_BLOB_REP"]) + P * E}
    layout["DYN_BLOB_MHREP"] = layout["DYN_BLOB_MHIDX"] + E
    layout["DYN_BLOB_SHOST"] = layout["DYN_BLOB_MHREP"] + mh_max * P
    layout["DYN_BLOB_CNT_LSA"] = layout["DYN_BLOB_SHOST"] + E
    layout["DYN_BLOB_AFF"] = layout["DYN_BLOB_CNT_LSA"] + P
    layout["DYN_BLOB_MHMASK"] = layout["DYN_BLOB_AFF"] + 1
    if "DYN_BLOB_EARLY" in C:
        layout["DYN_BLOB_EARLY"] = layout["DYN_BLOB_MHMASK"] + mh_max
    for name, at in layout.items():
        if name in C and int(C[name]) != at:
            raise ValueError(f"dyn_placement_blob: layout mismatch at {name} ({C[name]} != {at})")
    words = int(C["DYN_BLOB_WORDS"])
    if len(blob) > words:
        raise ValueError("dyn_placement_blob: blob larger than DYN_BLOB_WORDS")
    return blob + [0] * (words - len(blob))
