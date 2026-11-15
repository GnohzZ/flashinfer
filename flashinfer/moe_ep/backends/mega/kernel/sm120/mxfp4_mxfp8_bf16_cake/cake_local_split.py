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

"""Source-local EPLB split (experimental, megamoe-unified notes/DESIGN-20261025-local-split.md).

With static EPLB a replicated logical expert has physical copies on several ranks.  Here every rank maps ITS OWN
routes of a replicated expert onto the copies from its own information only (no count exchange, no wait for the
other ranks): it predicts the step's global rows of every expert as

    p[e] = (alpha * ranks * c[e] + (256 - alpha) * prev[e]) >> 8          (alpha 256 = "own", 128 = "mix")

from its own route counts ``c`` and (optionally) the previous step's global counts ``prev``, plans the tile split of
the replicated experts over their holders with an integer water-fill (non-replicated experts load their owner), and
splits its own rows of each replicated expert in proportion to the plan (largest remainder), its own rank's copy
first.  The result is a physical topk for the unchanged static kernel; a replica's weights are byte copies of the
logical expert's, so the output equals the static run's.

Exact integer spec (the CUDA kernel and :func:`local_split_reference` agree bit for bit):
  1. c[e] = routes of this rank to e (negative ids = padding, skipped).
  2. p[e] as above; b[e] = ceil(p[e] / tile).
  3. L[r] = sum of b[e] over the non-replicated experts owned by r.
  4. Replicated experts in (b desc, expert asc) order; holders h_0..h_{k-1} = their ranks by ascending physical id:
     lambda = smallest integer with sum_j max(0, lambda - L[h_j]) >= b; x_j = max(0, lambda - L[h_j]); the excess
     sum x - b is taken back one tile each from the holders with x_j > 0, highest j first; L[h_j] += x_j.
  5. Own rows of the expert: tot = sum x; tot == 0 -> all to h_0; else q_j = floor(c x_j / tot) and the remainder
     goes one row each to the largest (c x_j mod tot), lower j first.
  6. Mapping order: this rank's copy first (if it holds one), then the other holders in j order; the route with
     ordinal o among this rank's routes of the expert (flattened (row, k) order) goes to the copy whose cumulative
     quota interval holds o.  Non-replicated experts -> their only copy.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch

__all__ = ["LocalSplitTables", "LocalSplit", "local_split_reference", "stamp", "spin_until", "MODES"]

MODES = {"own": 256, "mix": 128, "prev": 0}
NR_MAX, H_MAX, R_MAX, E_MAX = 64, 16, 16, 512


class LocalSplitTables:
    """Per-layer host tables from a ``physical_to_logical`` list (rank-major, ``ranks`` equal blocks; -1 = an unused
    slot, never routed to)."""

    def __init__(self, physical_to_logical, ranks: int) -> None:
        p2l = [int(x) for x in physical_to_logical]
        if len(p2l) % ranks:
            raise ValueError(f"{len(p2l)} physical experts on {ranks} ranks")
        if ranks > R_MAX:
            raise ValueError(f"ranks {ranks} > {R_MAX}")
        self.ranks, self.physical, self.local = ranks, len(p2l), len(p2l) // ranks
        self.logical = max(p2l) + 1
        if self.logical > E_MAX:
            raise ValueError(f"{self.logical} logical experts > {E_MAX}")
        copies: list[list[int]] = [[] for _ in range(self.logical)]
        for p, e in enumerate(p2l):
            if e >= 0:
                copies[e].append(p)
        if any(not c for c in copies):
            raise ValueError("every logical expert needs a physical copy")
        self.owner_phys = [c[0] for c in copies]
        self.rep_slot = [-1] * self.logical
        self.rep_e, self.rep_rank, self.rep_phys = [], [], []
        for e, c in enumerate(copies):
            ranks_seen, hr, hp = set(), [], []
            for p in c:  # one copy per rank (the lowest physical id)
                r = p // self.local
                if r not in ranks_seen:
                    ranks_seen.add(r)
                    hr.append(r)
                    hp.append(p)
            if len(hr) > 1:
                self.rep_slot[e] = len(self.rep_e)
                self.rep_e.append(e)
                self.rep_rank.append(hr)
                self.rep_phys.append(hp)
        if len(self.rep_e) > NR_MAX:
            raise ValueError(f"{len(self.rep_e)} replicated experts > {NR_MAX}")
        self.nrep = len(self.rep_e)

    def device(self, device) -> dict:
        def pad2(rows):
            out = torch.full((NR_MAX, H_MAX), -1, dtype=torch.int32)
            for i, r in enumerate(rows):
                out[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
            return out

        rep_e = torch.full((NR_MAX,), -1, dtype=torch.int32)
        rep_k = torch.zeros((NR_MAX,), dtype=torch.int32)
        for i, e in enumerate(self.rep_e):
            rep_e[i], rep_k[i] = e, len(self.rep_rank[i])
        t = {
            "owner_phys": torch.tensor(self.owner_phys, dtype=torch.int32),
            "rep_slot": torch.tensor(self.rep_slot, dtype=torch.int32),
            "rep_e": rep_e,
            "rep_k": rep_k,
            "rep_rank": pad2(self.rep_rank),
            "rep_phys": pad2(self.rep_phys),
        }
        return {k: v.to(device) for k, v in t.items()}


def local_split_reference(ids, tables: LocalSplitTables, rank: int, *, alpha: int = 256, prev=None, tile: int = 128):
    """Host reference of the spec.  ids: logical topk (any shape, int).  -> (physical ids like ids, counts[E])."""
    flat = [int(x) for x in torch.as_tensor(ids).reshape(-1).tolist()]
    E, ranks, local = tables.logical, tables.ranks, tables.local
    c = [0] * E
    for e in flat:
        if e >= 0:
            c[e] += 1
    pv = [0] * E if prev is None else [int(x) for x in torch.as_tensor(prev).reshape(-1).tolist()]

    def blocks(e):
        p = (alpha * ranks * c[e] + (256 - alpha) * pv[e]) >> 8
        return (p + tile - 1) // tile

    L = [0] * ranks
    for e in range(E):
        if tables.rep_slot[e] < 0:
            L[tables.owner_phys[e] // local] += blocks(e)
    bb = [blocks(e) for e in tables.rep_e]
    order = sorted(range(tables.nrep), key=lambda i: (-bb[i], tables.rep_e[i]))
    plan = {}  # slot -> [(phys, quota)] in mapping order
    for i in order:
        e, hr, hp = tables.rep_e[i], tables.rep_rank[i], tables.rep_phys[i]
        k, b = len(hr), bb[i]
        x = [0] * k
        if b > 0:
            lo = min(L[h] for h in hr)
            hi = lo + b
            while lo < hi:  # smallest lambda with fill >= b
                mid = (lo + hi) // 2
                if sum(max(0, mid - L[h]) for h in hr) >= b:
                    hi = mid
                else:
                    lo = mid + 1
            x = [max(0, lo - L[h]) for h in hr]
            excess = sum(x) - b
            for j in range(k - 1, -1, -1):
                if excess == 0:
                    break
                if x[j] > 0:
                    x[j] -= 1
                    excess -= 1
            for j, h in enumerate(hr):
                L[h] += x[j]
        ce, tot = c[e], sum(x)
        if tot == 0:
            q = [ce] + [0] * (k - 1)
        else:
            q = [ce * xj // tot for xj in x]
            rem = ce - sum(q)
            rest = sorted(range(k), key=lambda j: (-(ce * x[j] % tot), j))
            for j in rest[:rem]:
                q[j] += 1
        mo = ([hr.index(rank)] if rank in hr else []) + [j for j in range(k) if hr[j] != rank]
        plan[i] = [(hp[j], q[j]) for j in mo]
    out, seen = [], [0] * tables.nrep
    for e in flat:
        if e < 0:
            out.append(e)
            continue
        s = tables.rep_slot[e]
        if s < 0:
            out.append(tables.owner_phys[e])
            continue
        o, acc = seen[s], 0
        seen[s] += 1
        for p, qj in plan[s]:
            acc += qj
            if o < acc:
                out.append(p)
                break
        else:
            raise AssertionError("ordinal beyond the quotas")
    t = torch.as_tensor(ids)
    return torch.tensor(out, dtype=torch.int64).reshape(t.shape), torch.tensor(c, dtype=torch.int32)


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define NT 1024
#define NW (NT / 32)  // NW == 32: the warp-offset scan maps one lane per warp
#define E_MAX 512
#define NR_MAX 64
#define H_MAX 16
#define R_MAX 16
#define IDC_MAX 24576
#define V 4  // routes per thread per mapping round

// Phases (one CTA, NT threads): stage the tables in SMEM; SMEM atomic counts; per-expert tile predictions and owner
// loads; rank sort of the replicated experts (b desc, e asc); warp 0 water-fills them one by one, lanes = holders
// (lambda = min over holders j of ceil((b + sum_{L_i <= L_j} L_i) / #{L_i <= L_j}) is the smallest integer level whose
// fill reaches b: every candidate's fill reaches b, the one of the level's own segment is exact); per-expert quotas
// (one warp per expert); then rounds of NT * V routes: routes of the same replicated expert inside a warp are found
// with bit-sliced ballots (cheaper than __match_any_sync here), a warp-shuffle scan over the warps gives the warp's
// offset and a per-expert carry links the rounds -> ordinals in route order (deterministic).  Measured phase costs drove these choices (match_any, warp
// water-fill with per-expert shuffles and 64-bit / local-memory quota loops were 2-6 us each).
__global__ void __launch_bounds__(NT) local_split_kernel(
    const int64_t* __restrict__ ids, int64_t* __restrict__ out, int n, int* __restrict__ counts_out,
    const int* __restrict__ prev, const int* __restrict__ owner_phys, const int* __restrict__ rep_slot,
    const int* __restrict__ rep_e, const int* __restrict__ rep_k, const int* __restrict__ rep_rank,
    const int* __restrict__ rep_phys, int nrep, int rank, int ranks, int local, int alpha, int tile, int E,
    int cached) {
  extern __shared__ short idc[];  // expert ids of the routes (cached != 0): the mapping rounds skip global loads
  __shared__ int cnt[E_MAX];
  __shared__ int bk[E_MAX];
  __shared__ short slot_of[E_MAX];
  __shared__ int ophys[E_MAX];
  __shared__ int L[R_MAX];
  __shared__ int s_e[NR_MAX], s_k[NR_MAX], w_i[NR_MAX], w_k[NR_MAX], w_b[NR_MAX];
  __shared__ int s_hr[NR_MAX][H_MAX], s_hp[NR_MAX][H_MAX], s_x[NR_MAX][H_MAX];
  __shared__ int bnd[NR_MAX][H_MAX];
  __shared__ int mph[NR_MAX][H_MAX];
  __shared__ int wcnt[NW][NR_MAX];  // per-warp counts of a round (only leaders write; they reset after use)
  __shared__ int woff[NW][NR_MAX];  // per-warp exclusive offsets of a round
  __shared__ int carry[NR_MAX];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += NT) {
    cnt[e] = 0;
    slot_of[e] = (short)rep_slot[e];
    ophys[e] = owner_phys[e];
  }
  for (int z = tid; z < NR_MAX * H_MAX; z += NT) {
    const int i = z / H_MAX, j = z % H_MAX;
    s_hr[i][j] = rep_rank[z];
    s_hp[i][j] = rep_phys[z];
    s_x[i][j] = 0;
  }
  if (tid < NR_MAX) { s_e[tid] = rep_e[tid]; s_k[tid] = rep_k[tid]; carry[tid] = 0; }
  if (tid < R_MAX) L[tid] = 0;
  __syncthreads();
  for (int base = 0; base < n; base += NT) {
    const int i = base + tid;
    const int e = i < n ? (int)ids[i] : -1;
    if (cached && i < n) idc[i] = (short)e;
    if (e >= 0) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  for (int e = tid; e < E; e += NT) {
    if (counts_out) counts_out[e] = cnt[e];
    const int p = (int)(((long long)alpha * ranks * cnt[e] + (long long)(256 - alpha) * (prev ? prev[e] : 0)) >> 8);
    const int b = (p + tile - 1) / tile;
    bk[e] = b;
    if (slot_of[e] < 0 && b) atomicAdd(&L[ophys[e] / local], b);
  }
  __syncthreads();
  if (tid < nrep) {  // rank sort: (b desc, e asc); the water-fill reads its per-expert scalars in sorted order
    const int e = s_e[tid], b = bk[e];
    int r = 0;
    for (int j = 0; j < nrep; ++j) {
      const int f = s_e[j], c = bk[f];
      r += (c > b) || (c == b && f < e);
    }
    w_i[r] = tid;
    w_k[r] = s_k[tid];
    w_b[r] = b;
  }
  __syncthreads();
  if (warp == 0) {  // sequential water-fill over the replicated experts; lanes = holders
    // Every lane reads all k holder loads from SMEM itself (independent loads) instead of k dependent shuffles;
    // measured: thread-0 scalar loops ~0.8 us / expert, shuffle-chain warp version ~0.3 us / expert.
    for (int oi = 0; oi < nrep; ++oi) {
      const int i = w_i[oi], k = w_k[oi], b = w_b[oi];
      if (b == 0) continue;
      const bool act = lane < k;
      int mj = 0, sj = 0, Lj = 0, hj = 0;
      if (act) {
        hj = s_hr[i][lane];
        Lj = L[hj];
        for (int t = 0; t < k; ++t) {
          const int Lt = L[s_hr[i][t]];
          if (Lt <= Lj) { ++mj; sj += Lt; }
        }
      }
      const int lam = __reduce_min_sync(0xffffffffu, act ? (b + sj + mj - 1) / mj : 0x7fffffff);
      int x = act ? max(0, lam - Lj) : 0;
      const int excess = (int)__reduce_add_sync(0xffffffffu, (unsigned)x) - b;
      const unsigned pos = __ballot_sync(0xffffffffu, x > 0);
      if (x > 0 && __popc(pos & ~((2u << lane) - 1u)) < excess) --x;
      __syncwarp();
      if (act) { s_x[i][lane] = x; L[hj] = Lj + x; }
      __syncwarp();
    }
  }
  __syncthreads();
  for (int i = warp; i < nrep; i += NW) {  // this rank's row quotas of expert i (lanes = holders), mapping order
    const int k = s_k[i];
    const bool act = lane < k;
    const unsigned ce = (unsigned)cnt[s_e[i]];
    const unsigned x = act ? (unsigned)s_x[i][lane] : 0u;
    const int hr = act ? s_hr[i][lane] : -1;
    const unsigned tot = __reduce_add_sync(0xffffffffu, x);
    unsigned q, md = 0;
    if (tot == 0) {
      q = lane == 0 ? ce : 0u;
    } else {
      const unsigned v = ce * x;  // ce <= rows, x <= tiles: < 2^32
      q = v / tot;
      md = v - q * tot;
      const unsigned rem = ce - __reduce_add_sync(0xffffffffu, q);
      unsigned r = 0;  // +1 for the rem largest remainders (ties: lower j)
      for (int t = 0; t < k; ++t) {
        const unsigned mt = __shfl_sync(0xffffffffu, md, t);
        r += (mt > md) || (mt == md && t < lane);
      }
      if (act && r < rem) ++q;
    }
    const unsigned lt = (1u << lane) - 1u;
    const unsigned own = __ballot_sync(0xffffffffu, act && hr == rank);
    const unsigned oth = __ballot_sync(0xffffffffu, act && hr != rank);
    const int mpos = hr == rank ? 0 : (own != 0) + __popc(oth & lt);
    int acc = 0;
    for (int t = 0; t < k; ++t) {
      const int pt = __shfl_sync(0xffffffffu, mpos, t);
      const unsigned qt = __shfl_sync(0xffffffffu, q, t);
      if (pt <= mpos) acc += (int)qt;
    }
    if (act) { bnd[i][mpos] = acc; mph[i][mpos] = s_hp[i][lane]; }
  }
  for (int z = tid; z < NW * NR_MAX; z += NT) (&wcnt[0][0])[z] = 0;
  __syncthreads();
  // Mapping rounds of NT * V routes, thread t owns routes base + t*V .. +V-1 (route order = (warp, lane, v)).
  // Same-slot lanes per (route v, other route u) from 7 ballots of every u (bit-sliced equality); measured ~0.9 us
  // per round, so V = 4 cuts the rounds 4x.
  for (int base = 0; base < n; base += NT * V) {
    int sv[V], ev[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
      const int i = base + tid * V + v;
      ev[v] = i < n ? (cached ? (int)idc[i] : (int)ids[i]) : -1;
      sv[v] = ev[v] >= 0 ? (int)slot_of[ev[v]] : -1;
    }
    unsigned bal[V][7];
#pragma unroll
    for (int u = 0; u < V; ++u)
#pragma unroll
      for (int bit = 0; bit < 7; ++bit) bal[u][bit] = __ballot_sync(0xffffffffu, ((sv[u] + 1) >> bit) & 1);
    const unsigned lt = (1u << lane) - 1u;
    int rk[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
      int r = 0, tot = 0;
#pragma unroll
      for (int u = 0; u < V; ++u) {
        unsigned m = 0xffffffffu;
#pragma unroll
        for (int bit = 0; bit < 7; ++bit) m &= (((sv[v] + 1) >> bit) & 1) ? bal[u][bit] : ~bal[u][bit];
        r += __popc(m & lt) + (u < v && sv[u] == sv[v]);
        tot += __popc(m);
      }
      rk[v] = r;
      if (sv[v] >= 0 && r == 0) wcnt[warp][sv[v]] = tot;
    }
    __syncthreads();
    for (int sl = warp; sl < nrep; sl += NW) {  // exclusive scan over the warps (lane = warp index)
      const int v = wcnt[lane][sl];
      int inc = v;
      for (int d = 1; d < 32; d <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, inc, d);
        if (lane >= d) inc += t;
      }
      const int c0 = carry[sl];
      woff[lane][sl] = c0 + inc - v;
      __syncwarp();
      if (lane == 31) carry[sl] = c0 + inc;
    }
    __syncthreads();
#pragma unroll
    for (int v = 0; v < V; ++v) {
      const int i = base + tid * V + v, s = sv[v], e = ev[v];
      if (s >= 0 && rk[v] == 0) wcnt[warp][s] = 0;  // reset for the next round (only leaders wrote)
      if (i >= n) continue;
      if (e < 0) {
        out[i] = ids[i];
      } else if (s < 0) {
        out[i] = ophys[e];
      } else {
        const int o = woff[warp][s] + rk[v];
        int p = -1;
        for (int m = 0; m < s_k[s]; ++m)
          if (o < bnd[s][m]) { p = mph[s][m]; break; }
        out[i] = p;
      }
    }
    __syncthreads();
  }
}

// Diagnostics: buf[idx] = %globaltimer (ns; one clock for all GPUs of a host, as CUPTI traces use it).
__global__ void stamp_kernel(int64_t* buf, int64_t idx) {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  buf[idx] = t;
}

void stamp(torch::Tensor buf, int64_t idx) {
  TORCH_CHECK(buf.dtype() == torch::kInt64 && idx >= 0 && idx < buf.numel(), "stamp: buf / idx");
  const c10::cuda::CUDAGuard guard(buf.device());
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(buf.data_ptr<int64_t>(), idx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Diagnostics: the stream does not proceed before %globaltimer >= t_ns (replays recorded per-rank arrival times).
__global__ void spin_until_kernel(long long t_ns) {
  long long t;
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t < t_ns);
}

void spin_until(int64_t t_ns) {
  spin_until_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((long long)t_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void local_split(torch::Tensor ids, torch::Tensor out, torch::Tensor counts_out, torch::Tensor prev,
                 torch::Tensor owner_phys, torch::Tensor rep_slot, torch::Tensor rep_e, torch::Tensor rep_k,
                 torch::Tensor rep_rank, torch::Tensor rep_phys, int64_t nrep, int64_t rank, int64_t ranks,
                 int64_t local, int64_t alpha, int64_t tile) {
  TORCH_CHECK(ids.dtype() == torch::kInt64 && out.dtype() == torch::kInt64, "ids / out must be int64");
  TORCH_CHECK(ids.is_contiguous() && out.is_contiguous() && ids.numel() == out.numel(), "ids / out");
  TORCH_CHECK(ids.numel() < (1LL << 30), "too many routes");
  const int E = (int)owner_phys.numel();
  TORCH_CHECK(E <= E_MAX && nrep <= NR_MAX && ranks <= R_MAX, "table limits");
  const c10::cuda::CUDAGuard guard(ids.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  // id cache in dynamic SMEM up to IDC_MAX routes (rows 4096 x top-6 at 48 KB), global loads beyond
  const int n_ = (int)ids.numel();
  const int cached = n_ <= IDC_MAX;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(local_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        IDC_MAX * (int)sizeof(short)));
    attr = true;
  }
  local_split_kernel<<<1, NT, cached ? n_ * sizeof(short) : 0, stream>>>(
      ids.data_ptr<int64_t>(), out.data_ptr<int64_t>(), (int)ids.numel(),
      counts_out.numel() ? counts_out.data_ptr<int>() : nullptr, prev.numel() ? prev.data_ptr<int>() : nullptr,
      owner_phys.data_ptr<int>(), rep_slot.data_ptr<int>(), rep_e.data_ptr<int>(), rep_k.data_ptr<int>(),
      rep_rank.data_ptr<int>(), rep_phys.data_ptr<int>(), (int)nrep, (int)rank, (int)ranks, (int)local,
      (int)alpha, (int)tile, E, cached);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_DECL = (
    "void local_split(torch::Tensor ids, torch::Tensor out, torch::Tensor counts_out, torch::Tensor prev, "
    "torch::Tensor owner_phys, torch::Tensor rep_slot, torch::Tensor rep_e, torch::Tensor rep_k, "
    "torch::Tensor rep_rank, torch::Tensor rep_phys, int64_t nrep, int64_t rank, int64_t ranks, int64_t local, "
    "int64_t alpha, int64_t tile);\n"
    "void stamp(torch::Tensor buf, int64_t idx);\n"
    "void spin_until(int64_t t_ns);"
)
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
        base = os.environ.get("FLASHINFER_WORKSPACE_BASE") or os.path.expanduser("~/.cache")
        build = os.path.join(base, "cake_local_split", tag)
        os.makedirs(build, exist_ok=True)
        _MODULE = load_inline(
            name=f"cake_local_split_{tag}",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["local_split", "stamp", "spin_until"],
            build_directory=build,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _MODULE


def stamp(buf: torch.Tensor, idx: int) -> None:
    """Diagnostics: ``buf[idx] = %globaltimer`` (ns) on the current stream."""
    _module().stamp(buf, idx)


def spin_until(t_ns: int) -> None:
    """Diagnostics: the current stream waits on the GPU until ``%globaltimer >= t_ns``."""
    _module().spin_until(int(t_ns))


class LocalSplit:
    """One layer's device tables + launcher.  ``__call__(ids)`` maps a logical topk (int64, any shape) to physical
    ids on the current stream; ``counts_out`` (int32 [E], optional) receives this rank's route counts; ``prev``
    (int32 [E]) is the previous step's global counts for alpha < 256."""

    def __init__(self, physical_to_logical, ranks: int, rank: int, device, *, mode: str = "own", tile: int = 128):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: expected one of {sorted(MODES)}")
        self.tables = LocalSplitTables(physical_to_logical, ranks)
        self.rank, self.alpha, self.tile = rank, MODES[mode], tile
        self.dev = self.tables.device(device)
        self._empty = torch.empty(0, dtype=torch.int32, device=device)

    def __call__(self, ids: torch.Tensor, *, prev: Optional[torch.Tensor] = None,
                 counts_out: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None,
                 tile: Optional[int] = None) -> torch.Tensor:
        ids = ids.contiguous()
        if out is None:
            out = torch.empty_like(ids)
        if self.alpha < 256 and prev is None:
            raise ValueError("mode with alpha < 256 needs prev")
        d, t = self.dev, self.tables
        _module().local_split(
            ids.view(-1), out.view(-1), self._empty if counts_out is None else counts_out,
            self._empty if prev is None else prev, d["owner_phys"], d["rep_slot"], d["rep_e"], d["rep_k"],
            d["rep_rank"], d["rep_phys"], t.nrep, self.rank, t.ranks, t.local, self.alpha, tile or self.tile)
        return out
