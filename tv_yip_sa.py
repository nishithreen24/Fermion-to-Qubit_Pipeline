#!/usr/bin/env python3
"""
tv_yip_sa.py -- Yip VarQA ground-state ENERGY via SA-sampled subspace
diagonalisation (selected-CI), for the 1D spinless t-V chain.

This replaces the ED "cheat" in tv_yip.py (which sampled from the exact |c|^2
with exact signs).  Here the ground-state energy is genuinely FOUND:

  1. SIMULATED ANNEALING is the sampler.  It anneals the classical diagonal
     Ising energy  E_diag(cfg) = V/4 sum_<ij> z_i z_j  in the fixed-N
     (half-filling) occupation space, with nearest-neighbour-swap moves, and
     collects the low-energy configurations it visits -> a seed support set.
  2. The seed set is CLOSED under one nearest-neighbour hop (repeatedly), so the
     off-diagonal XY coupling is represented in the subspace.
  3. H restricted to that support is diagonalised (Lanczos) -> E_var.
     As the support grows, E_var decreases monotonically toward the true E0.

The honest cost quantity is the SUPPORT SIZE |B| needed to reach a target
accuracy -- the classical analogue of Yip's sample count S.  For small L the
support saturates the sector and E_var == E0 exactly (checked against ED); for
large L (cluster) it is a genuine variational estimate.

Usage:
    python3 tv_yip_sa.py --L 8 --t 1.0 --V 2.0 --reads 200 --csv tv_yip_sa_L8.csv
"""
import argparse
import numpy as np

import tv_gs_common as C


# ---------------------------------------------------------------------------
# SA sampler in fixed-N occupation space (NN-swap moves) -> visited configs
# ---------------------------------------------------------------------------
def sa_collect(L, N, V, bonds, reads, sweeps, T_start, T_end, seed):
    """Run `reads` independent SA chains; return a set of visited config
    bitmasks (biased toward low diagonal energy)."""
    rng = np.random.default_rng(seed)
    visited = set()
    alpha = (T_end / T_start) ** (1.0 / max(sweeps, 1))

    for r in range(reads):
        # random half-filling start
        occ = np.zeros(L, dtype=np.int8)
        occ[rng.choice(L, N, replace=False)] = 1
        cfg = int(sum((1 << i) for i in range(L) if occ[i]))
        E = C.diag_energy(cfg, L, V, bonds)
        visited.add(cfg)
        T = T_start
        for _ in range(sweeps):
            # one sweep = L attempted NN swaps on differing bonds
            for _ in range(L):
                b = bonds[rng.integers(len(bonds))]
                i, j = b
                bi, bj = (cfg >> i) & 1, (cfg >> j) & 1
                if bi == bj:
                    continue
                new = cfg ^ (1 << i) ^ (1 << j)
                Enew = C.diag_energy(new, L, V, bonds)
                dE = Enew - E
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    cfg, E = new, Enew
                    visited.add(cfg)
            T *= alpha
    return visited


# ---------------------------------------------------------------------------
# hop-closure expansion: grow support and record E_var(|B|)
# ---------------------------------------------------------------------------
def closure_ladder(seed, L, t, V, bonds, max_size, n_levels):
    """Starting from `seed`, repeatedly add one-hop neighbours; diagonalise at a
    geometric ladder of support sizes.  Returns list of (size, E_var)."""
    frontier = set(seed)
    support = set(seed)
    # geometric checkpoints in support size
    checkpoints = sorted(set(int(x) for x in np.unique(
        np.geomspace(max(2, len(seed) // 4), max_size, n_levels).astype(int))))
    rows = []
    ci = 0

    def record(sz):
        # keep the MOST IMPORTANT configs first: rank the support by diagonal
        # energy (low -> the CDW core carrying the GS weight) so |B| is a
        # meaningful "how many configs do you need" axis, not bitmask order.
        ranked = sorted(support, key=lambda c: C.diag_energy(c, L, V, bonds))
        basis = ranked[:sz] if sz < len(ranked) else ranked
        Hsp = C.build_sparse_H(basis, L, t, V, bonds)
        rows.append((len(basis), C.lowest_eig(Hsp)))

    # expand by BFS hop-closure until we hit max_size or run dry
    while len(support) < max_size:
        new = set()
        for c in frontier:
            for nb in C.hop_neighbors(c, bonds):
                if nb not in support:
                    new.add(nb)
        if not new:
            break
        for nb in new:
            if len(support) >= max_size:
                break
            support.add(nb)
        frontier = new
        # record at checkpoints as we cross them
        while ci < len(checkpoints) and len(support) >= checkpoints[ci]:
            record(checkpoints[ci]); ci += 1
    # always record the final (largest) support
    record(len(support))
    # dedupe by size, keep the lowest energy at each size
    best = {}
    for sz, e in rows:
        if sz not in best or e < best[sz]:
            best[sz] = e
    return sorted(best.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--V", type=float, default=2.0)
    ap.add_argument("--reads", type=int, default=200, help="# SA chains (seeds)")
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--max_support", type=int, default=20000,
                    help="cap on subspace size (memory/eig cost)")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    L, t, V = args.L, args.t, args.V
    bonds = C.bonds_obc(L)
    N = C.half_filling_N(L)
    from math import comb
    sector_dim = comb(L, N)

    # ED reference only where feasible
    E0 = C.ed_reference(L, t, V) if L <= 16 else None

    T_start = max(4.0 * abs(V), 2.0)
    seed = sa_collect(L, N, V, bonds, args.reads, args.sweeps,
                      T_start, 1e-2, args.seed)

    max_size = min(args.max_support, sector_dim)
    ladder = closure_ladder(seed, L, t, V, bonds, max_size, args.levels)

    E_final = ladder[-1][1]
    size_final = ladder[-1][0]

    print(f"# t-V Yip-SA subspace:  L={L} t={t} V={V}  N={N}"
          f"  sector_dim={sector_dim}  |seed|={len(seed)}")
    if E0 is not None:
        print(f"# ED E0 = {E0:.6f}")
    print(f"#   {'|B|':>10} {'E_var':>12} {'|dE|':>12} {'B/sector':>10}")
    for sz, e in ladder:
        de = abs(e - E0) if E0 is not None else float('nan')
        print(f"#   {sz:>10} {e:>12.6f} {de:>12.2e} {sz/sector_dim:>10.4f}")
    tag = "EXACT" if (E0 is not None and abs(E_final - E0) < 1e-9) else "variational"
    print(f"# final: E_var={E_final:.6f}  |B|={size_final}"
          f"  ({size_final/sector_dim*100:.1f}% of sector)  [{tag}]")

    if args.csv:
        meta = (f"L={L} t={t} V={V} N={N} sector_dim={sector_dim} "
                f"E0_ed={E0 if E0 is not None else float('nan'):.6f} "
                f"E_var_final={E_final:.6f} seed_size={len(seed)}")
        rows = [(sz, e, (abs(e - E0) if E0 is not None else float('nan')),
                 sz / sector_dim) for sz, e in ladder]
        C.write_csv(args.csv, ["support", "E_var", "absdE", "B_over_sector"],
                    rows, meta)
        print(f"# wrote {args.csv}")


if __name__ == "__main__":
    main()
