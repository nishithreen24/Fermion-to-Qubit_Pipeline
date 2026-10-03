"""
run_siam.py — XBK ground-state pipeline for the SIAM, swept over N_bath AND U.

Parallelism strategy
--------------------
On a single 8-core machine the hard speedup ceiling is the core count, and
MPI gives no advantage over multiprocessing (MPI only helps across multiple
nodes).  The key to approaching the ceiling is a *balanced* task pool: we
flatten the entire (U, N_bath, sector, p) cross-product into one global list
and feed it to a ProcessPoolExecutor.  A U-sweep makes this pool large and
well-balanced, so all cores stay busy instead of waiting on one slow sector.

Each task is one XBK/Dinkelbach/SA solve (the heavy budget is kept as-is).
Results are reduced by min over (sector, p) for each (U, N_bath).

Usage:
    python run_siam.py --baths 2 3 4 5 --U 0 2 4 6 8 --r 2 --workers 8
"""
import argparse
import math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xbk_pipeline as xp


def _solve(task):
    """One (U,L,sector,p) XBK solve. task carries everything needed; returns
    ((U,L), E0).  Kept top-level and picklable for ProcessPoolExecutor."""
    U, L, si, p, terms, n, r, seed, penalty, const = task
    if n == 0:
        return ((U, L), const)
    H_is = xp.build_xbk_ising(terms, n, r, p)
    C_is = xp.build_counting(n, r, p)
    if not C_is:
        return ((U, L), math.inf)
    E0 = xp.dinkelbach(H_is, C_is, n * r, penalty, seed, quadratic=False)
    return ((U, L), E0)


def build_tasks(baths, U_values, r, seed=3, penalty=12.0):
    """Flatten every (U, N_bath, sector, p) into one balanced task list."""
    tasks = []
    for U in U_values:
        for L in baths:
            H_bk = xp.bk_transform(xp.siam(L, U=U))
            for si, sec in enumerate(xp.z2_taper(H_bk)):
                terms, n = xp._qubitop_to_terms(sec)
                if n == 0:
                    const = float(np.real(sec.terms.get((), 0.0)))
                    tasks.append((U, L, si, 0, (), 0, r,
                                  seed, penalty, const))
                    continue
                for p in range(math.ceil(r / 2) + 1):
                    tasks.append((U, L, si, p, terms, n, r,
                                  seed + si * 101 + p * 37, penalty, 0.0))
    return tasks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baths", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--U", type=float, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--csv", default="siam_xbk_sweep.csv")
    args = ap.parse_args()

    print(f"SIAM sweep  r={args.r}  baths={args.baths}  U={args.U}", flush=True)
    tasks = build_tasks(args.baths, args.U, args.r, seed=args.seed)
    print(f"{len(tasks)} total tasks across "
          f"{len(args.U)}xU * {len(args.baths)}xN_bath", flush=True)

    t0 = time.time()
    best = defaultdict(lambda: math.inf)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for (key, E0) in ex.map(_solve, tasks):
            best[key] = min(best[key], E0)
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                print(f"  ... {done}/{len(tasks)} tasks "
                      f"({time.time()-t0:.0f}s)", flush=True)
    wall = time.time() - t0

    # exact references
    exact = {}
    if not args.no_exact:
        for U in args.U:
            for L in args.baths:
                exact[(U, L)] = xp.exact_ground_energy(xp.siam(L, U=U))

    print(f"\nDONE in {wall:.0f}s\n", flush=True)
    print(f"{'U':>5} {'N_bath':>6} {'E0_XBK':>10} {'E0_exact':>10} {'err':>8}",
          flush=True)
    rows = []
    for U in args.U:
        for L in args.baths:
            e = best[(U, L)]
            ex_e = exact.get((U, L))
            err = "-" if ex_e is None else f"{abs(e-ex_e):.4f}"
            exs = "-" if ex_e is None else f"{ex_e:.4f}"
            print(f"{U:>5} {L:>6} {e:>10.4f} {exs:>10} {err:>8}", flush=True)
            rows.append((U, L, e, ex_e))

    with open(args.csv, "w") as fh:
        fh.write("U,N_bath,E0_XBK,E0_exact\n")
        for U, L, e, ex_e in rows:
            fh.write(f"{U},{L},{e:.6f},{'' if ex_e is None else f'{ex_e:.6f}'}\n")
    print(f"\nSaved {args.csv}", flush=True)
