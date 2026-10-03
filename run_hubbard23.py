"""
run_hubbard23.py — XBK ground-state run for the 2x3 Hubbard cluster (n=10 frontier).

Uses the numpy-vectorised fast SA (sa_fast). Loops every (taper sector, p) subproblem,
Dinkelbach each, takes the min. Prints E0_XBK vs exact ED and the gap.

    python run_hubbard23.py [r] [U]        # defaults r=2 U=4
"""
import sys, time, math
import numpy as np
import xbk_pipeline as xp
import sa_fast

r = int(sys.argv[1]) if len(sys.argv) > 1 else 2
U = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
NR, NS = 10, max(2000, 50 * 10 * r)          # restarts, sweeps (per _budget_for)

H = xp.hubbard2d_rect(2, 3, t=1.0, U=U)
print(f"=== 2x3 Hubbard OBC  U/t={U}  r={r}  (fast SA {NR}x{NS}) ===", flush=True)

t0 = time.time()
Ex = xp.exact_ground_energy(H)
print(f"exact ED E0 = {Ex:.6f}   ({time.time()-t0:.0f}s)", flush=True)

sectors = xp.z2_taper(xp.bk_transform(H))
best = math.inf
for si, sec in enumerate(sectors):
    terms, n = xp._qubitop_to_terms(sec)
    if n == 0:
        best = min(best, float(np.real(sec.terms.get((), 0.0)))); continue
    for p in range(math.ceil(r / 2) + 1):
        H_is = xp.build_xbk_ising(terms, n, r, p)
        C_is = xp.build_counting(n, r, p)
        if not C_is:
            continue
        tt = time.time()
        e = sa_fast.dinkelbach_fast(H_is, C_is, n * r, seed=3 + si * 101 + p * 37,
                                    n_restarts=NR, n_sweeps=NS)
        best = min(best, e)
        print(f"  sector {si} p{p}: n={n} terms={len(H_is)}  E={e:.5f}  "
              f"(best {best:.5f}, {time.time()-tt:.0f}s)", flush=True)

gap = best - Ex
print(f"\nXBK^({r}) E0 = {best:.6f}   exact = {Ex:.6f}   gap = {gap:+.6f} "
      f"({100*gap/abs(Ex):.1f}% of |E0|)", flush=True)
