"""
tune_sa.py — measure SA budget/schedule needed to hit the TRUE r=4 XBK optimum.

Targets (brute-forced, no SA):
  tv1d L3 V2          -2.0000
  tv1d L3 V1.5 PBC    -2.0417   <- was stalling at -1.50
  tv1d L4 V0.5        -2.2500
  hubbard L2 U4       -4.8000
"""
import math, time
from collections import defaultdict
import numpy as np
import xbk_pipeline as xp

CASES = [
    ('tv1d L3 V2',      xp.tv1d(3, 1, 2.0),                 -2.0000),
    ('tv1d L3 V1.5 PBC', xp.tv1d(3, 1, 1.5, periodic=True), -2.0417),
    ('tv1d L4 V0.5',    xp.tv1d(4, 1, 0.5),                 -2.2500),
    ('hubbard L2 U4',   xp.hubbard1d(2, 1, 4.0),            -4.8000),
]
R = 4


def dinkelbach_klocal(H_is, C_is, n_phys, seed, n_restarts, n_sweeps):
    """Dinkelbach where the inner SA anneals the *physical k-local* objective
    directly (no quadratization / no aux spins). Same algorithm, n_phys spins."""
    lam, best = 0.0, math.inf
    base = (max((abs(c) for c in H_is.values()), default=1.0)
            + max((abs(c) for c in C_is.values()), default=1.0))
    for it in range(30):
        comb = defaultdict(float, H_is)
        for m, c in C_is.items():
            comb[m] -= lam * c
        comb = {m: v for m, v in comb.items() if abs(v) > 1e-12}
        if not comb:
            break
        T0 = max(2.0 * base, 5.0)
        spins, _ = xp.sa_run(comb, n_phys, n_restarts, n_sweeps,
                             T0, 1e-3, seed + it * 1009)
        Hv, Cv = xp._eval(H_is, spins), xp._eval(C_is, spins)
        if abs(Cv) < 1e-8:
            continue
        new = Hv / Cv
        best = min(best, new)
        if abs(new - lam) < 1e-6:
            break
        lam = new
    return best


def run_klocal(H, r, n_restarts, n_sweeps, seed=3):
    H_bk = xp.bk_transform(H)
    best = math.inf
    for si, sec in enumerate(xp.z2_taper(H_bk)):
        terms, n = xp._qubitop_to_terms(sec)
        if n == 0:
            best = min(best, float(np.real(sec.terms.get((), 0.0))))
            continue
        for p in range(math.ceil(r / 2) + 1):
            H_is = xp.build_xbk_ising(terms, n, r, p)
            C_is = xp.build_counting(n, r, p)
            if not C_is:
                continue
            E0 = dinkelbach_klocal(H_is, C_is, n * r, seed + si * 101 + p * 37,
                                   n_restarts, n_sweeps)
            best = min(best, E0)
    return best


if __name__ == "__main__":
    print(f"{'case':18s} {'target':>8s} "
          f"{'quad r=4':>10s} {'klocal r=4':>11s} {'k-time':>7s}")
    for name, H, target in CASES:
        t0 = time.time()
        e_quad = xp.ground_state_energy(H, r=R, n_restarts=8, n_sweeps=2000, seed=3)
        t1 = time.time()
        e_kloc = run_klocal(H, R, n_restarts=8, n_sweeps=600)
        t2 = time.time()
        hit = '  HIT' if abs(e_kloc - target) < 1e-3 else f'  off {e_kloc-target:+.3f}'
        print(f"{name:18s} {target:8.4f} {e_quad:10.4f} {e_kloc:11.4f} "
              f"{t2-t1:6.1f}s{hit}")
