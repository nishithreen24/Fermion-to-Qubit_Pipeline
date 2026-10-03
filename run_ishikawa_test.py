"""
run_ishikawa_test.py — end-to-end check of the XBK pipeline using ISHIKAWA
quadratization instead of Rosenberg.  Does not modify xbk_pipeline.py.

For each small instance, compares ground-state energy from:
  exact ED  |  k-local SA (no quad)  |  Rosenberg-quad SA  |  Ishikawa-quad SA
All four should agree (XBK is a valid bound; the three SA paths solve the SAME
objective and should land on the same E0).
"""
import math
import numpy as np
import xbk_pipeline as xp
from ishikawa import quadratize_ishikawa


def dinkelbach_ishikawa(H_is, C_is, n_phys, seed, n_restarts=8, n_sweeps=1500,
                        max_iter=30, tol=1e-6):
    """Dinkelbach with Ishikawa quadratization (no penalty M)."""
    lam, best = 0.0, math.inf
    base = max((abs(c) for c in H_is.values()), default=1.0) \
        + max((abs(c) for c in C_is.values()), default=1.0)
    for it in range(max_iter):
        from collections import defaultdict
        comb = defaultdict(float, H_is)
        for m, c in C_is.items():
            comb[m] -= lam * c
        comb = {m: v for m, v in comb.items() if abs(v) > 1e-12}
        if not comb:
            break
        binary = xp._ising_to_binary(comb)
        q2, nid = quadratize_ishikawa(binary, n_phys)     # <-- Ishikawa, no M
        q2_is = xp._binary_to_ising(q2)
        T0 = max(2.0 * base, 5.0)
        spins, _ = xp.sa_run(q2_is, nid, n_restarts, n_sweeps,
                             T0, 1e-3, seed + it * 1009)
        Hv, Cv = xp._eval(H_is, spins), xp._eval(C_is, spins)  # physical spins
        if abs(Cv) < 1e-8:
            continue
        new = Hv / Cv
        best = min(best, new)
        if abs(new - lam) < tol:
            break
        lam = new
    return best


def gse_ishikawa(H, r=2, taper=True, seed=3):
    H_q = xp.bk_transform(H)
    sectors = xp.z2_taper(H_q) if taper else [H_q]
    best = math.inf
    for si, sec in enumerate(sectors):
        terms, n = xp._qubitop_to_terms(sec)
        if n == 0:
            best = min(best, float(np.real(sec.terms.get((), 0.0))))
            continue
        for p in range(math.ceil(r / 2) + 1):
            H_is = xp.build_xbk_ising(terms, n, r, p)
            C_is = xp.build_counting(n, r, p)
            if not C_is:
                continue
            e = dinkelbach_ishikawa(H_is, C_is, n * r, seed + si * 101 + p * 37)
            best = min(best, e)
    return best


if __name__ == "__main__":
    # tv1d L=2 (tapers fully -> trivial), and cases that DO exercise quadratization
    cases = [
        ("tv1d L2 V1 (taper)",   xp.tv1d(2, 1, 1.0), True),
        ("tv1d L2 V1 (no-taper)", xp.tv1d(2, 1, 1.0), False),  # n=2: exercises quad
        ("tv1d L4 V2 (taper)",   xp.tv1d(4, 1, 2.0), True),    # n=2 sectors
        ("tv1d L3 V2 (taper)",   xp.tv1d(3, 1, 2.0), True),
    ]
    print(f"{'case':22s} {'exact':>9} {'k-local':>9} {'Rosenberg':>9} "
          f"{'Ishikawa':>9}  {'agree?':>6}")
    for name, H, tap in cases:
        Ex = xp.exact_ground_energy(H)
        e_k = xp.ground_state_energy(H, r=2, taper=tap, n_restarts=8,
                                     n_sweeps=1500, seed=3)
        e_r = xp.ground_state_energy(H, r=2, taper=tap, n_restarts=8,
                                     n_sweeps=1500, seed=3, quadratic=True)
        e_i = gse_ishikawa(H, r=2, taper=tap, seed=3)
        ok = "OK" if (abs(e_k - e_i) < 1e-6 and abs(e_r - e_i) < 1e-6
                      and e_i >= Ex - 1e-6) else "DIFF"
        print(f"{name:22s} {Ex:9.4f} {e_k:9.4f} {e_r:9.4f} {e_i:9.4f}  {ok:>6}",
              flush=True)
