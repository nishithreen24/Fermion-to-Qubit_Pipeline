"""Validate every model in the xbk_pipeline registry against exact ED.

XBK energy is a Rayleigh quotient of the XBK ansatz, so it is always a
variational UPPER bound: the check is E0_XBK >= E0_exact (within tol) and
finite, for both fermionic (BK) and spin (no-BK) models.

A light fixed SA budget is used on purpose -- it still gives a valid (if looser)
upper bound and keeps the 8-qubit cases fast.
"""
import time
import xbk_pipeline as xp

# (label, builder) at small sizes that diagonalise quickly
CASES = [
    ("tv1d    L3 V2",    lambda: xp.tv1d(3, 1, 2.0)),
    ("tv2d    L2 V1",    lambda: xp.tv2d(2, 1, 1.0)),
    ("hubbard1d L2 U4",  lambda: xp.hubbard1d(2, 1, 4.0)),
    ("hubbard2d L2 U4",  lambda: xp.hubbard2d(2, 1, 4.0)),
    ("siam    L2 U4",    lambda: xp.siam(2, U=4.0)),
    ("pam     L2 U4",    lambda: xp.pam(2, 1, 1.0, 4.0)),
    ("kitaev  L4 D1",    lambda: xp.kitaev(4, 1, 1.0)),
    ("j1j2_1d L4 J2=.5", lambda: xp.j1j2_1d(4, 1, 0.5)),
    ("j1j2_2d L2 J2=.5", lambda: xp.j1j2_2d(2, 1, 0.5)),
]

BUDGET = dict(n_restarts=3, n_sweeps=400)   # light: validity, not tightness

print(f"{'model':18s} {'nq':>3} {'exact':>9} {'XBK r=2':>9} {'gap':>8} "
      f"{'valid?':>7} {'t':>6}")
for label, build in CASES:
    H = build()
    nq = xp.count_qubits(H)
    Ex = xp.exact_ground_energy(H, nq)
    t0 = time.time()
    E = xp.ground_state_energy(H, r=2, seed=3, **BUDGET)
    dt = time.time() - t0
    valid = "OK" if E >= Ex - 1e-6 else "BELOW!"
    print(f"{label:18s} {nq:>3} {Ex:9.4f} {E:9.4f} {E-Ex:>+8.4f} "
          f"{valid:>7} {dt:5.0f}s", flush=True)
