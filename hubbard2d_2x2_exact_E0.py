"""
hubbard2d_2x2_exact_E0.py
=========================
EXACT ground-state energy of the 2D half-filled Fermi-Hubbard model on a 2x2
lattice via sparse exact diagonalisation, swept over U/t. PH-symmetric form
(consistent with the 1D Hubbard / SIAM references), t = 1.

Two bond conventions are shown, because they genuinely differ on a 2x2:

  * "periodic 2x2"  (matches model_hubbard2d.py):
        full PBC in both directions. On a 2-wide lattice each neighbour pair is
        joined by TWO wrap-around bonds -> effective 2t hopping per pair.
  * "2x2 plaquette" (single bonds):
        the 4 sites joined in a single ring 0-1-3-2-0 (one bond per pair).
        Topologically identical to the 1D 4-site PBC ring.

Convention (PH-symmetric, half-filled; t=1; mode = 2*site + spin,
site = iy*L + ix):
    H = -t sum_{<i,j>,s} (c†_{i,s} c_{j,s} + h.c.)
        + U sum_i (n_{i,up}-1/2)(n_{i,down}-1/2)

Both lattices have a non-interacting zero mode at half filling, so 2nd-order
non-degenerate PT in U is singular -> only exact ED is reported.
"""

import os
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openfermion import FermionOperator, get_sparse_operator

L = 2
N_SITES = L * L                      # 4 sites


def _site(ix, iy):
    return iy * L + ix


def _mode(site, spin):
    return 2 * site + spin


def bonds_periodic():
    """Bonds exactly as model_hubbard2d.py generates them (PBC, double bonds on 2x2)."""
    blist = []
    for iy in range(L):
        for ix in range(L):
            i = _site(ix, iy)
            for jx, jy in [((ix + 1) % L, iy), (ix, (iy + 1) % L)]:
                blist.append((i, _site(jx, jy)))
    return blist                     # 8 directed bonds -> 4 pairs counted twice


def bonds_plaquette():
    """Single-bond 4-site ring 0-1-3-2-0 (one bond per neighbour pair)."""
    return [(0, 1), (1, 3), (3, 2), (2, 0)]


def hubbard2d_ph(U, bondlist):
    H = FermionOperator()
    for (i, j) in bondlist:
        for s in (0, 1):
            H += FermionOperator(((_mode(i, s), 1), (_mode(j, s), 0)), -1.0)
            H += FermionOperator(((_mode(j, s), 1), (_mode(i, s), 0)), -1.0)
    for i in range(N_SITES):
        up, dn = _mode(i, 0), _mode(i, 1)
        H += FermionOperator(((up, 1), (up, 0), (dn, 1), (dn, 0)), U)
        H += FermionOperator(((up, 1), (up, 0)), -U / 2)
        H += FermionOperator(((dn, 1), (dn, 0)), -U / 2)
        H += FermionOperator((), U / 4)
    return H


def E0_exact(U, bondlist):
    Hs = get_sparse_operator(hubbard2d_ph(U, bondlist), n_qubits=2 * N_SITES)
    vals = spla.eigsh(Hs.real, k=1, which="SA", return_eigenvectors=False, maxiter=5000)
    return float(vals[0])


UT = np.round(np.arange(0.0, 8.0 + 1e-9, 0.4), 6)

curves = {
    "periodic 2x2  (model_hubbard2d.py, double bonds)": bonds_periodic(),
    "2x2 plaquette  (single bonds = 4-site ring)":       bonds_plaquette(),
}

data = {}
for label, bl in curves.items():
    data[label] = np.array([E0_exact(ut, bl) for ut in UT])
    print(f"[done] {label}", flush=True)
    print(f"        U=0: E0 = {data[label][0]:+.6f}   U=8: E0 = {data[label][-1]:+.6f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
colors = ["#c1121f", "#003049"]
fig, ax = plt.subplots(figsize=(7.8, 5.6))
for color, (label, arr) in zip(colors, data.items()):
    ax.plot(UT, arr, "-o", color=color, ms=4, lw=1.8, label=label)
ax.set_xlabel(r"$U/t$", fontsize=13)
ax.set_ylabel(r"Ground-state energy  $E_0$  (units of $t$)", fontsize=13)
ax.set_title("2D Hubbard model, 2x2 lattice — exact ED\n"
             "half-filled, PH-symmetric,  t = 1", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubbard2d_2x2_exact_E0_vs_Ut.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {out}")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubbard2d_2x2_exact_E0_vs_Ut.csv")
with open(csv_path, "w") as fh:
    fh.write("convention,U_over_t,E0_exact\n")
    tags = {"periodic 2x2  (model_hubbard2d.py, double bonds)": "periodic2x2",
            "2x2 plaquette  (single bonds = 4-site ring)": "plaquette"}
    for label, arr in data.items():
        for ut, e in zip(UT, arr):
            fh.write(f"{tags[label]},{ut:.4f},{e:.10f}\n")
print(f"Saved table : {csv_path}")
