"""
hubbard1d_exact_E0.py
=====================
EXACT ground-state energy of the 1D half-filled Fermi-Hubbard model via sparse
exact diagonalisation, swept over U/t, one curve per L = 2,4,6,8, OBC and PBC.
Dashed lines overlay the 2nd-order perturbation-theory estimate so the small-U
agreement and large-U breakdown are visible.

Convention (particle-hole symmetric, half-filled; t=1; mode = 2j+sigma):
    H = -t sum_{<i,j>,s} (c†_{i,s} c_{j,s} + h.c.)
        + U sum_i (n_{i,up}-1/2)(n_{i,down}-1/2)
"""

import os
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openfermion import FermionOperator, get_sparse_operator


def mode(j, s):                      # s: 0=up, 1=down
    return 2 * j + s


def hubbard_ph(L, t, U, periodic):
    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic and L >= 3:
        bonds.append((L - 1, 0))
    H = FermionOperator()
    for (i, j) in bonds:
        for s in (0, 1):
            H += FermionOperator(((mode(i, s), 1), (mode(j, s), 0)), -t)
            H += FermionOperator(((mode(j, s), 1), (mode(i, s), 0)), -t)
    for j in range(L):
        up, dn = mode(j, 0), mode(j, 1)
        # (n_up - 1/2)(n_dn - 1/2) = n_up n_dn - n_up/2 - n_dn/2 + 1/4
        H += FermionOperator(((up, 1), (up, 0), (dn, 1), (dn, 0)), U)
        H += FermionOperator(((up, 1), (up, 0)), -U / 2)
        H += FermionOperator(((dn, 1), (dn, 0)), -U / 2)
        H += FermionOperator((), U / 4)
    return H


def E0_exact(L, t, U, periodic):
    Hs = get_sparse_operator(hubbard_ph(L, t, U, periodic), n_qubits=2 * L)
    if Hs.shape[0] <= 4:
        w = np.linalg.eigvalsh(Hs.toarray().real)
        return float(w[0])
    vals = spla.eigsh(Hs.real, k=1, which="SA", return_eigenvectors=False,
                      maxiter=5000)
    return float(vals[0])


# ── 2nd-order PT coefficient (for dashed overlay) ────────────────────────────
def pt_params(L, t, periodic, tol=1e-9):
    h = np.zeros((L, L))
    for j in range(L - 1):
        h[j, j + 1] = h[j + 1, j] = -t
    if periodic and L >= 3:
        h[L - 1, 0] = h[0, L - 1] = -t
    eps, psi = np.linalg.eigh(h)
    if np.any(np.abs(eps) < tol):
        return float(2 * eps[eps < -tol].sum()), None
    occ, emp = np.where(eps < 0)[0], np.where(eps > 0)[0]
    E0_free = float(2 * eps[occ].sum())
    A, D = [], []
    for hh in occ:
        for pp in emp:
            A.append(psi[:, pp] * psi[:, hh]); D.append(eps[pp] - eps[hh])
    A, D = np.array(A), np.array(D)
    c2 = -float(np.sum((A @ A.T) ** 2 / (D[:, None] + D[None, :])))
    return E0_free, c2


T = 1.0
L_VALUES = [2, 4, 6, 8]
UT = np.round(np.arange(0.0, 8.0 + 1e-9, 0.4), 6)        # exact sweep
UT_PT = np.linspace(0.0, 8.0, 81)                         # smooth PT overlay

exact = {}
for periodic in (False, True):
    bc = "PBC" if periodic else "OBC"
    for L in L_VALUES:
        exact[(bc, L)] = np.array([E0_exact(L, T, ut * T, periodic) for ut in UT])
        print(f"[{bc}] L={L} done", flush=True)

# ── Plot ─────────────────────────────────────────────────────────────────────
cmap = plt.cm.plasma(np.linspace(0, 0.85, len(L_VALUES)))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

for ax, (bc, periodic) in zip(axes, (("OBC", False), ("PBC", True))):
    for color, L in zip(cmap, L_VALUES):
        ax.plot(UT, exact[(bc, L)], "-o", color=color, ms=3.5, lw=1.7,
                label=f"L = {L}")
        e0, c2 = pt_params(L, T, periodic)
        if c2 is not None:
            ax.plot(UT_PT, e0 + c2 * UT_PT ** 2, "--", color=color, lw=1.1,
                    alpha=0.7)
    ax.set_xlabel(r"$U/t$", fontsize=13)
    ax.set_ylabel(r"Ground-state energy  $E_0$  (units of $t$)", fontsize=13)
    ax.set_title(f"1D half-filled Hubbard — exact ED  ({bc})", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(title="solid: exact   dashed: 2nd-order PT", fontsize=10, ncol=2)

fig.suptitle("1D half-filled Hubbard chain: exact ground-state energy vs U/t "
             "(t=1, PH-symmetric form)", fontsize=13)
fig.subplots_adjust(top=0.9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "hubbard1d_exact_E0_vs_Ut.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {out}")

# CSV dump
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hubbard1d_exact_E0_vs_Ut.csv")
with open(csv_path, "w") as fh:
    fh.write("BC,L,U_over_t,E0_exact\n")
    for (bc, L), arr in exact.items():
        for ut, e in zip(UT, arr):
            fh.write(f"{bc},{L},{ut:.4f},{e:.10f}\n")
print(f"Saved table : {csv_path}")
