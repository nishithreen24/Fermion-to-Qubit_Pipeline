"""
siam_exact_E0.py
================
EXACT ground-state energy of the particle-hole-symmetric SIAM via sparse exact
diagonalisation, swept over U/V, one curve per bath size N_bath = 1, 2, 3.
Dashed lines overlay the 2nd-order PT estimate (where non-singular).

Convention (matches model_siam.py; t = V = 1; mode = 2*site + spin,
site 0 = impurity, sites 1..N = bath):
    H = U (n_{d,up}-1/2)(n_{d,down}-1/2)
      + sum_{b,s} eps_b n_{b,s}
      + sum_{b,s} V_b (c†_{d,s} c_{b,s} + h.c.)
"""

import os
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openfermion import FermionOperator, get_sparse_operator


def _mode(site, spin):
    return 2 * site + spin


def siam_hamiltonian(N, U, t=1.0, V=1.0):
    b = np.arange(1, N + 1)
    eps_b = -2.0 * t * np.cos(b * np.pi / (N + 1))
    V_b = V * np.sqrt(2.0 / (N + 1)) * np.sqrt(np.maximum(0.0, 1.0 - (eps_b / (2 * t)) ** 2))

    H = FermionOperator()
    up, dn = _mode(0, 0), _mode(0, 1)
    H += FermionOperator(((up, 1), (dn, 1), (dn, 0), (up, 0)), U)
    for spin in (0, 1):
        H += FermionOperator(((_mode(0, spin), 1), (_mode(0, spin), 0)), -U / 2)
    H += FermionOperator((), U / 4)

    for bs, e in zip(range(1, N + 1), eps_b):
        for spin in (0, 1):
            p = _mode(bs, spin)
            H += FermionOperator(((p, 1), (p, 0)), float(e))
    for bs, v in zip(range(1, N + 1), V_b):
        for spin in (0, 1):
            im, bm = _mode(0, spin), _mode(bs, spin)
            H += FermionOperator(((im, 1), (bm, 0)), float(v))
            H += FermionOperator(((bm, 1), (im, 0)), float(v))
    return H


def E0_exact(N, U):
    Hs = get_sparse_operator(siam_hamiltonian(N, U), n_qubits=2 * (N + 1))
    if Hs.shape[0] <= 4:
        return float(np.linalg.eigvalsh(Hs.toarray().real)[0])
    vals = spla.eigsh(Hs.real, k=1, which="SA", return_eigenvectors=False, maxiter=5000)
    return float(vals[0])


def pt_params(N, t=1.0, V=1.0, tol=1e-9):
    b = np.arange(1, N + 1)
    eps_b = -2.0 * t * np.cos(b * np.pi / (N + 1))
    V_b = V * np.sqrt(2.0 / (N + 1)) * np.sqrt(np.maximum(0.0, 1.0 - (eps_b / (2 * t)) ** 2))
    h = np.zeros((N + 1, N + 1))
    for bi, (e, v) in enumerate(zip(eps_b, V_b), start=1):
        h[bi, bi] = e; h[0, bi] = h[bi, 0] = v
    eps, psi = np.linalg.eigh(h)
    if np.any(np.abs(eps) < tol):
        return float(2 * eps[eps < -tol].sum()), None
    occ, emp = np.where(eps < 0)[0], np.where(eps > 0)[0]
    E0_free = float(2 * eps[occ].sum())
    a, D = [], []
    for hh in occ:
        for pp in emp:
            a.append(psi[0, pp] * psi[0, hh]); D.append(eps[pp] - eps[hh])
    a, D = np.array(a), np.array(D)
    c2 = -float(np.sum(np.outer(a, a) ** 2 / (D[:, None] + D[None, :])))
    return E0_free, c2


N_VALUES = [1, 2, 3]
UV = np.round(np.arange(0.0, 8.0 + 1e-9, 0.4), 6)
UV_PT = np.linspace(0.0, 8.0, 81)

exact = {}
for N in N_VALUES:
    exact[N] = np.array([E0_exact(N, uv) for uv in UV])
    print(f"N_bath={N} done", flush=True)

cmap = plt.cm.cividis(np.linspace(0.05, 0.85, len(N_VALUES)))
fig, ax = plt.subplots(figsize=(7.6, 5.6))
for color, N in zip(cmap, N_VALUES):
    ax.plot(UV, exact[N], "-o", color=color, ms=4, lw=1.8, label=f"N_bath = {N}")
    e0, c2 = pt_params(N)
    if c2 is not None:
        ax.plot(UV_PT, e0 + c2 * UV_PT ** 2, "--", color=color, lw=1.1, alpha=0.7)
ax.set_xlabel(r"$U/V$", fontsize=13)
ax.set_ylabel(r"Ground-state energy  $E_0$  (units of $t$)", fontsize=13)
ax.set_title("SIAM — exact ED (t = V = 1, PH-symmetric)\n"
             "solid: exact    dashed: 2nd-order PT", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(title="bath size", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "siam_exact_E0_vs_UV.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {out}")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "siam_exact_E0_vs_UV.csv")
with open(csv_path, "w") as fh:
    fh.write("N_bath,U_over_V,E0_exact\n")
    for N, arr in exact.items():
        for uv, e in zip(UV, arr):
            fh.write(f"{N},{uv:.4f},{e:.10f}\n")
print(f"Saved table : {csv_path}")
