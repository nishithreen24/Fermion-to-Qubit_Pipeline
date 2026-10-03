"""
tv1d_exact_E0.py
================
Exact (theoretical) ground-state energy of the 1D spinless t-V model, swept
over V/t, with one curve per system size L = 2..8, for BOTH open (OBC) and
periodic (PBC) boundary conditions.

Convention (matches tvtest.py / tv1d_L2_pipeline.py docstrings):

    H = -t  sum_<i,j> (c†_i c_j + c†_j c_i)
        + V  sum_<i,j> (n_i - 1/2)(n_j - 1/2)

with t = 1 fixed, <i,j> = unique nearest-neighbour bonds.

  * OBC : bonds (0,1),(1,2),...,(L-2,L-1)        -> L-1 bonds
  * PBC : OBC bonds plus the wrap bond (L-1,0)    -> L bonds   (for L>=3)

Note on L=2: under PBC the single pair (0,1) is the only nearest-neighbour
pair on the ring, so to avoid double-counting one physical bond we use the
UNIQUE-pair convention. This makes L=2 OBC == L=2 PBC by construction; the
two curves separate for L>=3 where the wrap bond is a genuinely new bond.

E0 is the lowest eigenvalue over the FULL 2^L Fock space (no fixed filling),
i.e. the true ground state of H. Computed by dense diagonalisation
(2^8 = 256 is tiny).

Sanity check (V=0, free fermions): E0 = sum over single-particle energies
eps_k < 0. OBC: eps_k = -2t cos(pi k/(L+1)), k=1..L. PBC: eps_k = -2t cos(2 pi k/L).
The script prints these analytic values next to the ED result.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openfermion import FermionOperator, get_sparse_operator


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian (particle-hole symmetric t-V), unique-bond convention
# ─────────────────────────────────────────────────────────────────────────────
def tv1d_ph(L, t, V, periodic):
    """1D t-V Hamiltonian, H = -t hop + V sum (n_i-1/2)(n_j-1/2)."""
    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic and L >= 3:
        bonds.append((L - 1, 0))          # wrap bond; unique for L>=3

    H = FermionOperator()
    for (i, j) in bonds:
        # hopping
        H += FermionOperator(((i, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (i, 0)), -t)
        # (n_i - 1/2)(n_j - 1/2) = n_i n_j - n_i/2 - n_j/2 + 1/4
        H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)
        H += FermionOperator(((i, 1), (i, 0)), -V / 2)
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator((), V / 4)
    return H


def E0_exact(L, t, V, periodic):
    H = get_sparse_operator(tv1d_ph(L, t, V, periodic), n_qubits=L)
    w = np.linalg.eigvalsh(H.toarray().real)
    return float(w[0])


def E0_free(L, t, periodic):
    """Analytic V=0 ground-state energy: sum of negative single-particle levels."""
    if periodic and L >= 3:
        eps = [-2 * t * np.cos(2 * np.pi * k / L) for k in range(L)]
    else:
        eps = [-2 * t * np.cos(np.pi * k / (L + 1)) for k in range(1, L + 1)]
    return float(sum(e for e in eps if e < 0))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
T = 1.0
L_VALUES = [2, 3, 4, 5, 6, 7, 8]
VT = np.round(np.arange(0.0, 4.0 + 1e-9, 0.2), 6)

results = {}   # (bc, L) -> array of E0 over VT
for periodic in (False, True):
    bc = "PBC" if periodic else "OBC"
    for L in L_VALUES:
        results[(bc, L)] = np.array([E0_exact(L, T, vt * T, periodic) for vt in VT])

# ── V=0 cross-check against the free-fermion closed form ─────────────────────
print("V=0 cross-check (ED vs analytic free-fermion):")
print(f"{'BC':>4} {'L':>3} {'E0_ED':>12} {'E0_analytic':>14} {'diff':>10}")
for periodic in (False, True):
    bc = "PBC" if periodic else "OBC"
    for L in L_VALUES:
        ed = results[(bc, L)][0]
        an = E0_free(L, T, periodic)
        print(f"{bc:>4} {L:>3} {ed:>12.6f} {an:>14.6f} {abs(ed-an):>10.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: E0 vs V/t, one curve per L, two panels (OBC, PBC)
# ─────────────────────────────────────────────────────────────────────────────
def pt_slope(L, t, periodic):
    """First-order PT slope: E0(V) ≈ E0_free + slope * V (free-fermion correlator)."""
    h = np.zeros((L, L))
    for j in range(L - 1):
        h[j, j + 1] = h[j + 1, j] = -t
    if periodic and L >= 3:
        h[L - 1, 0] = h[0, L - 1] = -t
    eps, psi = np.linalg.eigh(h)
    occ = eps < -1e-9
    E0_free = float(eps[occ].sum())
    G = psi[:, occ] @ psi[:, occ].T
    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic and L >= 3:
        bonds.append((L - 1, 0))
    slope = sum((G[i, i] - 0.5) * (G[j, j] - 0.5) - G[i, j] ** 2 for (i, j) in bonds)
    return E0_free, float(slope)


cmap = plt.cm.viridis(np.linspace(0, 0.92, len(L_VALUES)))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False)

for ax, (bc, periodic) in zip(axes, (("OBC", False), ("PBC", True))):
    for color, L in zip(cmap, L_VALUES):
        ax.plot(VT, results[(bc, L)], "-o", color=color, ms=3.5,
                lw=1.6, label=f"L = {L}")
        e0, slope = pt_slope(L, T, periodic)
        ax.plot(VT, e0 + slope * (VT * T), "--", color=color, lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$V/t$", fontsize=13)
    ax.set_ylabel(r"Ground-state energy  $E_0$  (units of $t$)", fontsize=13)
    ax.set_title(f"1D spinless t-V model — exact $E_0$  ({bc})", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(title="solid: exact   dashed: 1st-order PT", fontsize=10, ncol=2)

fig.suptitle(
    "Exact ground-state energy of the 1D t-V chain   "
    r"$H=-t\sum(c^\dagger_i c_j+\mathrm{h.c.})+V\sum(n_i-1/2)(n_j-1/2)$,  $t=1$",
    fontsize=13)
fig.subplots_adjust(top=0.9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tv1d_exact_E0_vs_Vt.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {out}")

# Also dump the raw numbers as CSV for use as a benchmark table
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tv1d_exact_E0_vs_Vt.csv")
with open(csv_path, "w") as fh:
    fh.write("BC,L,V_over_t,E0\n")
    for (bc, L), arr in results.items():
        for vt, e in zip(VT, arr):
            fh.write(f"{bc},{L},{vt:.4f},{e:.10f}\n")
print(f"Saved table : {csv_path}")
