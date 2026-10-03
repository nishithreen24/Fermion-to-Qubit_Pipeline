"""
tv1d_estimate_E0.py
===================
THEORETICAL ESTIMATE (no many-body exact diagonalisation) of the
ground-state energy of the 1D spinless t-V model, swept over V/t, with
one curve per system size L = 2..8, for both OBC and PBC.

Convention (matches the project's t-V code):
    H = -t sum_<i,j> (c†_i c_j + h.c.) + V sum_<i,j> (n_i-1/2)(n_j-1/2),  t=1.

Method — free fermions + first-order perturbation theory in V
-------------------------------------------------------------
At V=0 the model is non-interacting. Diagonalise only the L x L single-
particle hopping matrix h (NOT the 2^L many-body Hamiltonian):
    h_{ij} = -t for nearest neighbours, 0 otherwise.
Eigenvalues eps_a, eigenvectors psi_a. The V=0 ground state fills every
orbital with eps_a < 0, giving the exact free energy
    E0_free = sum_{eps_a < 0} eps_a.

First-order shift: treat V*sum(n_i-1/2)(n_j-1/2) as a perturbation. Its
expectation in the (Slater-determinant) free ground state follows from
Wick's theorem with the equal-time correlator
    G_{ij} = <c†_i c_j>_0 = sum_{occ a} psi_a(i) psi_a(j):
    <(n_i-1/2)(n_j-1/2)>_0 = (G_ii-1/2)(G_jj-1/2) - |G_ij|^2.
Hence
    E0(V) ≈ E0_free + V * sum_<i,j> [ (G_ii-1/2)(G_jj-1/2) - |G_ij|^2 ].

This is EXACT at V=0 and correct to first order in V (the leading slope of
E0 vs V/t). It is a genuine analytic estimate, accurate for small/moderate
V/t and giving the right qualitative trend with L.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def single_particle(L, t, periodic):
    """Return (eigvals, eigvecs) of the L x L hopping matrix h_ij = -t on bonds."""
    h = np.zeros((L, L))
    for j in range(L - 1):
        h[j, j + 1] = h[j + 1, j] = -t
    if periodic and L >= 3:
        h[L - 1, 0] = h[0, L - 1] = -t
    eps, psi = np.linalg.eigh(h)
    return eps, psi


def estimate(L, t, periodic):
    """Return (E0_free, slope) so that E0(V) ≈ E0_free + slope * V."""
    eps, psi = single_particle(L, t, periodic)
    occ = eps < -1e-9                      # fill strictly-negative orbitals
    E0_free = float(eps[occ].sum())

    # correlator G_ij = sum over occupied orbitals of psi_a(i) psi_a(j)
    P = psi[:, occ]
    G = P @ P.T                            # L x L

    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic and L >= 3:
        bonds.append((L - 1, 0))

    slope = 0.0
    for (i, j) in bonds:
        slope += (G[i, i] - 0.5) * (G[j, j] - 0.5) - G[i, j] ** 2
    return E0_free, float(slope)


T = 1.0
L_VALUES = [2, 3, 4, 5, 6, 7, 8]
VT = np.linspace(0.0, 4.0, 41)

params = {}   # (bc, L) -> (E0_free, slope)
for periodic in (False, True):
    bc = "PBC" if periodic else "OBC"
    for L in L_VALUES:
        params[(bc, L)] = estimate(L, T, periodic)

print("First-order estimate  E0(V) ≈ E0_free + slope * V")
print(f"{'BC':>4} {'L':>3} {'E0_free':>10} {'slope':>9}")
for (bc, L), (e0, s) in params.items():
    print(f"{bc:>4} {L:>3} {e0:>10.5f} {s:>9.5f}")

# ── Plot: E0 vs V/t, one curve per L, two panels (OBC, PBC) ──────────────────
cmap = plt.cm.viridis(np.linspace(0, 0.92, len(L_VALUES)))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

for ax, bc in zip(axes, ("OBC", "PBC")):
    for color, L in zip(cmap, L_VALUES):
        e0, s = params[(bc, L)]
        ax.plot(VT, e0 + s * (VT * T), "-", color=color, lw=1.8, label=f"L = {L}")
        ax.plot(0, e0, "o", color=color, ms=4)
    ax.set_xlabel(r"$V/t$", fontsize=13)
    ax.set_ylabel(r"Ground-state energy  $E_0$  (units of $t$)", fontsize=13)
    ax.set_title(f"1D t-V model — estimate  ({bc})", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(title="system size", fontsize=10, ncol=2)

fig.suptitle("1D spinless t-V chain: estimated ground-state energy "
             "(free fermions + first-order perturbation theory in V,  t=1)",
             fontsize=13)
fig.subplots_adjust(top=0.9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tv1d_estimate_E0_vs_Vt.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {out}")
