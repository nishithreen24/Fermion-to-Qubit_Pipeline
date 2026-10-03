"""
hubbard1d_estimate_E0.py
========================
THEORETICAL ESTIMATE (no many-body exact diagonalisation) of the ground-state
energy of the 1D Fermi-Hubbard model at half filling, swept over U/t, with one
curve per system size L = 2,4,6,8, for OBC and PBC.

Convention (particle-hole symmetric, half-filled form; t=1):
    H = -t sum_{<i,j>,s} (c†_{i,s} c_{j,s} + h.c.)
        + U sum_i (n_{i,up}-1/2)(n_{i,down}-1/2)

(This differs from the project's raw  U sum_i n_up n_down  only by the constant
 -U*L/4 and a chemical potential that pins half filling; the PH form is the
 standard half-filled Hubbard model and is the one with a clean expansion.)

Method — free fermions + perturbation theory in U
-------------------------------------------------
At U=0 the two spin species are independent free hopping chains. Diagonalise
only the L x L single-particle hopping matrix (NOT the 4^L many-body space):
eigenpairs (eps_a, psi_a). Half-filled free ground state fills eps_a < 0 in each
spin, with exact energy
    E0_free = 2 * sum_{eps_a < 0} eps_a.

FIRST order: <(n_up-1/2)(n_down-1/2)>_0 = (G_ii-1/2)^2 with G_ii = 1/2 at
half filling (even L) -> the O(U) shift VANISHES. The leading effect is O(U^2).

SECOND order (Rayleigh-Schroedinger): the perturbation
W = U sum_i dn_{i,up} dn_{i,down}  (dn = n - 1/2) connects |0> to states with one
particle-hole pair in EACH spin channel. With the ph amplitude
    A_{hp}(i) = psi_p(i) psi_h(i)        (h occupied, p empty)
and excitation energy  D_{hp} = eps_p - eps_h > 0,
    E0(U) ≈ E0_free  -  U^2 * sum_{alpha,beta} (A_alpha . A_beta)^2 / (D_alpha + D_beta)
where alpha runs over up-channel ph pairs and beta over down-channel ph pairs.
This is EXACT at U=0 and correct to O(U^2) (the leading curvature of E0 vs U).

Validation: for L=2 OBC (the Hubbard dimer) the closed form is
    E0_exact(PH) = -(1/2) sqrt(U^2 + 16 t^2)  =  -2t - U^2/(16t) + O(U^4),
so the second-order coefficient must be -1/16 = -0.0625.  (Checked below.)

Validity: small/moderate U. At large U the true half-filled chain crosses over
to the antiferromagnetic Heisenberg model (J = 4 t^2 / U), where E0 -> 0 from
below as ~ -L * 4 ln2 * t^2 / U; the O(U^2) parabola overshoots there.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def single_particle(L, t, periodic):
    h = np.zeros((L, L))
    for j in range(L - 1):
        h[j, j + 1] = h[j + 1, j] = -t
    if periodic and L >= 3:
        h[L - 1, 0] = h[0, L - 1] = -t
    eps, psi = np.linalg.eigh(h)
    return eps, psi


def estimate(L, t, periodic, tol=1e-9):
    """Return (E0_free, c2, degenerate_flag).  E0(U) ≈ E0_free + c2 * U^2."""
    eps, psi = single_particle(L, t, periodic)

    if np.any(np.abs(eps) < tol):
        # zero-energy single-particle mode -> non-interacting ground state is
        # degenerate -> non-degenerate PT in U is singular. Flag and bail.
        return float(2 * eps[eps < -tol].sum()), None, True

    occ = np.where(eps < 0)[0]
    emp = np.where(eps > 0)[0]
    E0_free = float(2 * eps[occ].sum())

    # all particle-hole pairs: amplitude vector A(i)=psi_p(i)psi_h(i), gap D
    A = []      # list of length-L vectors
    D = []      # excitation energies
    for h in occ:
        for p in emp:
            A.append(psi[:, p] * psi[:, h])
            D.append(eps[p] - eps[h])
    A = np.array(A)              # (P, L)
    D = np.array(D)              # (P,)

    overlap = A @ A.T           # (P,P):  A_alpha . A_beta
    denom = D[:, None] + D[None, :]
    c2 = -float(np.sum(overlap ** 2 / denom))
    return E0_free, c2, False


T = 1.0
L_VALUES = [2, 4, 6, 8]
UT = np.linspace(0.0, 4.0, 41)

results = {}    # (bc, L) -> (E0_free, c2, degenerate)
for periodic in (False, True):
    bc = "PBC" if periodic else "OBC"
    for L in L_VALUES:
        results[(bc, L)] = estimate(L, T, periodic)

print("Second-order estimate  E0(U) ≈ E0_free + c2 * U^2   (c1 = 0 at half filling)")
print(f"{'BC':>4} {'L':>3} {'E0_free':>10} {'c2':>10}   note")
for (bc, L), (e0, c2, deg) in results.items():
    note = "DEGENERATE (zero modes) - PT singular, skipped" if deg else ""
    c2s = "   ---   " if c2 is None else f"{c2:>10.5f}"
    print(f"{bc:>4} {L:>3} {e0:>10.5f} {c2s}   {note}")

# validation against the exact Hubbard dimer (L=2 OBC)
c2_dimer = results[("OBC", 2)][1]
print(f"\nValidation L=2 OBC: c2 = {c2_dimer:.6f}  (exact dimer = -1/16 = -0.0625)")

# ── Plot ─────────────────────────────────────────────────────────────────────
cmap = plt.cm.plasma(np.linspace(0, 0.85, len(L_VALUES)))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

for ax, bc in zip(axes, ("OBC", "PBC")):
    skipped = []
    for color, L in zip(cmap, L_VALUES):
        e0, c2, deg = results[(bc, L)]
        if deg:
            skipped.append(L)
            continue
        ax.plot(UT, e0 + c2 * (UT * T) ** 2, "-", color=color, lw=1.8, label=f"L = {L}")
        ax.plot(0, e0, "o", color=color, ms=4)
    ax.set_xlabel(r"$U/t$", fontsize=13)
    ax.set_ylabel(r"Ground-state energy  $E_0$  (units of $t$)", fontsize=13)
    ax.set_title(f"1D Hubbard model — estimate  ({bc})", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(title="system size", fontsize=10, ncol=2)
    if skipped:
        ax.text(0.03, 0.04,
                "L = " + ", ".join(map(str, skipped)) +
                " omitted:\nzero-mode degeneracy (PT singular)",
                transform=ax.transAxes, fontsize=9, color="0.35",
                va="bottom", ha="left")

fig.suptitle("1D half-filled Hubbard chain: estimated ground-state energy "
             "(free fermions + 2nd-order perturbation theory in U,  t=1)",
             fontsize=13)
fig.subplots_adjust(top=0.9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubbard1d_estimate_E0_vs_Ut.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {out}")
