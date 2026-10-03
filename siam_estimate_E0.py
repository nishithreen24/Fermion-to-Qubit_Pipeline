"""
siam_estimate_E0.py
===================
THEORETICAL ESTIMATE (no many-body ED) of the ground-state energy of the
particle-hole-symmetric Single Impurity Anderson Model (SIAM), swept over U/V,
one curve per bath size N_bath = 1, 2, 3.

Convention (matches model_siam.py; t = V = 1):
    H = U (n_{d,up}-1/2)(n_{d,down}-1/2)               # impurity (PH symmetric)
      + sum_{b,s} eps_b n_{b,s}                         # bath
      + sum_{b,s} V_b (c†_{d,s} c_{b,s} + h.c.)         # hybridisation
    eps_b = -2t cos(b pi/(N+1)),  V_b = V sqrt(2/(N+1)) sqrt(1-(eps_b/2t)^2)

Method — free fermions + perturbation theory in U
-------------------------------------------------
At U=0 the model is a non-interacting (N+1)-site single-particle problem per
spin. Diagonalise the (N+1)x(N+1) matrix h (impurity site 0, bath sites 1..N):
    h[0,0]=0,  h[b,b]=eps_b,  h[0,b]=h[b,0]=V_b.
Eigenpairs (eps_a, psi_a); fill eps_a<0 in each spin:
    E0_free = 2 * sum_{eps_a<0} eps_a.

The interaction U(n_{d,up}-1/2)(n_{d,down}-1/2) acts ONLY on the impurity (site
0). At half filling <n_{d,s}> = 1/2, so the O(U) shift VANISHES; leading order
is O(U^2). With impurity-projected ph amplitude  a_{hp} = psi_p(0) psi_h(0)  and
gap D_{hp}=eps_p-eps_h>0,
    E0(U) ≈ E0_free  -  U^2 * sum_{alpha,beta} (a_alpha a_beta)^2 / (D_alpha+D_beta).
Exact at U=0, correct to O(U^2).

N_bath=2 has a single-particle zero mode (3 sites, PH-symmetric spectrum) ->
degenerate U=0 ground state -> PT singular; flagged and skipped.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def siam_single_particle(N, t=1.0, V=1.0):
    b = np.arange(1, N + 1)
    eps_b = -2.0 * t * np.cos(b * np.pi / (N + 1))
    V_b = V * np.sqrt(2.0 / (N + 1)) * np.sqrt(np.maximum(0.0, 1.0 - (eps_b / (2 * t)) ** 2))
    h = np.zeros((N + 1, N + 1))
    for bi, (e, v) in enumerate(zip(eps_b, V_b), start=1):
        h[bi, bi] = e
        h[0, bi] = h[bi, 0] = v
    eps, psi = np.linalg.eigh(h)
    return eps, psi


def estimate(N, tol=1e-9):
    """Return (E0_free, c2, degenerate).  E0(U) ≈ E0_free + c2 * U^2."""
    eps, psi = siam_single_particle(N)
    if np.any(np.abs(eps) < tol):
        return float(2 * eps[eps < -tol].sum()), None, True
    occ, emp = np.where(eps < 0)[0], np.where(eps > 0)[0]
    E0_free = float(2 * eps[occ].sum())
    a, D = [], []
    for h in occ:
        for p in emp:
            a.append(psi[0, p] * psi[0, h])     # impurity-site amplitude only
            D.append(eps[p] - eps[h])
    a, D = np.array(a), np.array(D)
    c2 = -float(np.sum(np.outer(a, a) ** 2 / (D[:, None] + D[None, :])))
    return E0_free, c2, False


N_VALUES = [1, 2, 3]
UV = np.linspace(0.0, 8.0, 81)

res = {N: estimate(N) for N in N_VALUES}
print("Second-order estimate  E0(U) ≈ E0_free + c2 * U^2   (c1 = 0 at half filling)")
print(f"{'N_bath':>6} {'E0_free':>10} {'c2':>10}   note")
for N, (e0, c2, deg) in res.items():
    note = "DEGENERATE (zero mode) - PT singular, skipped" if deg else ""
    c2s = "   ---   " if c2 is None else f"{c2:>10.5f}"
    print(f"{N:>6} {e0:>10.5f} {c2s}   {note}")

cmap = plt.cm.cividis(np.linspace(0.05, 0.85, len(N_VALUES)))
fig, ax = plt.subplots(figsize=(7.2, 5.4))
skipped = []
for color, N in zip(cmap, N_VALUES):
    e0, c2, deg = res[N]
    if deg:
        skipped.append(N)
        continue
    ax.plot(UV, e0 + c2 * UV ** 2, "-", color=color, lw=1.9, label=f"N_bath = {N}")
    ax.plot(0, e0, "o", color=color, ms=5)
ax.set_xlabel(r"$U/V$", fontsize=13)
ax.set_ylabel(r"Ground-state energy  $E_0$  (units of $t$)", fontsize=13)
ax.set_title("SIAM — estimate (free fermions + 2nd-order PT in U,  t=V=1)", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(title="bath size", fontsize=11)
if skipped:
    ax.text(0.03, 0.04, "N_bath = " + ", ".join(map(str, skipped)) +
            " omitted:\nzero-mode degeneracy (PT singular)",
            transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "siam_estimate_E0_vs_UV.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {out}")
