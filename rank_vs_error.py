"""
rank_vs_error.py — the paper's central thesis as a measured law.

Claim: XBK r=2 accuracy is governed by REPRESENTABILITY of the ground state
(how few computational-basis configs carry its weight), NOT by entanglement
entropy S and NOT by the presence of spin.

For each Table-II instance we compute, from the EXACT ground state:
  S   : half-cut von Neumann entanglement entropy (nats)
  PR  : participation ratio 1/sum|c|^4 (effective # of basis configs)
and pair it with the paper's authoritative r=2 gap |dE| (hard-coded from
Table II so the x/y axes are a single consistent replication factor).

Produces a two-panel figure:
  (a) |dE| vs S   -> NO correlation (Majumdar-Ghosh point: S large, gap 0)
  (b) |dE| vs PR  -> clear monotone trend (the controlling variable)
The contrast between the panels IS the result.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openfermion import get_sparse_operator
from openfermion.utils import count_qubits
import models_mpi as mm

# (model, L, ratio, r=2 gap from Table II, label)
INSTANCES = [
    ('hubbard1d', 2, 4.0, 0.828, 'Hub1D L2'),
    ('hubbard1d', 3, 4.0, 1.236, 'Hub1D L3'),
    ('hubbard2d', 2, 4.0, 5.254, 'Hub2D 2x2'),
    ('siam',      3, 4.0, 0.919, 'SIAM Nb3'),
    ('pam',       2, 4.0, 2.647, 'PAM Nb2'),
    ('j1j2_1d',   4, 0.0, 0.500, 'J1J2 L4 J2=0'),
    ('j1j2_1d',   4, 0.5, 0.000, 'J1J2 L4 MG'),
    ('j1j2_1d',   8, 0.5, 1.500, 'J1J2 L8 MG'),
    ('kitaev',    4, 0.5, 0.422, 'Kitaev L4'),
]

def gs_metrics(H, nq):
    M = get_sparse_operator(H, n_qubits=nq).toarray()
    w, v = np.linalg.eigh(M)
    psi = v[:, 0].astype(complex); psi /= np.linalg.norm(psi)
    p = np.abs(psi) ** 2
    PR = 1.0 / np.sum(p ** 2)
    a = nq // 2
    lam = np.linalg.svd(psi.reshape(2**a, 2**(nq-a)), compute_uv=False) ** 2
    lam = lam[lam > 1e-14]
    S = float(-np.sum(lam * np.log(lam)))
    return S, PR

rows = []
print(f"{'label':14s} {'nq':>3} {'S':>6} {'PR':>7} {'gap_r2':>7}")
for name, L, ratio, gap, lab in INSTANCES:
    H = mm.model_hamiltonian(name, ratio, L)
    nq = count_qubits(H)
    S, PR = gs_metrics(H, nq)
    rows.append((lab, S, PR, gap))
    print(f"{lab:14s} {nq:>3} {S:>6.3f} {PR:>7.2f} {gap:>7.3f}", flush=True)

S  = np.array([r[1] for r in rows])
PR = np.array([r[2] for r in rows])
gap= np.array([r[3] for r in rows])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, x, xl in [(ax[0], S, r"half-cut entanglement entropy $S$ (nats)"),
                 (ax[1], PR, r"participation ratio  $\mathrm{PR}=1/\sum_b|\psi_b|^4$")]:
    a.scatter(x, gap, s=60, c='#3b1f6b', zorder=3)
    for (lab, _, _, _), xi, yi in zip(rows, x, gap):
        a.annotate(lab, (xi, yi), fontsize=7, xytext=(4,3),
                   textcoords='offset points')
    a.set_xlabel(xl); a.set_ylabel(r"XBK $r{=}2$ error  $|\Delta E|$")
    a.grid(alpha=0.3)
# Spearman rank correlations (monotonicity, robust to nonlinearity)
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]
rS, rPR = spearman(S, gap), spearman(PR, gap)
ax[0].set_title(f"(a)  vs entropy   (Spearman $\\rho={rS:.2f}$)")
ax[1].set_title(f"(b)  vs representability   (Spearman $\\rho={rPR:.2f}$)")
fig.tight_layout()
fig.savefig("rank_vs_error.png", dpi=170, bbox_inches='tight', facecolor='white')
print(f"\nSpearman rho:  gap vs S = {rS:.3f}   gap vs PR = {rPR:.3f}")
print("wrote rank_vs_error.png")
