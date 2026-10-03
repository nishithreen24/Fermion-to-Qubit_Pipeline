"""
Central new result: does the XBK r=2 gap correlate with ground-state
entanglement?  For each instance compute, from the EXACT ground state:
  - S    : bipartite von Neumann entanglement entropy (half-mode cut)
  - PR   : participation ratio 1/sum|c|^4 (effective # of basis configs)
  - wmax : weight of the dominant computational-basis configuration
and print alongside the XBK r=2 gap (read from the cluster CSVs).

Mechanism under test: the rank-r XBK trial state is a sum of r product
states (Schmidt rank <= r), so the bound should be exact at low S / small PR
and degrade as entanglement grows.
"""
import csv, glob, math
import numpy as np
from openfermion import get_sparse_operator
from openfermion.utils import count_qubits
import xbk_pipeline as xp
import models_mpi as mm


def gs_metrics(H, nq):
    M = get_sparse_operator(H, n_qubits=nq).toarray()
    w, v = np.linalg.eigh(M)
    psi = v[:, 0].astype(complex)
    psi /= np.linalg.norm(psi)
    p = np.abs(psi) ** 2
    PR = 1.0 / np.sum(p ** 2)                      # effective # of configs
    wmax = float(p.max())
    a = nq // 2                                     # half-mode cut
    Apsi = psi.reshape(2 ** a, 2 ** (nq - a))
    s = np.linalg.svd(Apsi, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 1e-14]
    S = float(-np.sum(lam * np.log(lam)))          # von Neumann entropy (nats)
    return S, PR, wmax, float(w[0])


def xbk_gap_from_csv(model, L, ratio):
    for path in glob.glob(f"{model}_xbk_sweep_job*.csv"):
        with open(path) as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith(('ratio', 'U')):
                    continue
                try:
                    r0, l0 = float(row[0]), int(row[1])
                except ValueError:
                    continue
                if abs(r0 - ratio) < 1e-6 and l0 == L and row[4]:
                    return float(row[4])
    return None


# instances spanning the entanglement range (model, L, ratio)
INSTANCES = [
    ('kitaev', 4, 1.0), ('kitaev', 4, 0.5), ('kitaev', 4, 1.5),
    ('j1j2_1d', 4, 0.5), ('j1j2_1d', 4, 0.0), ('j1j2_1d', 6, 0.5),
    ('tv1d', 4, 2.0), ('tv1d', 3, 2.0),
    ('hubbard1d', 2, 4.0), ('hubbard1d', 3, 4.0),
    ('siam', 2, 4.0), ('siam', 3, 4.0), ('siam', 4, 4.0),
    ('hubbard2d', 2, 4.0),
    ('pam', 2, 4.0),
]

print(f"{'model':10s} {'L':>2} {'ratio':>5} {'nq':>3} {'S':>6} {'PR':>7} "
      f"{'wmax':>6} {'gap':>7}")
for name, L, ratio in INSTANCES:
    H = mm.model_hamiltonian(name, ratio, L)
    nq = count_qubits(H)
    S, PR, wmax, E0 = gs_metrics(H, nq)
    gap = xbk_gap_from_csv(name, L, ratio)
    gs = '-' if gap is None else f"{gap:.3f}"
    print(f"{name:10s} {L:>2} {ratio:>5.2f} {nq:>3} {S:>6.3f} {PR:>7.2f} "
          f"{wmax:>6.3f} {gs:>7}", flush=True)
