"""
model_hubbard1d.py  —  1D Fermi-Hubbard model  (periodic ring of L sites)

H = -t sum_{j,sigma} (c†_{j,sigma} c_{j+1,sigma} + h.c.)
    + U sum_j n_{j,up} n_{j,down}

Mode ordering: mode = 2*j + sigma  (sigma: 0=up, 1=down)
ratio = U/t  (t fixed at 1)
"""

import numpy as np
from openfermion import FermionOperator
from pipeline_base import run_sweep, save_results, save_plot, rank

OUT_DIR    = "output/hubbard1d"
MODEL_NAME = "1D_Hubbard"


def _mode(j, sigma):
    return 2 * j + sigma


def hamiltonian(ratio: float, L: int) -> FermionOperator:
    t, U = 1.0, ratio
    H = FermionOperator()
    for j in range(L):
        jp = (j + 1) % L
        for sigma in range(2):
            H += FermionOperator(((_mode(j, sigma), 1), (_mode(jp, sigma), 0)), -t)
            H += FermionOperator(((_mode(jp, sigma), 1), (_mode(j, sigma), 0)), -t)
        up, dn = _mode(j, 0), _mode(j, 1)
        H += FermionOperator(((up, 1), (dn, 1), (dn, 0), (up, 0)), U)
    return H


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 8.01, 0.4), 6))
    L_values     = [2, 4, 6]

    results = run_sweep(hamiltonian, ratio_values, L_values,
                        n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME, ratio_label=r"$U/t$", out_dir=OUT_DIR)
