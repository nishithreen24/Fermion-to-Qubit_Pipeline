"""
model_tv1d.py  —  1D spinless t-V model  (periodic ring of L sites)

H = -t sum_j (c†_j c_{j+1} + h.c.)  +  V sum_j (n_j - 1/2)(n_{j+1} - 1/2)

ratio = V/t  (t fixed at 1)
"""

import numpy as np
from openfermion import FermionOperator
from pipeline_base import run_sweep, save_results, save_plot, rank

OUT_DIR    = "output/tv1d"
MODEL_NAME = "1D_tV"


def hamiltonian(ratio: float, L: int) -> FermionOperator:
    t, V = 1.0, ratio
    H = FermionOperator()
    for j in range(L):
        jp = (j + 1) % L
        H += FermionOperator(((j, 1), (jp, 0)), -t)
        H += FermionOperator(((jp, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (j, 0), (jp, 1), (jp, 0)), V)
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator(((jp, 1), (jp, 0)), -V / 2)
        H += FermionOperator((), V / 4)
    return H


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 4.01, 0.2), 6))
    L_values     = [2, 4, 6]

    results = run_sweep(hamiltonian, ratio_values, L_values,
                        n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME, ratio_label=r"$V/t$", out_dir=OUT_DIR)
