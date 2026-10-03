"""
model_tv2d.py  —  2D spinless t-V model  (L x L square lattice, periodic)

H = -t sum_{<i,j>} (c†_i c_j + h.c.)  +  V sum_{<i,j>} (n_i - 1/2)(n_j - 1/2)

ratio = V/t  (t fixed at 1).  L is the linear dimension; total sites = L*L.
"""

import numpy as np
from openfermion import FermionOperator
from pipeline_base import run_sweep, save_results, save_plot, rank

OUT_DIR    = "output/tv2d"
MODEL_NAME = "2D_tV"


def _site(ix, iy, L):
    return iy * L + ix


def hamiltonian(ratio: float, L: int) -> FermionOperator:
    t, V = 1.0, ratio
    H = FermionOperator()
    for iy in range(L):
        for ix in range(L):
            i = _site(ix, iy, L)
            for jx, jy in [((ix+1)%L, iy), (ix, (iy+1)%L)]:
                j = _site(jx, jy, L)
                H += FermionOperator(((i, 1), (j, 0)), -t)
                H += FermionOperator(((j, 1), (i, 0)), -t)
                H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)
                H += FermionOperator(((i, 1), (i, 0)), -V / 2)
                H += FermionOperator(((j, 1), (j, 0)), -V / 2)
                H += FermionOperator((), V / 4)
    return H


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 4.01, 0.4), 6))
    L_values     = [2, 3]

    results = run_sweep(hamiltonian, ratio_values, L_values,
                        n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME, ratio_label=r"$V/t$", out_dir=OUT_DIR)
