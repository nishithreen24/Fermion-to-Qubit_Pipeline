"""
model_hubbard2d.py  —  2D Fermi-Hubbard model  (L x L square lattice, periodic)

H = -t sum_{<i,j>,sigma} (c†_{i,sigma} c_{j,sigma} + h.c.)
    + U sum_i n_{i,up} n_{i,down}

Mode ordering: mode = 2*site + sigma  (site = iy*L + ix)
ratio = U/t  (t fixed at 1).  L is the linear dimension.
"""

import numpy as np
from openfermion import FermionOperator
from pipeline_base import run_sweep, save_results, save_plot, rank

OUT_DIR    = "output/hubbard2d"
MODEL_NAME = "2D_Hubbard"


def _site(ix, iy, L):
    return iy * L + ix


def _mode(site, sigma):
    return 2 * site + sigma


def hamiltonian(ratio: float, L: int) -> FermionOperator:
    t, U = 1.0, ratio
    H = FermionOperator()
    for iy in range(L):
        for ix in range(L):
            i = _site(ix, iy, L)
            for jx, jy in [((ix+1)%L, iy), (ix, (iy+1)%L)]:
                j = _site(jx, jy, L)
                for sigma in range(2):
                    H += FermionOperator(((_mode(i, sigma), 1), (_mode(j, sigma), 0)), -t)
                    H += FermionOperator(((_mode(j, sigma), 1), (_mode(i, sigma), 0)), -t)
            up, dn = _mode(i, 0), _mode(i, 1)
            H += FermionOperator(((up, 1), (dn, 1), (dn, 0), (up, 0)), U)
    return H


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 8.01, 0.8), 6))
    L_values     = [2, 3]

    results = run_sweep(hamiltonian, ratio_values, L_values,
                        n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME, ratio_label=r"$U/t$", out_dir=OUT_DIR)
