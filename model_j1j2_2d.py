"""
model_j1j2_2d.py  —  2D J1-J2 Heisenberg model  (L x L square lattice, periodic)

H = J1 sum_{<i,j>} S_i . S_j  +  J2 sum_{<<i,j>>} S_i . S_j

<i,j>   = nearest neighbours  (horizontal + vertical bonds)
<<i,j>> = next-nearest neighbours  (diagonal bonds)
S_i . S_j = (1/4)(X_i X_j + Y_i Y_j + Z_i Z_j)

Spin model: BK is skipped.
ratio = J2/J1  (J1 fixed at 1).  L is the linear dimension (total sites = L*L).
"""

import numpy as np
from openfermion import QubitOperator
from pipeline_base import (parse_qubit_op, run_sweep_qubit,
                            save_results, save_plot, rank)

OUT_DIR    = "output/j1j2_2d"
MODEL_NAME = "2D_J1J2"


def _site(ix, iy, L):
    return iy * L + ix


def qubit_hamiltonian(ratio: float, L: int):
    """Returns (bk_terms, n_qubits)."""
    J1, J2 = 1.0, ratio
    H = QubitOperator()
    for iy in range(L):
        for ix in range(L):
            i = _site(ix, iy, L)
            # Nearest neighbours
            for jx, jy in [((ix+1)%L, iy), (ix, (iy+1)%L)]:
                j = _site(jx, jy, L)
                for p in ('X', 'Y', 'Z'):
                    H += QubitOperator(f'{p}{i} {p}{j}', J1 / 4.0)
            # Next-nearest neighbours (diagonals)
            for jx, jy in [((ix+1)%L, (iy+1)%L), ((ix-1)%L, (iy+1)%L)]:
                j = _site(jx, jy, L)
                for p in ('X', 'Y', 'Z'):
                    H += QubitOperator(f'{p}{i} {p}{j}', J2 / 4.0)
    return parse_qubit_op(H), L * L


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 1.01, 0.1), 6))
    L_values     = [2, 3, 4]

    results = run_sweep_qubit(qubit_hamiltonian, ratio_values, L_values,
                              n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME, ratio_label=r"$J_2/J_1$", out_dir=OUT_DIR)
