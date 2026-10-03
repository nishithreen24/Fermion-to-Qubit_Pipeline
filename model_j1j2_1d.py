"""
model_j1j2_1d.py  —  1D J1-J2 frustrated Heisenberg chain  (periodic, L sites)

H = J1 sum_j S_j . S_{j+1}  +  J2 sum_j S_j . S_{j+2}
S_i . S_j = (1/4)(X_i X_j + Y_i Y_j + Z_i Z_j)

Spin model: BK is skipped.  QubitOperator parsed directly into bk_terms format.
ratio = J2/J1  (J1 fixed at 1)
"""

import numpy as np
from openfermion import QubitOperator
from pipeline_base import (parse_qubit_op, run_sweep_qubit,
                            save_results, save_plot, rank)

OUT_DIR    = "output/j1j2_1d"
MODEL_NAME = "1D_J1J2"


def qubit_hamiltonian(ratio: float, L: int):
    """Returns (bk_terms, n_qubits)."""
    J1, J2 = 1.0, ratio
    H = QubitOperator()
    for j in range(L):
        jp1 = (j + 1) % L
        jp2 = (j + 2) % L
        for p in ('X', 'Y', 'Z'):
            H += QubitOperator(f'{p}{j} {p}{jp1}', J1 / 4.0)
            H += QubitOperator(f'{p}{j} {p}{jp2}', J2 / 4.0)
    return parse_qubit_op(H), L


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 1.01, 0.05), 6))
    L_values     = [4, 6, 8]

    results = run_sweep_qubit(qubit_hamiltonian, ratio_values, L_values,
                              n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME, ratio_label=r"$J_2/J_1$", out_dir=OUT_DIR)
