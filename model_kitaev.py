"""
model_kitaev.py  —  Kitaev chain  (open boundary conditions, L sites)

From the Wikipedia form:

H = -mu sum_{j=1}^{N} (c†_j c_j - 1/2)
  + sum_{j=1}^{N-1} [ -t (c†_{j+1} c_j + c†_j c_{j+1})
                      + |Delta| (c†_{j+1} c†_j + c_j c_{j+1}) ]

Open boundary conditions are standard for the Kitaev chain to expose the
topological edge modes.  The phase boundary is at |mu| = 2t.

Note on pairing ordering:
    c†_{j+1} c†_j = -c†_j c†_{j+1}  (one anti-commutation)
    In openfermion normal-ordered form: c†_j c†_{j+1}  has coefficient +Delta
    (the sign flip is absorbed into the overall phase of Delta).

Mode ordering: mode j  (0-indexed, j = 0,...,L-1)

ratio = Delta/t  (t fixed at 1, mu fixed at 0 — topological sweet spot)
"""

import numpy as np
from openfermion import FermionOperator
from pipeline_base import run_sweep, save_results, save_plot, rank

OUT_DIR    = "output/kitaev"
MODEL_NAME = "Kitaev_chain"

MU = 0.0   # chemical potential fixed at topological sweet spot


def hamiltonian(ratio: float, L: int) -> FermionOperator:
    """
    ratio = Delta/t, t = 1, mu = MU.
    Open boundary conditions: hopping and pairing only for j = 0,...,L-2.
    Chemical potential runs over all j = 0,...,L-1.
    """
    t     = 1.0
    Delta = ratio
    mu    = MU

    H = FermionOperator()

    # Chemical potential: -mu (c†_j c_j - 1/2) = -mu*n_j + mu/2
    for j in range(L):
        H += FermionOperator(((j, 1), (j, 0)), -mu)
        H += FermionOperator((), mu / 2.0)

    for j in range(L - 1):
        jp1 = j + 1

        # Hopping: -t (c†_{j+1} c_j + c†_j c_{j+1})
        H += FermionOperator(((jp1, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (jp1, 0)), -t)

        # Pairing: |Delta| (c†_{j+1} c†_j + h.c.)
        # c†_{j+1} c†_j  in normal order (j < j+1) is -c†_j c†_{j+1}
        # openfermion expects (higher_index, 1), (lower_index, 1) or vice versa;
        # we write it as c†_j c†_{j+1} with coefficient -Delta (absorbing the swap sign)
        # and add h.c. = c_{j+1} c_j with coefficient -Delta*
        H += FermionOperator(((j, 1), (jp1, 1)), -Delta)
        H += FermionOperator(((jp1, 0), (j, 0)), -Delta)

    return H


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 2.01, 0.1), 6))
    L_values     = [4, 6, 8]

    results = run_sweep(hamiltonian, ratio_values, L_values,
                        n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME,
                  ratio_label=r"$|\Delta|/t$",
                  out_dir=OUT_DIR)
