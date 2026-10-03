"""
tv_model.py
===========
Standalone re-implementation of the 1D spinless t-V chain, matching the
call signature used in playground.py (tV_hamiltonian_1d(L, t, V, mu,
periodic)) and the mode-ordering convention documented in
PROJECT_MEMORY.md ("t-V (spinless) | mode = j"). Models.py itself is not
available in project knowledge, so this is a clean re-derivation from the
standard spinless t-V Hamiltonian:

    H = -t * sum_<i,j> (c_i^dag c_j + c_j^dag c_i)
        + V * sum_<i,j> n_i n_j
        - mu * sum_i n_i

with nearest-neighbour bonds <i,j>, periodic or open boundary conditions.
"""

from openfermion import FermionOperator


def tV_hamiltonian_1d(L, t, V, mu, periodic=False):
    H = FermionOperator()
    if periodic:
        bonds = [(j, (j + 1) % L) for j in range(L)]
    else:
        bonds = [(j, j + 1) for j in range(L - 1)]

    for (i, j) in bonds:
        H += FermionOperator(((i, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (i, 0)), -t)
        H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)

    for j in range(L):
        H += FermionOperator(((j, 1), (j, 0)), -mu)

    return H