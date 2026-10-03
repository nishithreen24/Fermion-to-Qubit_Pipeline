"""
model_pam.py  —  Periodic Anderson Model  (real-space, 1D ring of L sites)

Real-space version of the PAM from the Wikipedia heavy-fermion form,
mapped to a 1D lattice with periodic boundary conditions:

H = -t  sum_{j,sigma} (c†_{j,sigma} c_{j+1,sigma} + h.c.)   [conduction hopping]
  + eps_f sum_{j,sigma} f†_{j,sigma} f_{j,sigma}             [f-level energy]
  + U  sum_j n^f_{j,up} n^f_{j,down}                         [f-level Hubbard U]
  + V  sum_{j,sigma} (c†_{j,sigma} f_{j,sigma} + h.c.)       [local c-f hybridisation]

Particle-hole symmetric choice: eps_f = -U/2
(so the f-level is half-filled at any U, analogous to the SIAM convention).

Mode ordering per site j  (total sites L, total modes 4*L):
    4*j + 0  ->  c_{j, up}     (conduction, spin up)
    4*j + 1  ->  c_{j, down}   (conduction, spin down)
    4*j + 2  ->  f_{j, up}     (localised f, spin up)
    4*j + 3  ->  f_{j, down}   (localised f, spin down)

ratio = U/t  (t = V = 1, eps_f = -U/2)
"""

import numpy as np
from openfermion import FermionOperator
from pipeline_base import run_sweep, save_results, save_plot, rank

OUT_DIR    = "output/pam"
MODEL_NAME = "PAM"


def _c(j: int, sigma: int) -> int:
    """Conduction electron mode: 4*j + sigma."""
    return 4 * j + sigma


def _f(j: int, sigma: int) -> int:
    """f-electron mode: 4*j + 2 + sigma."""
    return 4 * j + 2 + sigma


def hamiltonian(ratio: float, L: int) -> FermionOperator:
    """
    ratio = U/t, t = V = 1, eps_f = -U/2 (particle-hole symmetric).
    Periodic boundary conditions: site (L-1) hops to site 0.
    """
    t     = 1.0
    V_hyb = 1.0
    U     = ratio
    eps_f = -U / 2.0

    H = FermionOperator()

    for j in range(L):
        jp = (j + 1) % L

        # Conduction hopping: -t (c†_{j,sigma} c_{j+1,sigma} + h.c.)
        for sigma in range(2):
            H += FermionOperator(((_c(j, sigma), 1), (_c(jp, sigma), 0)), -t)
            H += FermionOperator(((_c(jp, sigma), 1), (_c(j, sigma), 0)), -t)

        # f-level on-site energy: eps_f * (n^f_{j,up} + n^f_{j,down})
        for sigma in range(2):
            H += FermionOperator(((_f(j, sigma), 1), (_f(j, sigma), 0)), eps_f)

        # f-level Hubbard U: U n^f_{j,up} n^f_{j,down}
        # Normal-ordered: a†_{f,up} a†_{f,dn} a_{f,dn} a_{f,up}  (one swap, sign +1)
        f_up, f_dn = _f(j, 0), _f(j, 1)
        H += FermionOperator(((f_up, 1), (f_dn, 1), (f_dn, 0), (f_up, 0)), U)

        # Local c-f hybridisation: V (c†_{j,sigma} f_{j,sigma} + h.c.)
        for sigma in range(2):
            H += FermionOperator(((_c(j, sigma), 1), (_f(j, sigma), 0)), V_hyb)
            H += FermionOperator(((_f(j, sigma), 1), (_c(j, sigma), 0)), V_hyb)

    return H


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 8.01, 0.8), 6))
    L_values     = [2, 3]

    results = run_sweep(hamiltonian, ratio_values, L_values,
                        n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME,
                  ratio_label=r"$U/t$",
                  out_dir=OUT_DIR)
