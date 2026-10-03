"""
model_siam.py  —  Single Impurity Anderson Model

H = U (n_{imp,up} - 1/2)(n_{imp,down} - 1/2)
  + sum_{b,sigma} eps_b n_{b,sigma}
  + sum_{b,sigma} V_b (a†_{imp,sigma} a_{b,sigma} + h.c.)

Bath parameters (star-geometry / 1D chain, t = V = 1):
    eps_b = -2t cos(b*pi / (N_bath+1))
    V_b   = V sqrt(2/(N_bath+1)) sqrt(1 - (eps_b/(2t))^2)

Mode ordering: mode = 2*site + spin  (site 0 = impurity, sites 1..N = bath)

ratio = U/V  (V = t = 1)
L = N_bath (number of bath sites)
"""

import numpy as np
from openfermion import FermionOperator
from pipeline_base import run_sweep, save_results, save_plot, rank

OUT_DIR    = "output/siam"
MODEL_NAME = "SIAM"


def _mode(site, spin):
    return 2 * site + spin


def hamiltonian(ratio: float, L: int) -> FermionOperator:
    U, t, V = ratio, 1.0, 1.0
    b_idx    = np.arange(1, L + 1)
    eps_bath = -2.0 * t * np.cos(b_idx * np.pi / (L + 1))
    V_bath   = V * np.sqrt(2.0 / (L + 1)) * np.sqrt(
                    np.maximum(0.0, 1.0 - (eps_bath / (2.0 * t)) ** 2))

    H = FermionOperator()

    # H_imp = U(n_up-1/2)(n_dn-1/2) = U n_up n_dn - (U/2)(n_up+n_dn) + U/4
    up, dn = _mode(0, 0), _mode(0, 1)
    H += FermionOperator(((up, 1), (dn, 1), (dn, 0), (up, 0)), U)
    for spin in range(2):
        H += FermionOperator(((_mode(0, spin), 1), (_mode(0, spin), 0)), -U / 2)
    H += FermionOperator((), U / 4)

    # H_bath
    for b_site, eps_b in zip(range(1, L + 1), eps_bath):
        for spin in range(2):
            p = _mode(b_site, spin)
            H += FermionOperator(((p, 1), (p, 0)), float(eps_b))

    # H_hyb
    for b_site, V_b in zip(range(1, L + 1), V_bath):
        for spin in range(2):
            imp_m, bath_m = _mode(0, spin), _mode(b_site, spin)
            H += FermionOperator(((imp_m, 1), (bath_m, 0)), float(V_b))
            H += FermionOperator(((bath_m, 1), (imp_m, 0)), float(V_b))

    return H


if __name__ == "__main__":
    ratio_values = list(np.round(np.arange(0.0, 8.01, 0.4), 6))
    L_values     = [1, 2, 3]   # L = N_bath

    results = run_sweep(hamiltonian, ratio_values, L_values,
                        n_sweeps=2000, penalty_scale=12.0)
    if rank == 0:
        save_results(results, OUT_DIR, MODEL_NAME)
        save_plot(results, MODEL_NAME, ratio_label=r"$U/V$", out_dir=OUT_DIR)
