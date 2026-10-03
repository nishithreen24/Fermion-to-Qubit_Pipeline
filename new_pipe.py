import numpy as np
from collections import defaultdict
from openfermion import QubitOperator, count_qubits, bravyi_kitaev
from dwave.samplers import SimulatedAnnealingSampler
import dimod
import math
def qubit_to_ising(qop):
    """
    Convert OpenFermion QubitOperator → {(i,j,...): coeff}
    Only Z terms survive after XBK
    """
    ising = defaultdict(float)

    for term, coeff in qop.terms.items():
        key = tuple(i for i, _ in term)
        ising[key] += coeff.real

    return dict(ising)
def xbk_transform(op, r, p):
    n = count_qubits(op)
    new_op = QubitOperator()

    for term, coeff in op.terms.items():
        out = QubitOperator()

        for j in range(r):
            for k in range(r):
                sign = 1 if (j < p) == (k < p) else -1
                sub = QubitOperator('', 1)

                idx = 0
                for i in range(n):
                    if idx < len(term) and term[idx][0] == i:
                        pauli = term[idx][1]
                        idx += 1
                    else:
                        pauli = 'I'

                    ij = i + n*j
                    ik = i + n*k

                    if pauli == 'X':
                        if j == k:
                            sub = QubitOperator('', 0)
                            break
                        sub *= QubitOperator('', 0.5) - QubitOperator(f'Z{ij} Z{ik}', 0.5)

                    elif pauli == 'Y':
                        if j == k:
                            sub = QubitOperator('', 0)
                            break
                        sub *= QubitOperator(f'Z{ik}', 0.5j) - QubitOperator(f'Z{ij}', 0.5j)

                    elif pauli == 'Z':
                        if j == k:
                            sub *= QubitOperator(f'Z{ij}')
                        else:
                            sub *= QubitOperator(f'Z{ij}', 0.5) + QubitOperator(f'Z{ik}', 0.5)

                    else:  # Identity
                        if j != k:
                            sub *= QubitOperator('', 0.5) + QubitOperator(f'Z{ij} Z{ik}', 0.5)

                out += sign * sub

        new_op += coeff * out

    new_op.compress()
    return new_op
import itertools

def counting_operator(n, r):
    C = QubitOperator('', 0)

    for perm in itertools.product([1, -1], repeat=n):
        term = QubitOperator('', 0)

        for j in range(r):
            prod = QubitOperator('', 1)

            for i in range(n):
                prod *= QubitOperator('', 0.5) + QubitOperator(f'Z{i + n*j}', perm[i]/2)

            term += prod

        C += term**2

    return C
def eval_ising(ising, spins):
    E = 0.0
    for mono, coeff in ising.items():
        prod = 1.0
        for i in mono:
            prod *= spins.get(i, 1)  # ignores auxiliaries
        E += coeff * prod
    return E
def quadratize(poly, M):
    return dimod.make_quadratic(poly, M, dimod.SPIN)
def dinkelbach_sa(H, C, n_phys, M, reads=200, max_iter=30):
    sampler = SimulatedAnnealingSampler()
    lam = 0.0
    best = np.inf

    for _ in range(max_iter):
        combined = defaultdict(float, H)
        for k, v in C.items():
            combined[k] -= lam * v

        bqm = quadratize(dict(combined), M)

        sampleset = sampler.sample(bqm, num_reads=reads)
        sample = sampleset.first.sample

        spins = {i: sample[i] for i in sample}

        # 🔥 THIS IS THE KEY FIX
        Hv = eval_ising(H, spins)
        Cv = eval_ising(C, spins)

        if abs(Cv) < 1e-8:
            continue

        new_lam = Hv / Cv

        if new_lam < best:
            best = new_lam

        if abs(new_lam - lam) < 1e-6:
            break

        lam = new_lam

    return best
def solve_xbk(qubit_op, r=2):
    n = count_qubits(qubit_op)

    C_op = counting_operator(n, r)
    C = qubit_to_ising(C_op)

    best = np.inf

    for p in range(int(math.ceil(r/2 + 1))):
        H_op = xbk_transform(qubit_op, r, p)
        H = qubit_to_ising(H_op)

        M = 10 * max(abs(v) for v in H.values())

        E = dinkelbach_sa(H, C, n*r, M)

        best = min(best, E)

    return best
from openfermion import QubitOperator
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



#H = QubitOperator('X0') + 0.5 * QubitOperator('Z0 Z1')
for i in np.arange(0, 8.1, 0.4):
    H = bravyi_kitaev(hamiltonian(i, 2))

    E = solve_xbk(H, r=2)

    print("Ground energy:", E)


