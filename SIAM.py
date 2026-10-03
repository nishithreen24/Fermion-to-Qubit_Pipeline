import openfermion as op

def XBK_fast(H_qubit, n, r, p):

    H_ising = op.QubitOperator()

    # precompute Z operators
    Z = [op.QubitOperator(f'Z{i}') for i in range(r*n)]

    for terms, coeff in H_qubit.terms.items():

        for j in range(r):

            Sj = -1 if j < p else 1

            for k in range(r):

                Sk = -1 if k < p else 1

                qc = len(terms) - 1
                Qbar = op.QubitOperator(())

                for i in range(n-1, -1, -1):

                    Zj = Z[j*n + i]
                    Zk = Z[k*n + i]

                    if len(terms) == 0 or i < terms[0][0] or i > terms[qc][0]:

                        pauliop = (Zj*Zk + 1) * 0.5

                    elif i == terms[qc][0]:

                        pauli = terms[qc][1]

                        if pauli == 'X':
                            pauliop = (1 - Zj*Zk) * 0.5

                        elif pauli == 'Y':
                            pauliop = (Zk - Zj) * (0.5j)

                        elif pauli == 'Z':
                            pauliop = (Zj + Zk) * 0.5

                        qc -= 1

                    Qbar *= pauliop

                H_ising += coeff * Sj * Sk * Qbar

    return H_ising

def construct_fast(n, r, p):

    C = op.QubitOperator()

    Z = [op.QubitOperator(f'Z{i}') for i in range(r*n)]

    for j in range(r):

        Sj = -1 if j < p else 1

        for k in range(r):

            Sk = -1 if k < p else 1

            prod = op.QubitOperator(())

            for i in range(n):

                Zj = Z[j*n + i]
                Zk = Z[k*n + i]

                prod *= (1 + Zj*Zk) * 0.5

            C += Sj * Sk * prod

    return C

import dimod

def qubitop_to_bqm(qubop):

    linear = {}
    quadratic = {}
    offset = 0

    for term, coeff in qubop.terms.items():

        if len(term) == 0:
            offset += coeff.real

        elif len(term) == 1:
            linear[term[0][0]] = coeff.real

        elif len(term) == 2:
            i = term[0][0]
            j = term[1][0]
            quadratic[(i,j)] = coeff.real

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.SPIN)


import numpy as np
import matplotlib.pyplot as plt
import openfermion as of
from openfermion import bravyi_kitaev
from scipy.sparse.linalg import eigsh

from Models import build_siam_hamiltonian
from playground import XBK, construct, convert_to_dict, reduction
from neal import SimulatedAnnealingSampler
from math import prod


# ------------------------------------------------------------
# Exact diagonalization
# ------------------------------------------------------------

def exact_ground_energy(H_qubit):

    sparse = of.linalg.get_sparse_operator(H_qubit)
    eig, _ = eigsh(sparse, k=1, which='SA')
    return eig[0]


# ------------------------------------------------------------
# Run full pipeline
# ------------------------------------------------------------

def run_pipeline(H_fermion, r):

    Q = bravyi_kitaev(H_fermion)
    n = of.count_qubits(Q)

    sampler = SimulatedAnnealingSampler()

    energies = []

    # only use p = 0 and p = r
    for p in [0, r]:

        Q_diag = XBK_fast(Q, n, r, p)

        C = construct_fast(n, r, p)

        lam = 100

        while True:

            H_shift = Q_diag - lam * C

            bqm = qubitop_to_bqm(H_shift)

            sampleset = sampler.sample(
                bqm, num_sweep=10000,
                num_reads=1000
            )

            gse = sampleset.first.energy

            if gse >= 0:
                break

            gs = dict(sampleset.first.sample)

            C_dict = C.terms

            bm_2 = 0

            for key, coeff in C_dict.items():

                if key == ():
                    bm_2 += coeff
                else:
                    prod_val = 1
                    for q,_ in key:
                        prod_val *= gs[q]
                    bm_2 += coeff * prod_val

            lam = lam + gse / bm_2

        energies.append(lam)

    return min(energies)

def run_pipeline_return_samples(H_fermion, r=3):

    Q = bravyi_kitaev(H_fermion)
    n = of.count_qubits(Q)

    sampler = SimulatedAnnealingSampler()

    best_energy = None
    best_sampleset = None

    for p in [0, r]:

        Q_diag = XBK_fast(Q, n, r, p)

        C = construct_fast(n, r, p)

        lam = 100

        while True:

            H_shift = Q_diag - lam * C

            bqm = qubitop_to_bqm(H_shift)

            sampleset = sampler.sample(
                bqm, num_sweeps=10000,
                num_reads=1000
            )

            gse = sampleset.first.energy

            if gse >= 0:
                break

            gs = dict(sampleset.first.sample)

            C_dict = C.terms

            bm_2 = 0

            for key, coeff in C_dict.items():

                if key == ():
                    bm_2 += coeff
                else:
                    prod_val = 1
                    for q,_ in key:
                        prod_val *= gs[q]
                    bm_2 += coeff * prod_val

            lam = lam + gse / bm_2

        if best_energy is None or lam < best_energy:
            best_energy = lam
            best_sampleset = sampleset

    return best_energy, best_sampleset


# ------------------------------------------------------------
# Experiment 1
# Ground state vs U
# ------------------------------------------------------------

def experiment_U_sweep(N=3):

    U_values = np.linspace(0, 8, 10)

    exact = []
    annealed = []

    for U in U_values:

        H = build_siam_hamiltonian(N=N, U=U)

        Q = bravyi_kitaev(H)

        exact_energy = exact_ground_energy(Q)

        approx_energy = run_pipeline(H, r=2)

        exact.append(exact_energy)
        annealed.append(approx_energy)

        print(U, exact_energy, approx_energy)

    plt.plot(U_values, exact, label="Exact")
    plt.plot(U_values, annealed, 'o', label="Annealer")
    plt.xlabel("U")
    plt.ylabel("Ground State Energy")
    plt.legend()
    plt.title("SIAM Ground State vs U")
    plt.show()


# ------------------------------------------------------------
# Experiment 2
# replication factor study
# ------------------------------------------------------------

def experiment_replication(N=3, U=4):

    r_values = range(2,6)

    errors = []

    H = build_siam_hamiltonian(N=N, U=U)

    Q = bravyi_kitaev(H)

    exact_energy = exact_ground_energy(Q)

    for r in r_values:

        approx_energy = run_pipeline(H, r=r)

        error = abs(approx_energy - exact_energy)

        errors.append(error)

        print("r=",r,"error=",error)

    plt.plot(r_values, errors, 'o-')
    plt.xlabel("Replication factor r")
    plt.ylabel("Energy error")
    plt.title("XBK replication convergence")
    plt.show()


# ------------------------------------------------------------
# Experiment 3
# bath size scaling
# ------------------------------------------------------------

def experiment_bath_scaling():

    bath_sizes = [1,2,3,4]

    energies = []

    for N in bath_sizes:

        H = build_siam_hamiltonian(N=N, U=4)

        energy = run_pipeline(H, r=3)

        energies.append(energy)

    plt.plot(bath_sizes, energies, 'o-')

    plt.xlabel("Bath sites")
    plt.ylabel("Ground State Energy")

    plt.title("SIAM energy vs bath size")

    plt.show()

def impurity_observables(sampleset):

    samples = sampleset.record.sample

    n_up = []
    n_dn = []

    for s in samples:

        # spin variables
        z_up = s[0]
        z_dn = s[1]

        # convert to occupation
        nup = (1 - z_up)/2
        ndn = (1 - z_dn)/2

        n_up.append(nup)
        n_dn.append(ndn)

    n_up = np.array(n_up)
    n_dn = np.array(n_dn)

    n_imp = n_up + n_dn
    double_occ = n_up * n_dn
    local_moment = (n_up - n_dn)**2

    return {
        "n_imp" : np.mean(n_imp),
        "double_occ" : np.mean(double_occ),
        "local_moment" : np.mean(local_moment)
    }

def impurity_observable_vs_U():

    U_values = np.linspace(0,8,12)

    double_occ = []
    moment = []

    for U in U_values:

        H = build_siam_hamiltonian(N=3, U=U)

        energy, sampleset = run_pipeline_return_samples(H)

        obs = impurity_observables(sampleset)

        double_occ.append(obs["double_occ"])
        moment.append(obs["local_moment"])

    plt.plot(U_values, double_occ, label="Double occupancy")
    plt.plot(U_values, moment, label="Local moment")

    plt.xlabel("U")
    plt.ylabel("Observable")
    plt.legend()
    plt.show()

def spin_correlation(sampleset):

    samples = sampleset.record.sample

    corr = []

    for s in samples:

        z_imp = s[0] - s[1]
        z_bath = s[2] - s[3]

        corr.append(z_imp * z_bath)

    return np.mean(corr)

def particle_number_penalty(n_qubits, target):

    penalty = op.QubitOperator()

    for i in range(n_qubits):
        Zi = op.QubitOperator(f"Z{i}")
        ni = (1 - Zi)/2
        penalty += ni

    return (penalty - target)**2

from openfermion.transforms import symmetry_conserving_bravyi_kitaev

def BK_with_tapering(H):

    Q, sym_vals = symmetry_conserving_bravyi_kitaev(H)

    print("Detected symmetries:", sym_vals)
    print("Reduced qubit count:", of.count_qubits(Q))

    return Q

import openfermion as op
from itertools import product

def XBK_vectorized(H_qubit, n, r, p):

    H_ising = op.QubitOperator()

    for terms, coeff in H_qubit.terms.items():

        for j in range(r):

            Sj = -1 if j < p else 1

            for k in range(r):

                Sk = -1 if k < p else 1

                # each pauli expands into 2 possibilities
                expansions = [(1.0, [])]

                for qubit, pauli in terms:

                    zj = j*n + qubit
                    zk = k*n + qubit

                    new_expansions = []

                    for c, ops in expansions:

                        if pauli == 'X':

                            new_expansions.append(
                                (c*0.5, ops)
                            )

                            new_expansions.append(
                                (-c*0.5, ops + [(zj,'Z'),(zk,'Z')])
                            )

                        elif pauli == 'Y':

                            new_expansions.append(
                                (c*0.5j, ops + [(zk,'Z')])
                            )

                            new_expansions.append(
                                (-c*0.5j, ops + [(zj,'Z')])
                            )

                        elif pauli == 'Z':

                            new_expansions.append(
                                (c*0.5, ops + [(zj,'Z')])
                            )

                            new_expansions.append(
                                (c*0.5, ops + [(zk,'Z')])
                            )

                    expansions = new_expansions

                # add contributions
                for c, ops in expansions:

                    term = op.QubitOperator(tuple(ops), coeff*Sj*Sk*c)

                    H_ising += term

    return H_ising

import openfermion as op

def XBK_fast_correct(H_qubit, n, r, p):

    H_ising = op.QubitOperator()

    # precompute Z operators
    Z = [op.QubitOperator(f'Z{i}') for i in range(r*n)]

    for terms, coeff in H_qubit.terms.items():

        term_dict = dict(terms)

        for j in range(r):

            Sj = -1 if j < p else 1

            for k in range(r):

                Sk = -1 if k < p else 1

                Qbar = op.QubitOperator(())

                for i in range(n):

                    Zj = Z[j*n + i]
                    Zk = Z[k*n + i]

                    if i not in term_dict:

                        pauliop = (1 + Zj*Zk) * 0.5

                    else:

                        pauli = term_dict[i]

                        if pauli == 'X':

                            pauliop = (1 - Zj*Zk) * 0.5

                        elif pauli == 'Y':

                            pauliop = (Zk - Zj) * (0.5j)

                        elif pauli == 'Z':

                            pauliop = (Zj + Zk) * 0.5

                    Qbar *= pauliop

                H_ising += coeff * Sj * Sk * Qbar

    return H_ising

# ------------------------------------------------------------

if __name__ == "__main__":

    print("Running SIAM experiments")

    print("\n1) Energy vs U")
    experiment_U_sweep()

    print("\n2) Replication factor study")
    experiment_replication()

    print("\n3) Bath size scaling")
    experiment_bath_scaling()

    print("\n4) Impurity observables vs U")
    impurity_observable_vs_U()