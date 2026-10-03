import openfermion as op
import itertools
from collections import defaultdict
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from openfermion import bravyi_kitaev


def XBK(H_qubit, n, r, p):#, S):
    H_ising = op.QubitOperator()
    for terms, coeffs in H_qubit.terms.items():
        for j in range(r):#[1]:#range(r):
            Sj = -1 if j < p else 1
            for k in range(r):
                if j == k: continue#[2]:#range(j+1, r):
                Sk = -1 if k < p else 1
                qc = -1
                Qbar = op.QubitOperator(())
                for i in range(n-1, -1, -1):
                    paulij = op.QubitOperator('Z' + str((j * n + i)))
                    paulik = op.QubitOperator('Z' + str((k * n + i)))
                    if len(terms) == 0 or i < terms[0][0] or i > terms[qc][0]:
                        pauliop = (paulij * paulik + 1) * (0.5)
                    #if i < terms[0][0]:
                        #pauliop = (paulij * paulik + 1) * (0.5)
                    #elif i > terms[qc][0]:
                        #pauliop = (paulij * paulik + 1) * (0.5)
                    elif i==terms[qc][0]:
                        if terms[qc][1] == 'X':
                            pauliop = ((paulij * paulik) - 1) * (-0.5)
                        elif terms[qc][1] == 'Y':
                            pauliop = (paulik - paulij) * (0.5j)
                        elif terms[qc][1] == 'Z':
                            pauliop = (paulij + paulik) * (0.5)
                        qc -= 1
                    Qbar *= pauliop
                H_ising += coeffs*Sj*Sk*Qbar
    return H_ising#-QubitOperator(())

def construct(n, r, p):
    C = op.QubitOperator()
    for signs in itertools.product([1, -1], repeat=n):
        bitstring = op.QubitOperator()
        for j in range(r):
            Sj = -1 if j < p else 1
            prod_op = op.QubitOperator(())
            for i in range(n):
                pauli = op.QubitOperator('Z' + str((j * n + i)))
                prod_op *= (pauli * signs[i] + 1) * 0.5
            bitstring += prod_op * Sj
        C += bitstring * bitstring
    return C

def convert_to_dict(qubop):
    qubdict = defaultdict(float)
    for terms, coeffs in qubop.terms.items():
        key = ()
        for i in range(len(terms)):
            key = key + (terms[i][0], )
        qubdict[key] += coeffs.real
    return qubdict

def reduction(qubdict):
    strength = max(abs(v) for v in qubdict.values())
    bqm = dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)
    return bqm

from Models import *
from math import prod

from openfermion import FermionOperator

h = -0.5  # Hartree = -13.6057 eV

H = FermionOperator('0^ 0', h)
#H += FermionOperator('1^ 1', h)
print(H)
print()
Q = bravyi_kitaev(H)
#Q = bravyi_kitaev(tV_hamiltonian_2d(nx = 2, ny = 2, t = 1.0, V = 2.0, mu = 0.0, periodic_x=False, periodic_y=False)[0])
print(Q)
print()
n = op.count_qubits(Q)
print(n)
r = 15
p = 1
Q_diag = XBK(Q, n, r, p)
print()
print(Q_diag)
#print(op.linalg.get_sparse_operator(Q_diag))
C = construct(n, r, p)
#print(op.linalg.get_sparse_operator(C))
lam = 100
iteration = 1
while True:
    print("Iteration:", iteration)
    iteration += 1
    H_shift = Q_diag - lam * C
    H_shifted = convert_to_dict(H_shift)
    sampler = SimulatedAnnealingSampler()
    bqm = reduction(H_shifted)
    sampleset = sampler.sample(bqm, num_sweeps=10000, num_reads=100)

    gse = sampleset.first.energy
    if gse >= 0:
        print("it's over")
        break
    print(gse)
    gs = dict(sampleset.first.sample)
    C_dict = convert_to_dict(C)
    print("it's never over")
    bm_2 = sum(
        coeff * (prod(gs[q] for q in key) if key else 1)
        for key, coeff in C_dict.items()
    )

    print(bm_2)
    print(gse / bm_2)

    lamnew = lam + (gse / bm_2)
    print(lamnew)
    lam = lamnew
    print()
    print()


print(lam)