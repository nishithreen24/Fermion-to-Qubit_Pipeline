"""
import openfermion as op
from openfermion import QubitOperator
Q1 = op.QubitOperator('Z1 Z2 Z3 Z5')
Q2 = op.QubitOperator(())#('Z0 Z1 Y2 X3')
Q = Q1 * Q2
#print(Q)
n = 6
j = 2
k = 3
#i = 0
Qbar = op.QubitOperator(())
for terms, coeffs in Q.terms.items():
    qc = -1
    for i in range(n-1, -1, -1):
        paulij = op.QubitOperator('Z' + str(((j - 1) * n + i)))
        paulik = op.QubitOperator('Z' + str(((k - 1) * n + i)))
        if i < terms[0][0]:
            #print("aye")
            #pauli = 'Y' + str(i)
            pauliop = (paulij*paulik + 1)*(0.5)
            #Qbar *= op.QubitOperator(pauli)
            Qbar *= pauliop
        elif i > terms[qc][0]:
            #print("aye")
            #pauli = 'Y' + str(i)
            pauliop = (paulij * paulik + 1)*(0.5)
            #Qbar *= op.QubitOperator(pauli)
            Qbar *= pauliop
        elif i==terms[qc][0]:
            #print("no aye")
            #pauli = 'X' + str(i)
            #Qbar *= op.QubitOperator(pauli)
            if terms[qc][1] == 'X':
                pauliop = ((paulij*paulik)-1)*(-0.5)
            elif terms[qc][1] == 'Y':
                pauliop = (paulik-paulij)*(0.5j)
            elif terms[qc][1] == 'Z':
                pauliop = (paulij+paulik)*(0.5)
            Qbar *= pauliop
            qc -= 1
        print(pauliop)
        #else:
            #print("no aye")
            #if(terms[i][0]==i):
                #pauli = 'X'+str(i)
                #print(pauli)
                #Qbar *= op.QubitOperator(pauli)
                #print(Qbar)
            #else:
                #pauli = 'Y'+str(i)
                #Qbar *= op.QubitOperator(pauli)
        #print(terms[i])
print()
print(Qbar)


#for terms, coeffs in Q.terms.items():
#    print(terms)
#    print(coeffs)
"""

import openfermion as op
import itertools

from dimod import BinaryQuadraticModel


def XBK(H_qubit, n, r):
    H_ising = op.QubitOperator()
    for terms, coeffs in H_qubit.terms.items():
        for j in range(r):
            Sj = -1 if j < r//2 else 1
            for k in range(r):
                if j == k: continue          # skip diagonal
                Sk = -1 if k < r//2 else 1
                Qbar = op.QubitOperator(())
                term_idx = len(terms) - 1   # start at highest qubit index in term
                for i in range(n-1, -1, -1):
                    paulij = op.QubitOperator('Z' + str((j * n + i)))
                    paulik = op.QubitOperator('Z' + str((k * n + i)))
                    if term_idx >= 0 and i == terms[term_idx][0]:
                        p = terms[term_idx][1]
                        if p == 'X':
                            pauliop = (paulij * paulik - 1) * (-0.5)
                        elif p == 'Y':
                            pauliop = (paulik - paulij) * 0.5j
                        elif p == 'Z':
                            pauliop = (paulij + paulik) * 0.5
                        term_idx -= 1        # consume this Pauli, move to next lower
                    else:
                        pauliop = (paulij * paulik + 1) * 0.5   # identity
                    Qbar *= pauliop
                H_ising += coeffs * Sj * Sk * Qbar
    return H_ising

def construct(n, r):#, S):
    C = op.QubitOperator()
    k = 0
    for signs in (itertools.product([1, -1], repeat=n)):
        bitstring = op.QubitOperator()
        print(k+1)
        k+=1
        for j in range(r):
            Sj = -1 if j < r//2 else 1
            prod = op.QubitOperator(())
            for i in range(n):
                pauli = op.QubitOperator('Z' + str((j * n + i)))
                prod *= (pauli*signs[i]+1)*(0.5)
            bitstring += prod*Sj
        C += bitstring*bitstring
    return C

from collections import defaultdict

def convert_to_dict(qubop):
    qubdict = defaultdict(float)
    for terms, coeffs in qubop.terms.items():
        key = tuple(t[0] for t in terms)
        val = complex(coeffs)
        if abs(val.imag) > 1e-10:  # catch genuine errors, not just noise
            raise ValueError(f'Large imaginary part {val.imag} in term {terms}')
        qubdict[key] += float(val.real)  # explicit float strips floating point noise
    return qubdict

import dimod

def reduction(qubdict):
    strength = 10 * max(abs(v) for v in qubdict.values())
    bqm = dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)
    return bqm

def initial_lambda(H_dict):
    return sum(abs(v) for v in H_dict.values())

#from dwave.samplers import SimulatedAnnealingSampler

def solver(H_qubit, n, r):
    l = initial_lambda(convert_to_dict(H_ising))



def build_S(r):
    S = []
    for i in range(r):
        if i < r//2:
            S.append(-1)
        else:
            S.append(1)
    return S

#print()
Q1 = op.QubitOperator('Z0 X3 Y7 Z8')
Q2 = op.QubitOperator('Y1 Z2 Z4 X6')
#print(XBK(Q1+Q2, 9, 4))#, build_S(4)))
#print(Construct(9, 4))
#print(Construct(9, 4, [1, -1, 1, 1]))
#for signs in (itertools.product([1, -1], repeat=2)):
    #print(signs)
#print(op.QubitOperator('Z0')*(-1))
#print(op.get_sparse_operator(XBK(Q1+Q2, 9, 4, build_S(4)), 36))
#print(build_S(5))
from Models import *
H_fermionic = build_siam_hamiltonian(N=2, U=4.0, t=1.0, V=1.0, eps_imp=0.0)
H_qubit = op.bravyi_kitaev(H_fermionic)
print(H_qubit)
H_ising = XBK(H_qubit, n = 6, r=3)
print(H_ising)
#print(Construct(n=6, r=3))
print("here")
print(reduction((convert_to_dict(H_ising))))
bqm = reduction(convert_to_dict(H_ising))

# Verify 2-locality explicitly
assert all(len(k) == 2 for k in bqm.quadratic.keys()), "Not 2-local!"
print(f"✓ 2-local BQM with {len(bqm.variables)} spins, {len(bqm.quadratic)} couplings")

# The auxiliary spins introduced by make_quadratic
original_vars = [v for v in bqm.variables if isinstance(v, int)]
aux_vars = [v for v in bqm.variables if isinstance(v, str)]
print(f"  Original spins: {len(original_vars)}")
print(f"  Auxiliary spins added for locality reduction: {len(aux_vars)}")