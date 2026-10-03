from collections import defaultdict

import openfermion as op
import itertools

from openfermion import bravyi_kitaev
from collections import defaultdict

def XBK(H_qubit, n, r):#, S):
    H_ising = op.QubitOperator()
    for terms, coeffs in H_qubit.terms.items():
        for j in range(r):#[1]:#range(r):
            Sj = -1 if j < r//2 else 1
            for k in range(j, r):#[2]:#range(j+1, r):
                Sk = -1 if j < r//2 else 1
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

def Construct(n, r):#, S):
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

def build_S(r):
    S = []
    for i in range(r):
        if i < r//2:
            S.append(-1)
        else:
            S.append(1)
    return S

def convert_to_dict(H_ising):
    H_reduced = defaultdict(list)
    for terms, coeffs in H_ising.terms.items():
        term = []
        for i in range(len(terms)):
            term.append(terms[i][0])
        H_reduced[tuple(term)] = coeffs


#print()
Q1 = op.QubitOperator('Z0 X3')
Q2 = op.QubitOperator('Y1 Z2 Z4')
#print(XBK(Q1+Q2, 9, 4))#, build_S(4)))
#print(Construct(5, 3))
#print(Construct(9, 4, [1, -1, 1, 1]))
#for signs in (itertools.product([1, -1], repeat=2)):
    #print(signs)
#print(op.QubitOperator('Z0')*(-1))
#print(op.get_sparse_operator(XBK(Q1+Q2, 9, 4, build_S(4)), 36))
#print(build_S(5))

from Models import *

H_fermion = build_siam_hamiltonian(N=2, U=4, t=1, V=1, eps_imp=0)
H_qubit = bravyi_kitaev(H_fermion)
print(H_qubit)
H_ising = XBK(H_qubit, n = 6, r = 3)
print()
print(H_ising)
print()
#print(Construct(n=6, r=3))
print(convert_to_dict(H_ising))