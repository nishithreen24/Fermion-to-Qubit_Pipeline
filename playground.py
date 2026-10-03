import openfermion as op
import itertools
from collections import defaultdict
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from openfermion import bravyi_kitaev


def XBK(H_qubit, n, r, p):#, S):
    H_ising = op.QubitOperator()
    #Z = [[op.QubitOperator('Z' + str(j * n + i)) for i in range(n)] for j in range(r)]
    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]
    for terms, coeffs in H_qubit.terms.items():
        for j in range(r):#[1]:#range(r):
            Sj = -1 if j < p else 1
            for k in range(r):
                Sk = -1 if k < p else 1
                qc = -1
                Qbar = op.QubitOperator(())
                for i in range(n-1, -1, -1):
                    Zj = Z[j * n + i]
                    Zk = Z[k * n + i]
                    if len(terms) == 0 or i < terms[0][0] or i > terms[qc][0]:
                        pauliop = (Zj * Zk + 1) * (0.5)
                    elif i==terms[qc][0]:
                        if terms[qc][1] == 'X':
                            pauliop = ((Zj * Zk) - 1) * (-0.5)
                        elif terms[qc][1] == 'Y':
                            pauliop = (Zk - Zj) * (0.5j)
                        elif terms[qc][1] == 'Z':
                            pauliop = (Zj + Zk) * (0.5)
                        qc -= 1
                    Qbar *= pauliop
                H_ising += coeffs*Sj*Sk*Qbar
    return H_ising#-QubitOperator(())

def construct(n, r, p):
    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]
    C = op.QubitOperator()
    m = 0
    for signs in itertools.product([1, -1], repeat=n):
        print(m+1)
        m+=1
        bitstring = op.QubitOperator()
        for j in range(r):
            Sj = -1 if j < p else 1
            prod_op = op.QubitOperator(())
            for i in range(n):
                pauli = Z[j*n+i]#op.QubitOperator('Z' + str((j * n + i)))
                prod_op *= (pauli * signs[i] + 1) * 0.5
            bitstring += prod_op * Sj
        C += bitstring * bitstring
    return C

def convert_to_dict(qubop):
        qubdict = defaultdict(float)
        for terms, coeffs in qubop.terms.items():
            key = tuple(t[0] for t in terms)  # built in one shot
            qubdict[key] += coeffs.real
        return qubdict

def reduction(qubdict):
    strength = 50*max(abs(v) for v in qubdict.values())
    bqm = dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)
    return bqm
def siam(N_bath, U, t, V):
    eps = lambda b: -2*t * np.cos(b * np.pi / (N_bath + 1))
    Vb  = lambda b:  V * np.sqrt(2/(N_bath+1)) * np.sqrt(1 - (eps(b)/(2*t))**2)
    H = FermionOperator('0^ 0 1^ 1', U)
    H += FermionOperator('0^ 0', -U / 2)
    H += FermionOperator('1^ 1', -U / 2)
    H += FermionOperator('', U / 4)
    for b in range(1, N_bath + 1):
        H += FermionOperator(f'{2 * b}^ {2 * b}', eps(b))
        H += FermionOperator(f'{2 * b + 1}^ {2 * b + 1}', eps(b))
        H += FermionOperator(f'0^ {2 * b}', Vb(b))
        H += FermionOperator(f'{2 * b}^ 0', Vb(b))
        H += FermionOperator(f'1^ {2 * b + 1}', Vb(b))
        H += FermionOperator(f'{2 * b + 1}^ 1', Vb(b))
    return H

from Models import *
from math import prod

from openfermion import FermionOperator
Us = np.arange(0.0, 10.5, 0.5)
Es = []
Cs = []
for p in range(2//2+1):
    Cs.append(construct(op.count_qubits(bravyi_kitaev(siam(2, 1, 1, 1))), 2, p))
for U in Us:
    #h = -0.5  # Hartree = -13.6057 eV
    t = 1.0
    V = 2.0
    #H = FermionOperator('0^ 0', h)
    #print(H)
    #print()
    # Q = bravyi_kitaev(H)
    #Q = bravyi_kitaev(tV_hamiltonian_2d(nx=8, ny=1, t=t, V=V, mu=0.0, periodic_x=True, periodic_y=True)[0])
    #Q = bravyi_kitaev(tV_hamiltonian_1d(L = 2, t = 1.0, V = 1.0, mu = 0.0, periodic=False))
    Q = bravyi_kitaev(siam(N_bath=2, U=U, t=1.0, V=1.0))
    print("Qubit Hamiltonian")
    print(Q)
    print()
    n = op.count_qubits(Q)
    print("No. of qubits:", n)
    r = 2
    print()
    print()
    print()
    gseig = []
    for p in range(r//2+1):
        print("r =", r)
        print("p =", p)
        Q_diag = XBK(Q, n, r, p)
        print()
        # print(Q_diag)
        print()
        # print(op.linalg.get_sparse_operator(Q_diag))
        C = Cs[p]
        # print(op.linalg.get_sparse_operator(C))
        lam = 100
        iteration = 1

        while True:
            print("Iteration:", iteration)
            iteration += 1
            H_shift = Q_diag - lam * C
            H_shifted = convert_to_dict(H_shift)
            sampler = SimulatedAnnealingSampler()
            bqm = reduction(H_shifted)
            try:
                sampleset = sampler.sample(bqm, num_sweeps=50000, num_reads=500)
            except ValueError:
                continue
            gse = sampleset.first.energy
            if gse >= 0 or abs(gse) < 1e-8:
                print("it's over")
                break
            print("\\sum_mb_m^2\\lambda^'-\\lambda =", gse)
            gs = dict(sampleset.first.sample)
            print(gs)
            C_dict = convert_to_dict(C)
            print("it's never over")
            bm_2 = sum(
                coeff * (prod(gs[q] for q in key) if key else 1)
                for key, coeff in C_dict.items()
            )

            print("\\sum_mb_m^2 =", bm_2)
            if abs(bm_2) < 1e-10:
                print("  bm_2 ~ 0: unphysical sample, resampling.")
                continue
            print("\\lamda^'-\\lambda =", gse / bm_2)

            lamnew = lam + (gse / bm_2)
            print("\\lambda' = ", lamnew)
            lam = lamnew
            print()
        gseig.append(lam)
        print()
        print()

    print(gseig)
    Es.append(min(gseig))
"""
    import networkx as nx
    import matplotlib.pyplot as plt

    diag = reduction(convert_to_dict(XBK(Q, n, r=r, p=gseig.index(min(gseig)))))

    G = dimod.to_networkx_graph(diag)

    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=20, font_size=3
    )
    plt.show()
"""
import matplotlib.pyplot as plt
plt.plot(Us, Es)
plt.show()