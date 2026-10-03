"""
parallel_siam_v1.py
===================
MPI-parallelised version of playground.py for Kamet HPC.

Parallelised:
  - XBK      : (terms, j, k) triples striped across ranks
  - construct : 2^n sign tuples striped across ranks
  - SA reads  : num_reads split across ranks

Plots are saved to siam_output/ (no display needed).
Logs go to logs/ via SLURM.

Run:
  mpirun -np <N> python3 parallel_siam_v1.py
"""

import openfermion as op
import itertools
from collections import defaultdict
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from openfermion import bravyi_kitaev, FermionOperator
from math import prod
from mpi4py import MPI
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------------------------------------------------------
# Parallel XBK
# ---------------------------------------------------------------------------

def XBK(H_qubit, n, r, p, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]

    triples = [
        (terms, coeffs, j, k)
        for terms, coeffs in H_qubit.terms.items()
        for j in range(r)
        for k in range(r)
    ]
    my_triples = [t for i, t in enumerate(triples) if i % size == rank]

    H_local = op.QubitOperator()
    for terms, coeffs, j, k in my_triples:
        Sj = -1 if j < p else 1
        Sk = -1 if k < p else 1
        qc = -1
        Qbar = op.QubitOperator(())
        for i in range(n - 1, -1, -1):
            Zj = Z[j * n + i]
            Zk = Z[k * n + i]
            if len(terms) == 0 or i < terms[0][0] or i > terms[qc][0]:
                pauliop = (Zj * Zk + 1) * 0.5
            elif i == terms[qc][0]:
                if terms[qc][1] == 'X':
                    pauliop = ((Zj * Zk) - 1) * (-0.5)
                elif terms[qc][1] == 'Y':
                    pauliop = (Zk - Zj) * 0.5j
                elif terms[qc][1] == 'Z':
                    pauliop = (Zj + Zk) * 0.5
                qc -= 1
            Qbar *= pauliop
        H_local += coeffs * Sj * Sk * Qbar

    all_local = comm.gather(H_local, root=0)
    if rank == 0:
        H_ising = op.QubitOperator()
        for h in all_local:
            H_ising += h
    else:
        H_ising = None
    H_ising = comm.bcast(H_ising, root=0)
    return H_ising


# ---------------------------------------------------------------------------
# Parallel construct
# ---------------------------------------------------------------------------

def construct(n, r, p, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]
    all_signs = list(itertools.product([1, -1], repeat=n))
    my_signs = [s for i, s in enumerate(all_signs) if i % size == rank]

    C_local = op.QubitOperator()
    for signs in my_signs:
        bitstring = op.QubitOperator()
        for j in range(r):
            Sj = -1 if j < p else 1
            prod_op = op.QubitOperator(())
            for i in range(n):
                pauli = Z[j * n + i]
                prod_op *= (pauli * signs[i] + 1) * 0.5
            bitstring += prod_op * Sj
        C_local += bitstring * bitstring

    all_local = comm.gather(C_local, root=0)
    if rank == 0:
        C = op.QubitOperator()
        for c in all_local:
            C += c
    else:
        C = None
    C = comm.bcast(C, root=0)
    return C


# ---------------------------------------------------------------------------
# Parallel SA sampling
# ---------------------------------------------------------------------------

def sample_parallel(bqm, comm, num_reads=100, num_sweeps=10000):
    rank = comm.Get_rank()
    size = comm.Get_size()

    reads_per_rank = max(1, num_reads // size)
    if rank == 0:
        reads_per_rank += num_reads - reads_per_rank * size

    sampler = SimulatedAnnealingSampler()
    ss = sampler.sample(bqm, num_reads=reads_per_rank, num_sweeps=num_sweeps)
    local_best = (ss.first.energy, dict(ss.first.sample))

    all_best = comm.gather(local_best, root=0)
    if rank == 0:
        best_energy, best_sample = min(all_best, key=lambda x: x[0])
    else:
        best_energy, best_sample = None, None
    best_energy = comm.bcast(best_energy, root=0)
    best_sample = comm.bcast(best_sample, root=0)
    return best_energy, best_sample


# ---------------------------------------------------------------------------
# Helpers — identical to serial
# ---------------------------------------------------------------------------

def convert_to_dict(qubop):
    qubdict = defaultdict(float)
    for terms, coeffs in qubop.terms.items():
        key = tuple(t[0] for t in terms)
        qubdict[key] += coeffs.real
    return qubdict


def reduction(qubdict):
    strength = 10 * max(abs(v) for v in qubdict.values())
    bqm = dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)
    return bqm


def siam(N_bath, U, t, V):
    eps = lambda b: -2 * t * np.cos(b * np.pi / (N_bath + 1))
    Vb  = lambda b: V * np.sqrt(2 / (N_bath + 1)) \
                      * np.sqrt(1 - (eps(b) / (2 * t)) ** 2)
    H = FermionOperator('0^ 0 1^ 1', U)
    H += FermionOperator('0^ 0', -U / 2)
    H += FermionOperator('1^ 1', -U / 2)
    H += FermionOperator('', U / 4)
    for b in range(1, N_bath + 1):
        H += FermionOperator(f'{2*b}^ {2*b}',     eps(b))
        H += FermionOperator(f'{2*b+1}^ {2*b+1}', eps(b))
        H += FermionOperator(f'0^ {2*b}',          Vb(b))
        H += FermionOperator(f'{2*b}^ 0',          Vb(b))
        H += FermionOperator(f'1^ {2*b+1}',        Vb(b))
        H += FermionOperator(f'{2*b+1}^ 1',        Vb(b))
    return H


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Output folder — created by rank 0 only
    OUT_DIR = "siam_output"
    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)

    t = 1.0
    V = 1.0

    Q = bravyi_kitaev(siam(N_bath=2, U=4.0, t=t, V=V))

    if rank == 0:
        print("Qubit Hamiltonian")
        print(Q)
        print()

    n = op.count_qubits(Q)

    if rank == 0:
        print("No. of qubits:", n)
        print()
        print()
        print()

    r = 2
    gseig = []

    for p in range(r // 2 + 1):
        if rank == 0:
            print("r =", r)
            print("p =", p)

        Q_diag = XBK(Q, n, r, p, comm)

        if rank == 0:
            print()
            print()

        C = construct(n, r, p, comm)
        C_dict = convert_to_dict(C)

        lam = 100
        iteration = 1

        while True:
            if rank == 0:
                print("Iteration:", iteration)
            iteration += 1

            H_shift = Q_diag - lam * C
            H_shifted = convert_to_dict(H_shift)
            bqm = reduction(H_shifted)

            try:
                gse, gs = sample_parallel(bqm, comm,
                                          num_reads=100, num_sweeps=10000)
            except ValueError:
                continue

            if gse >= 0:
                if rank == 0:
                    print("it's over")
                break

            if rank == 0:
                print("\\sum_mb_m^2\\lambda^'-\\lambda =", gse)
                print(gs)
                print("it's never over")

            bm_2 = sum(
                coeff * (prod(gs[q] for q in key) if key else 1)
                for key, coeff in C_dict.items()
            )

            if rank == 0:
                print("\\sum_mb_m^2 =", bm_2)

            if abs(bm_2) < 1e-10:
                if rank == 0:
                    print("  bm_2 ~ 0: unphysical sample, resampling.")
                continue

            if rank == 0:
                print("\\lamda^'-\\lambda =", gse / bm_2)

            lamnew = lam + (gse / bm_2)

            if rank == 0:
                print("\\lambda' = ", lamnew)
                print()

            lam = lamnew

        gseig.append(lam)

        if rank == 0:
            print()
            print()

    if rank == 0:
        print(gseig)

    # -----------------------------------------------------------------------
    # Plot — rank 0 only, saved to siam_output/
    # -----------------------------------------------------------------------
    if rank == 0:
        best_p = gseig.index(min(gseig))
        diag = reduction(convert_to_dict(XBK(Q, n, r=r, p=best_p, comm=comm)))

        G = dimod.to_networkx_graph(diag)
        pos = nx.spring_layout(G)

        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=20, font_size=3)
        ax.set_title(f"QUBO graph  |  r={r}  p={best_p}  E0={min(gseig):.6f}")

        plot_path = os.path.join(OUT_DIR, f"qubo_graph_r{r}_p{best_p}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {plot_path}")