import openfermion as op
import itertools
from collections import defaultdict
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from openfermion import bravyi_kitaev, FermionOperator
from math import prod
from mpi4py import MPI

from Models import *


# ---------------------------------------------------------------------------
# Parallel XBK
# ---------------------------------------------------------------------------

def XBK_parallel(H_qubit, n, r, p, comm):
    """
    Distributes the (terms, j, k) triple loop across MPI ranks.
    Each rank builds a local H_ising from its assigned triples,
    then all local results are summed into rank 0.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]

    # Build the full list of (terms, coeffs, j, k) triples on every rank.
    # This is cheap — we are just enumerating indices, not doing any algebra.
    triples = [
        (terms, coeffs, j, k)
        for terms, coeffs in H_qubit.terms.items()
        for j in range(r)
        for k in range(r)
    ]

    # Stripe-distribute triples across ranks
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

    # Gather all local H_ising to rank 0 and sum
    all_local = comm.gather(H_local, root=0)
    if rank == 0:
        H_ising = op.QubitOperator()
        for h in all_local:
            H_ising += h
    else:
        H_ising = None

    # Broadcast the final result to all ranks so everyone has it
    H_ising = comm.bcast(H_ising, root=0)
    return H_ising


# ---------------------------------------------------------------------------
# Parallel construct
# ---------------------------------------------------------------------------

def construct_parallel(n, r, p, comm):
    """
    Distributes the 2^n signs loop across MPI ranks.
    Each rank builds a local C from its assigned sign tuples,
    then all local results are summed into rank 0.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]

    all_signs = list(itertools.product([1, -1], repeat=n))  # 2^n entries
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

def parallel_sample(bqm, comm, total_reads=100, num_sweeps=10000):
    """
    Each MPI rank runs an independent subset of SA reads.
    Only the best (energy, sample) per rank is sent to rank 0,
    which finds the global best and broadcasts it to all ranks.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    reads_per_rank = max(1, total_reads // size)
    if rank == 0:
        reads_per_rank += total_reads - reads_per_rank * size

    sampler = SimulatedAnnealingSampler()
    local_ss = sampler.sample(bqm, num_reads=reads_per_rank, num_sweeps=num_sweeps)

    local_best = (local_ss.first.energy, dict(local_ss.first.sample))
    all_best = comm.gather(local_best, root=0)

    if rank == 0:
        best_energy, best_sample = min(all_best, key=lambda x: x[0])
    else:
        best_energy, best_sample = None, None

    best_energy = comm.bcast(best_energy, root=0)
    best_sample = comm.bcast(best_sample, root=0)
    return best_energy, best_sample


# ---------------------------------------------------------------------------
# Parallel bm_2
# ---------------------------------------------------------------------------

def bm_2_parallel(C_dict, gs, comm):
    """
    Distributes the sum over C_dict terms across MPI ranks.
    Each rank computes a partial sum; rank 0 reduces to the total.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    items = list(C_dict.items())
    my_items = [item for i, item in enumerate(items) if i % size == rank]

    local_sum = sum(
        coeff * (prod(gs[q] for q in key) if key else 1)
        for key, coeff in my_items
    )

    total = comm.reduce(local_sum, op=MPI.SUM, root=0)
    total = comm.bcast(total, root=0)
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convert_to_dict(qubop):
    qubdict = defaultdict(float)
    for terms, coeffs in qubop.terms.items():
        key = tuple(t[0] for t in terms)
        qubdict[key] += coeffs.real
    return qubdict


def reduction(qubdict):
    strength = 10 * max(abs(v) for v in qubdict.values())
    return dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)


def siam(N_bath, U, t, V):
    eps = lambda b: -2 * t * np.cos(b * np.pi / (N_bath + 1))
    Vb  = lambda b:  V * np.sqrt(2 / (N_bath + 1)) * np.sqrt(1 - (eps(b) / (2 * t)) ** 2)
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
# Main per-p loop
# ---------------------------------------------------------------------------

def run_p(Q_diag, C, p, comm):
    rank = comm.Get_rank()
    lam = 100
    iteration = 1
    gs_final = None
    C_dict = convert_to_dict(C)

    while True:
        if rank == 0:
            print(f"p={p} Iteration {iteration}", flush=True)
        iteration += 1

        H_shift = Q_diag - lam * C
        H_shifted = convert_to_dict(H_shift)
        bqm = reduction(H_shifted)

        try:
            gse, gs = parallel_sample(bqm, comm, total_reads=100, num_sweeps=10000)
        except ValueError:
            continue

        if gse >= 0:
            if rank == 0:
                print(f"p={p} converged: lambda={lam}", flush=True)
            break

        gs_final = gs
        bm_2 = bm_2_parallel(C_dict, gs, comm)
        lam = lam + (gse / bm_2)

        if rank == 0:
            print(f"p={p} lambda'={lam}", flush=True)

    return lam, gs_final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    r = 3
    p_values = list(range(r // 2 + 1))

    # All ranks build Q identically — avoids broadcasting a large object
    Q_fermion = siam(N_bath=6, U=4.0, t=1.0, V=1.0)
    Q = bravyi_kitaev(Q_fermion)
    n = op.count_qubits(Q)

    if rank == 0:
        print(f"n={n} qubits | r={r} | p_values={p_values} | {size} MPI ranks", flush=True)
        print(f"construct: 2^{n}={2**n} sign tuples split across {size} ranks", flush=True)
        print(f"XBK: {len(list(Q.terms.items()))*r*r} triples split across {size} ranks", flush=True)
        print(f"SA: 100 reads split across {size} ranks", flush=True)

    gseig = []
    gs_list = []
    for p in p_values:
        if rank == 0:
            print(f"\n--- p={p} ---", flush=True)

        Q_diag = XBK_parallel(Q, n, r, p, comm)
        C      = construct_parallel(n, r, p, comm)
        lam, gs_final = run_p(Q_diag, C, p, comm)
        gseig.append(lam)
        gs_list.append(gs_final)

    if rank == 0:
        best_idx = int(np.argmin(gseig))
        print("\n=== Results ===")
        for p, lam in zip(p_values, gseig):
            print(f"  p={p}  lambda={lam}")
        print(f"\nGround state energy : {gseig[best_idx]}")
        print(f"Best p              : {p_values[best_idx]}")
        best_gs = gs_list[best_idx]
        print(f"Ground state spins (rn-qubit space) : {best_gs}")
        if best_gs is not None:
            print(f"Original n-qubit spins (block 0)    : {[best_gs.get(i) for i in range(n)]}")