"""
siam_parallel.py
================
Fully MPI-parallelised SIAM ground-state solver for Kamet HPC.

Parallelisation:
  XBK_parallel   -- (terms, j, k) triples striped across ranks
  construct_par  -- 2^n sign tuples striped across ranks
  parallel_sample -- SA reads distributed across ranks
  bm_2_parallel  -- C_dict items striped, MPI.SUM reduce
  p loop         -- sequential, each iteration fully parallel

Run with:
  mpirun -np 128 python siam_parallel.py
"""

import openfermion as op
import itertools
from collections import defaultdict
from math import prod

import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from openfermion import bravyi_kitaev, FermionOperator
from mpi4py import MPI

from Models import build_siam_hamiltonian


# ---------------------------------------------------------------------------
# Hamiltonian builder
# Matches the star-geometry parameterisation used in playground.py / FMQA code
# ---------------------------------------------------------------------------

def siam_hamiltonian(N_bath, U, t, V):
    """
    Build the SIAM FermionOperator.

    Modes:
        0, 1          -> impurity spin-up, spin-down
        2b, 2b+1      -> bath site b (b = 1..N_bath) spin-up, spin-down

    Bath parameters (star-geometry / 1D-chain correspondence):
        eps_b = -2t * cos(b*pi / (N_bath+1))
        V_b   = V * sqrt(2/(N_bath+1)) * sqrt(1 - (eps_b/(2t))^2)
    """
    eps = lambda b: -2.0 * t * np.cos(b * np.pi / (N_bath + 1))
    Vb  = lambda b:  V  * np.sqrt(2.0 / (N_bath + 1)) \
                       * np.sqrt(1.0 - (eps(b) / (2.0 * t)) ** 2)

    # Impurity: U * n_{0,up} * n_{0,dn}  (particle-hole symmetric form)
    H = FermionOperator('0^ 0 1^ 1',  U)
    H += FermionOperator('0^ 0',      -U / 2.0)
    H += FermionOperator('1^ 1',      -U / 2.0)
    H += FermionOperator('',           U / 4.0)

    for b in range(1, N_bath + 1):
        eb = eps(b)
        vb = Vb(b)
        H += FermionOperator(f'{2*b}^ {2*b}',         eb)
        H += FermionOperator(f'{2*b+1}^ {2*b+1}',     eb)
        H += FermionOperator(f'0^ {2*b}',              vb)
        H += FermionOperator(f'{2*b}^ 0',              vb)
        H += FermionOperator(f'1^ {2*b+1}',            vb)
        H += FermionOperator(f'{2*b+1}^ 1',            vb)
    return H


# ---------------------------------------------------------------------------
# Parallel XBK
# ---------------------------------------------------------------------------

def XBK_parallel(H_qubit, n, r, p, comm):
    """
    Build the Z-only Ising Hamiltonian via the XBK embedding.

    The work is the triple loop over (terms, j, k).  We enumerate all
    triples once on every rank (cheap — indices only), then each rank
    processes the subset assigned to it by stripe distribution.

    After local accumulation, rank 0 sums all local results and
    broadcasts the final H_ising to every rank.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Pre-build Z operators once per rank to avoid repeated string parsing
    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]

    # Enumerate all triples — this is just index arithmetic, not algebra
    triples = [
        (terms, coeffs, j, k)
        for terms, coeffs in H_qubit.terms.items()
        for j in range(r)
        for k in range(r)
    ]

    # Stripe distribution: rank i handles triples i, i+size, i+2*size, ...
    my_triples = [t for idx, t in enumerate(triples) if idx % size == rank]

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

    # Gather local results on rank 0, sum, broadcast
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

def construct_parallel(n, r, p, comm):
    """
    Build the normalisation / counting operator C.

    The 2^n sign tuples are striped across ranks.  Each rank builds a
    local C from its assigned tuples; rank 0 sums and broadcasts.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]

    all_signs = list(itertools.product([1, -1], repeat=n))  # length 2^n
    my_signs  = [s for idx, s in enumerate(all_signs) if idx % size == rank]

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
    Distribute SA reads across ranks.

    Each rank runs an independent subset of reads with its own
    SimulatedAnnealingSampler instance.  Only the best (energy, sample)
    from each rank is sent to rank 0, which picks the global best and
    broadcasts it to all ranks.

    Parameters
    ----------
    bqm         : dimod BinaryQuadraticModel
    comm        : MPI communicator
    total_reads : int, total number of SA reads across all ranks
    num_sweeps  : int, sweeps per read
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    reads_per_rank = max(1, total_reads // size)
    # Rank 0 absorbs any remainder reads
    if rank == 0:
        reads_per_rank += total_reads - reads_per_rank * size

    sampler   = SimulatedAnnealingSampler()
    local_ss  = sampler.sample(bqm, num_reads=reads_per_rank,
                               num_sweeps=num_sweeps)

    local_best = (local_ss.first.energy, dict(local_ss.first.sample))
    all_best   = comm.gather(local_best, root=0)

    if rank == 0:
        best_energy, best_sample = min(all_best, key=lambda x: x[0])
    else:
        best_energy, best_sample = None, None

    best_energy = comm.bcast(best_energy, root=0)
    best_sample = comm.bcast(best_sample, root=0)
    return best_energy, best_sample


# ---------------------------------------------------------------------------
# Parallel bm_2 evaluation
# ---------------------------------------------------------------------------

def bm_2_parallel(C_dict, gs, comm):
    """
    Compute  sum_{terms} coeff * prod_{q in key} gs[q]
    with the sum striped across ranks and reduced via MPI.SUM.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    items    = list(C_dict.items())
    my_items = [item for idx, item in enumerate(items) if idx % size == rank]

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
    """Convert a QubitOperator to a plain dict {tuple_of_qubit_indices: coeff}."""
    qubdict = defaultdict(float)
    for terms, coeffs in qubop.terms.items():
        key = tuple(t[0] for t in terms)
        qubdict[key] += coeffs.real
    return qubdict


def reduction(qubdict):
    """Reduce a k-local Ising dict to a 2-local BQM via Rosenberg gadgets."""
    strength = 10 * max(abs(v) for v in qubdict.values())
    return dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)


# ---------------------------------------------------------------------------
# Per-sector Dinkelbach loop
# ---------------------------------------------------------------------------

def run_sector(Q_diag, C, p, comm,
               lam_init=100.0, total_reads=100, num_sweeps=10000):
    """
    Run the Dinkelbach fixed-point iteration for one symmetry sector p.

    Iteration:
      1. Form  H_shift = Q_diag - lam * C
      2. Minimise H_shift via SA  ->  (gse, gs)
      3. If gse >= 0 the root has been bracketed; return current lam
      4. Otherwise update  lam <- lam + gse / <C>_gs

    Returns
    -------
    lam     : converged ground-state energy estimate for sector p
    gs_final: best spin configuration found (None if loop never updated)
    """
    rank      = comm.Get_rank()
    lam       = lam_init
    iteration = 1
    gs_final  = None
    C_dict    = convert_to_dict(C)

    while True:
        if rank == 0:
            print(f"  [p={p}] iteration {iteration}", flush=True)
        iteration += 1

        H_shift   = Q_diag - lam * C
        H_shifted = convert_to_dict(H_shift)
        bqm       = reduction(H_shifted)

        try:
            gse, gs = parallel_sample(bqm, comm,
                                      total_reads=total_reads,
                                      num_sweeps=num_sweeps)
        except ValueError:
            # BQM is trivial (empty); skip this iteration
            if rank == 0:
                print(f"  [p={p}] ValueError in sampler, retrying", flush=True)
            continue

        if rank == 0:
            print(f"  [p={p}] gse = {gse:.6f}  lam = {lam:.6f}", flush=True)

        if gse >= 0:
            if rank == 0:
                print(f"  [p={p}] converged at lam = {lam:.8f}", flush=True)
            break

        gs_final = gs
        bm_2     = bm_2_parallel(C_dict, gs, comm)
        lam      = lam + gse / bm_2

        if rank == 0:
            print(f"  [p={p}] updated lam = {lam:.8f}", flush=True)

    return lam, gs_final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ------------------------------------------------------------------
    # Physical parameters — edit these to sweep over U, V, N_bath, etc.
    # ------------------------------------------------------------------
    N_BATH     = 6       # number of bath sites
    U          = 4.0     # on-site Coulomb repulsion on impurity
    T_HOP      = 1.0     # bath hopping amplitude (sets half-bandwidth W = 2t)
    V_HYB      = 1.0     # impurity-bath hybridisation
    R          = 3       # XBK replication factor
    TOTAL_READS  = 100   # total SA reads per Dinkelbach iteration
    NUM_SWEEPS   = 10000 # SA sweeps per read

    # ------------------------------------------------------------------
    # Build Hamiltonian — all ranks build identically to avoid large
    # object broadcasts
    # ------------------------------------------------------------------
    Q_fermion = siam_hamiltonian(N_BATH, U, T_HOP, V_HYB)
    Q         = bravyi_kitaev(Q_fermion)
    n         = op.count_qubits(Q)

    p_values  = list(range(R // 2 + 1))  # sectors p = 0, 1, ..., floor(R/2)

    if rank == 0:
        print("=" * 60, flush=True)
        print(f"SIAM  N_bath={N_BATH}  U={U}  t={T_HOP}  V={V_HYB}", flush=True)
        print(f"n={n} qubits  |  r={R}  |  sectors={p_values}", flush=True)
        print(f"MPI ranks={size}", flush=True)
        print(f"XBK triples total : {len(list(Q.terms.items())) * R * R}", flush=True)
        print(f"construct tuples  : 2^{n} = {2**n}", flush=True)
        print(f"SA reads total    : {TOTAL_READS}", flush=True)
        print("=" * 60, flush=True)

    # ------------------------------------------------------------------
    # Main loop over symmetry sectors
    # ------------------------------------------------------------------
    gseig    = []
    gs_list  = []

    for p in p_values:

        if rank == 0:
            print(f"\n--- sector p={p} ---", flush=True)

        Q_diag = XBK_parallel(Q, n, R, p, comm)
        C      = construct_parallel(n, R, p, comm)

        lam, gs_final = run_sector(
            Q_diag, C, p, comm,
            lam_init=100.0,
            total_reads=TOTAL_READS,
            num_sweeps=NUM_SWEEPS,
        )

        gseig.append(lam)
        gs_list.append(gs_final)

    # ------------------------------------------------------------------
    # Summary — rank 0 only
    # ------------------------------------------------------------------
    if rank == 0:
        best_idx = int(np.argmin(gseig))

        print("\n" + "=" * 60, flush=True)
        print("RESULTS", flush=True)
        print("=" * 60, flush=True)
        for p_val, lam_val in zip(p_values, gseig):
            marker = "  <-- best" if p_val == p_values[best_idx] else ""
            print(f"  p={p_val}  E0={lam_val:.8f}{marker}", flush=True)

        print(f"\nGround-state energy : {gseig[best_idx]:.8f}", flush=True)
        print(f"Best sector p       : {p_values[best_idx]}", flush=True)

        best_gs = gs_list[best_idx]
        if best_gs is not None:
            print(f"\nGround-state spins (r*n Ising space):", flush=True)
            print(best_gs, flush=True)
            print(f"\nOriginal n-qubit spins (replica block 0):", flush=True)
            print([best_gs.get(i) for i in range(n)], flush=True)
        else:
            print("\nWarning: no ground-state sample recorded.", flush=True)