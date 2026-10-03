"""
siam_local.py
=============
Multiprocessing-parallelised SIAM solver for a 4-core laptop.

Uses Python's multiprocessing.Pool to parallelise:
  - XBK      : (terms, j, k) triples split across workers
  - construct : 2^n sign tuples split across workers
  - SA reads  : num_reads split across workers

No MPI required. Run directly:
    python siam_local.py
"""

import openfermion as op
import itertools
from collections import defaultdict
from math import prod
import multiprocessing as mp

import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from openfermion import bravyi_kitaev, FermionOperator

from Models import *

# Number of parallel workers — set to number of physical cores
N_WORKERS = 4


# ---------------------------------------------------------------------------
# Hamiltonian builder
# ---------------------------------------------------------------------------

def siam_hamiltonian(N_bath, U, t, V):
    eps = lambda b: -2.0 * t * np.cos(b * np.pi / (N_bath + 1))
    Vb  = lambda b: V * np.sqrt(2.0 / (N_bath + 1)) \
                      * np.sqrt(1.0 - (eps(b) / (2.0 * t)) ** 2)

    H = FermionOperator('0^ 0 1^ 1',  U)
    H += FermionOperator('0^ 0',      -U / 2.0)
    H += FermionOperator('1^ 1',      -U / 2.0)
    H += FermionOperator('',           U / 4.0)

    for b in range(1, N_bath + 1):
        eb, vb = eps(b), Vb(b)
        H += FermionOperator(f'{2*b}^ {2*b}',     eb)
        H += FermionOperator(f'{2*b+1}^ {2*b+1}', eb)
        H += FermionOperator(f'0^ {2*b}',          vb)
        H += FermionOperator(f'{2*b}^ 0',          vb)
        H += FermionOperator(f'1^ {2*b+1}',        vb)
        H += FermionOperator(f'{2*b+1}^ 1',        vb)
    return H


# ---------------------------------------------------------------------------
# XBK — worker function (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _xbk_worker(args):
    """
    Process a subset of (terms, coeffs, j, k) triples and return a
    partial QubitOperator.

    args = (triples_subset, n, r, p)
    """
    triples_subset, n, r, p = args

    # Rebuild Z operators inside the worker — QubitOperator is picklable
    # but pre-built lists are cheap to reconstruct
    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]

    H_local = op.QubitOperator()

    for terms, coeffs, j, k in triples_subset:
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

    return H_local


def XBK_parallel(H_qubit, n, r, p, n_workers=N_WORKERS):
    """
    Parallel XBK: distribute (terms, j, k) triples across workers.
    """
    triples = [
        (terms, coeffs, j, k)
        for terms, coeffs in H_qubit.terms.items()
        for j in range(r)
        for k in range(r)
    ]

    # Split triples into n_workers roughly equal chunks
    chunks = [triples[i::n_workers] for i in range(n_workers)]
    # Remove empty chunks if triples < n_workers
    chunks = [c for c in chunks if c]

    args = [(chunk, n, r, p) for chunk in chunks]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_xbk_worker, args)

    H_ising = op.QubitOperator()
    for h in results:
        H_ising += h

    return H_ising


# ---------------------------------------------------------------------------
# construct — worker function
# ---------------------------------------------------------------------------

def _construct_worker(args):
    """
    Process a subset of sign tuples and return a partial QubitOperator C.

    args = (signs_subset, n, r, p)
    """
    signs_subset, n, r, p = args

    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]
    C_local = op.QubitOperator()

    for signs in signs_subset:
        bitstring = op.QubitOperator()
        for j in range(r):
            Sj = -1 if j < p else 1
            prod_op = op.QubitOperator(())
            for i in range(n):
                pauli = Z[j * n + i]
                prod_op *= (pauli * signs[i] + 1) * 0.5
            bitstring += prod_op * Sj
        C_local += bitstring * bitstring

    return C_local


def construct_parallel(n, r, p, n_workers=N_WORKERS):
    """
    Parallel construct: distribute 2^n sign tuples across workers.
    """
    all_signs = list(itertools.product([1, -1], repeat=n))

    chunks = [all_signs[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]

    args = [(chunk, n, r, p) for chunk in chunks]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_construct_worker, args)

    C = op.QubitOperator()
    for c in results:
        C += c

    return C


# ---------------------------------------------------------------------------
# SA sampling — worker function
# ---------------------------------------------------------------------------

def _sa_worker(args):
    """
    Run a subset of SA reads and return (best_energy, best_sample).

    args = (bqm, num_reads, num_sweeps)
    """
    bqm, num_reads, num_sweeps = args
    sampler = SimulatedAnnealingSampler()
    ss = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
    return ss.first.energy, dict(ss.first.sample)


def parallel_sample(bqm, total_reads=100, num_sweeps=10000,
                    n_workers=N_WORKERS):
    """
    Distribute SA reads across workers, return global best.
    """
    reads_per_worker = max(1, total_reads // n_workers)
    # Give remainder to last worker
    read_counts = [reads_per_worker] * n_workers
    read_counts[-1] += total_reads - reads_per_worker * n_workers

    args = [(bqm, rc, num_sweeps) for rc in read_counts]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_sa_worker, args)

    best_energy, best_sample = min(results, key=lambda x: x[0])
    return best_energy, best_sample


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


# ---------------------------------------------------------------------------
# Dinkelbach loop
# ---------------------------------------------------------------------------

def run_sector(Q_diag, C, p,
               lam_init=100.0, total_reads=100, num_sweeps=10000):
    lam       = lam_init
    iteration = 1
    gs_final  = None
    C_dict    = convert_to_dict(C)

    while True:
        print(f"  [p={p}] iteration {iteration}", flush=True)
        iteration += 1

        H_shift   = Q_diag - lam * C
        H_shifted = convert_to_dict(H_shift)
        bqm       = reduction(H_shifted)

        try:
            gse, gs = parallel_sample(bqm,
                                      total_reads=total_reads,
                                      num_sweeps=num_sweeps)
        except ValueError:
            print(f"  [p={p}] ValueError in sampler, retrying", flush=True)
            continue

        print(f"  [p={p}] gse={gse:.6f}  lam={lam:.6f}", flush=True)

        if gse >= 0:
            print(f"  [p={p}] converged: lam={lam:.8f}", flush=True)
            break

        gs_final = gs
        C_dict   = convert_to_dict(C)
        bm_2 = sum(
            coeff * (prod(gs[q] for q in key) if key else 1)
            for key, coeff in C_dict.items()
        )
        lam = lam + gse / bm_2
        print(f"  [p={p}] updated lam={lam:.8f}", flush=True)

    return lam, gs_final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # multiprocessing on macOS requires this guard — do not remove it
    mp.set_start_method('spawn', force=True)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    N_BATH     = 3
    U          = 4.0
    T_HOP      = 1.0
    V_HYB      = 1.0
    R          = 2
    TOTAL_READS  = 100
    NUM_SWEEPS   = 10000

    # ------------------------------------------------------------------
    # Build Hamiltonian
    # ------------------------------------------------------------------
    Q_fermion = siam_hamiltonian(N_BATH, U, T_HOP, V_HYB)
    Q         = bravyi_kitaev(Q_fermion)
    n         = op.count_qubits(Q)
    p_values  = list(range(R // 2 + 1))

    print("=" * 55)
    print(f"SIAM  N_bath={N_BATH}  U={U}  t={T_HOP}  V={V_HYB}")
    print(f"n={n} qubits  |  r={R}  |  sectors={p_values}")
    print(f"Workers={N_WORKERS}")
    print(f"XBK triples: {len(list(Q.terms.items())) * R * R}")
    print(f"construct  : 2^{n}={2**n} sign tuples")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    gseig   = []
    gs_list = []

    for p in p_values:
        print(f"\n--- sector p={p} ---")

        Q_diag = XBK_parallel(Q, n, R, p)
        C      = construct_parallel(n, R, p)

        lam, gs_final = run_sector(
            Q_diag, C, p,
            lam_init=100.0,
            total_reads=TOTAL_READS,
            num_sweeps=NUM_SWEEPS,
        )

        gseig.append(lam)
        gs_list.append(gs_final)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    best_idx = int(np.argmin(gseig))

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    for p_val, lam_val in zip(p_values, gseig):
        marker = "  <-- best" if p_val == p_values[best_idx] else ""
        print(f"  p={p_val}  E0={lam_val:.8f}{marker}")

    print(f"\nGround-state energy : {gseig[best_idx]:.8f}")
    print(f"Best sector p       : {p_values[best_idx]}")

    best_gs = gs_list[best_idx]
    if best_gs is not None:
        print(f"\nGround-state spins (r*n Ising space):")
        print(best_gs)
        print(f"\nOriginal n-qubit spins (replica block 0):")
        print([best_gs.get(i) for i in range(n)])