"""
parallel_siam_v2.py
===================
MPI-parallelised U-sweep version for Kamet HPC.

Parallelised:
  - construct() : 2^n sign tuples striped across ranks
  - XBK()       : (terms, j, k) triples striped across ranks
  - SA reads    : num_reads split across ranks

Key optimisation from serial code:
  - construct() is called ONCE per sector p before the U loop,
    since C does not depend on U. All ranks share the result.
  - The U loop then runs serially, reusing the precomputed C.

Run:
  mpirun -np <N> python3 parallel_siam_v2.py
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Hamiltonian builder
# ---------------------------------------------------------------------------

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
# Parallel SA sampling
# ---------------------------------------------------------------------------

def sample_parallel(bqm, comm, num_reads=500, num_sweeps=50000):
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
# Helpers
# ---------------------------------------------------------------------------

def convert_to_dict(qubop):
    qubdict = defaultdict(float)
    for terms, coeffs in qubop.terms.items():
        key = tuple(t[0] for t in terms)
        qubdict[key] += coeffs.real
    return qubdict


def reduction(qubdict):
    strength = 50 * max(abs(v) for v in qubdict.values())
    bqm = dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)
    return bqm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    os.makedirs("siam_output", exist_ok=True)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    N_BATH    = 2
    t         = 1.0
    V         = 1.0
    r         = 4
    NUM_READS  = 500
    NUM_SWEEPS = 50000
    Us = np.arange(0.0, 10.5, 0.5)

    # ------------------------------------------------------------------
    # Precompute C once per sector — C does not depend on U
    # ------------------------------------------------------------------
    n = op.count_qubits(bravyi_kitaev(siam(N_BATH, 1.0, t, V)))

    if rank == 0:
        print(f"n={n} qubits | r={r} | N_bath={N_BATH} | {comm.Get_size()} ranks",
              flush=True)
        print("Precomputing C for all sectors...", flush=True)

    Cs = []
    for p in range(r // 2 + 1):
        C = construct(n, r, p, comm)
        Cs.append(C)
        if rank == 0:
            print(f"  C[p={p}] done", flush=True)

    # ------------------------------------------------------------------
    # U sweep
    # ------------------------------------------------------------------
    Es = []

    for U in Us:
        if rank == 0:
            print(f"\n=== U={U:.1f} ===", flush=True)

        Q = bravyi_kitaev(siam(N_BATH, U, t, V))
        gseig = []

        for p in range(r // 2 + 1):
            if rank == 0:
                print(f"  r={r}  p={p}", flush=True)

            Q_diag = XBK(Q, n, r, p, comm)
            C = Cs[p]
            C_dict = convert_to_dict(C)

            lam = 100
            iteration = 1

            while True:
                if rank == 0:
                    print(f"    Iteration {iteration}", flush=True)
                iteration += 1

                H_shift   = Q_diag - lam * C
                H_shifted = convert_to_dict(H_shift)
                bqm       = reduction(H_shifted)

                try:
                    gse, gs = sample_parallel(bqm, comm,
                                              num_reads=NUM_READS,
                                              num_sweeps=NUM_SWEEPS)
                except ValueError:
                    continue

                if gse >= 0:
                    if rank == 0:
                        print("    it's over", flush=True)
                    break

                if rank == 0:
                    print(f"    gse={gse:.6f}", flush=True)

                bm_2 = sum(
                    coeff * (prod(gs[q] for q in key) if key else 1)
                    for key, coeff in C_dict.items()
                )

                if abs(bm_2) < 1e-10:
                    if rank == 0:
                        print("    bm_2 ~ 0: resampling.", flush=True)
                    continue

                lam = lam + gse / bm_2

                if rank == 0:
                    print(f"    lam={lam:.8f}", flush=True)

            gseig.append(lam)

        E0 = min(gseig)
        Es.append(E0)

        if rank == 0:
            print(f"  E0(U={U:.1f}) = {E0:.8f}", flush=True)

    # ------------------------------------------------------------------
    # Save results and plot — rank 0 only
    # ------------------------------------------------------------------
    if rank == 0:

        # Analytic E0(U, V=0): impurity decoupled from bath.
        # E0(U, V=0) = -U/4 + 2 * sum_{b: eps_b < 0} eps_b
        def E0_V0(N_bath, U, t):
            b = np.arange(1, N_bath + 1)
            eps = -2.0 * t * np.cos(b * np.pi / (N_bath + 1))
            return -U / 4.0 + 2.0 * np.sum(eps[eps < 0])

        E0_ref   = np.array([E0_V0(N_BATH, U, t) for U in Us])
        Delta_Es = np.array(Es) - E0_ref

        # Save raw E0, reference E0(V=0), and Delta E0
        results = np.column_stack([Us, Es, E0_ref, Delta_Es])
        np.savetxt("siam_output/E0_vs_U.txt", results,
                   header="U  E0  E0_V0  DeltaE0", fmt="%.6f")
        print("\nResults saved to siam_output/E0_vs_U.txt", flush=True)

        # Plot 1: raw E0 vs U/t
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(Us, Es, 'o-', linewidth=1.5, markersize=4)
        ax.set_xlabel("$U/t$", fontsize=13)
        ax.set_ylabel("$E_0$", fontsize=13)
        ax.set_title(f"SIAM ground state energy  |  $N_{{bath}}={N_BATH}$, "
                     f"$V={V}$, $t={t}$", fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("siam_output/E0_vs_U.png", dpi=150)
        plt.close(fig)
        print("Plot saved to siam_output/E0_vs_U.png", flush=True)

        # Plot 2: Delta E0 vs U/Gamma
        # Gamma = pi * rho_0 * V^2, rho_0 = 1/(2t) for 1D chain
        Gamma = np.pi * (1.0 / (2.0 * t)) * V**2
        U_over_Gamma = Us / Gamma

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(U_over_Gamma, Delta_Es, 'o-', linewidth=1.5, markersize=4,
                color='tab:blue')
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_xlabel(r"$U/\Gamma$", fontsize=13)
        ax.set_ylabel(r"$\Delta E_0 = E_0(U,V) - E_0(U,V{=}0)$", fontsize=12)
        ax.set_title(f"SIAM hybridisation energy  |  $N_{{bath}}={N_BATH}$, "
                     f"$V={V}$, $t={t}$, "
                     r"$\Gamma=$" + f"{Gamma:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("siam_output/DeltaE0_vs_UoverGamma.png", dpi=150)
        plt.close(fig)
        print("Plot saved to siam_output/DeltaE0_vs_UoverGamma.png", flush=True)