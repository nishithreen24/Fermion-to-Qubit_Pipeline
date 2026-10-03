"""
tV_validation.py
================
Validates the XBK pipeline against exact diagonalisation for the
1D spinless t-V model at half filling, reproducing Fig. 14 of
Nagpal, Kumar, Hassan (2025) for L = 2.

Produces:
  tV_output/tV_E0_vs_Vt_L2.png   -- E0 vs V/t: pipeline + ED
  tV_output/tV_data_L2.txt        -- raw numbers

Run:
    python tV_validation.py
"""

import openfermion as op
import itertools
from collections import defaultdict
from math import prod
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from openfermion import bravyi_kitaev, get_sparse_operator
import scipy.sparse.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from Models import tV_hamiltonian_1d
from openfermion import FermionOperator as FO


def tV_hamiltonian_ph(L, t=1.0, V=1.0, periodic=False):
    """
    1D spinless t-V Hamiltonian in particle-hole symmetric form.

    H = -t * sum_{<i,j>} (c†_i c_j + h.c.)
      + V * sum_{<i,j>} (n_i - 1/2)(n_j - 1/2)

    Expanding: V*(n_i*n_j - n_i/2 - n_j/2 + 1/4)

    This form produces V-dependence even at L=2 half filling,
    matching the convention used in Nagpal et al. (2025).
    """
    H = FO()
    bonds = list(range(L - 1))
    if periodic and L > 2:
        bonds_wrap = [(L-1, 0)]
    else:
        bonds_wrap = []

    all_bonds = [(i, i+1) for i in bonds] + bonds_wrap

    for i, j in all_bonds:
        # hopping
        H += FO(f"{i}^ {j}", -t)
        H += FO(f"{j}^ {i}", -t)
        # interaction: V*(ni - 1/2)*(nj - 1/2)
        #            = V*ni*nj - V/2*ni - V/2*nj + V/4
        H += FO(f"{i}^ {i} {j}^ {j}",  V)
        H += FO(f"{i}^ {i}",           -V / 2.0)
        H += FO(f"{j}^ {j}",           -V / 2.0)
        H += FO("",                      V / 4.0)

    return H


os.makedirs("tV_output", exist_ok=True)


# ---------------------------------------------------------------------------
# XBK, construct, helpers  (identical to pipeline)
# ---------------------------------------------------------------------------

def XBK(H_qubit, n, r, p):
    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]
    H_ising = op.QubitOperator()
    for terms, coeffs in H_qubit.terms.items():
        for j in range(r):
            Sj = -1 if j < p else 1
            for k in range(r):
                Sk = -1 if k < p else 1
                qc = -1
                Qbar = op.QubitOperator(())
                for i in range(n - 1, -1, -1):
                    Zj, Zk = Z[j*n+i], Z[k*n+i]
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
                H_ising += coeffs * Sj * Sk * Qbar
    return H_ising


def construct(n, r, p):
    Z = [op.QubitOperator(f'Z{i}') for i in range(r * n)]
    C = op.QubitOperator()
    for signs in itertools.product([1, -1], repeat=n):
        bitstring = op.QubitOperator()
        for j in range(r):
            Sj = -1 if j < p else 1
            prod_op = op.QubitOperator(())
            for i in range(n):
                prod_op *= (Z[j*n+i] * signs[i] + 1) * 0.5
            bitstring += prod_op * Sj
        C += bitstring * bitstring
    return C


def convert_to_dict(qubop):
    d = defaultdict(float)
    for terms, coeffs in qubop.terms.items():
        d[tuple(t[0] for t in terms)] += coeffs.real
    return d


def reduction(qubdict):
    strength = 100 * max(abs(v) for v in qubdict.values())
    return dimod.higherorder.utils.make_quadratic(qubdict, strength, dimod.SPIN)


# ---------------------------------------------------------------------------
# Dinkelbach loop for one (Q_diag, C) pair
# ---------------------------------------------------------------------------

def dinkelbach(Q_diag, C, num_reads=500, num_sweeps=50000,
               lam_init=-1.0, tol=1e-8):
    C_dict = convert_to_dict(C)
    lam = lam_init
    sampler = SimulatedAnnealingSampler()

    for _ in range(200):   # max iterations
        H_shift = Q_diag - lam * C
        bqm = reduction(convert_to_dict(H_shift))
        try:
            ss = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
        except ValueError:
            continue

        gse = ss.first.energy
        gs  = dict(ss.first.sample)

        if gse >= 0 or abs(gse) < tol:
            break

        bm_2 = sum(
            coeff * (prod(gs[q] for q in key) if key else 1)
            for key, coeff in C_dict.items()
        )
        if abs(bm_2) < 1e-10:
            continue

        lam = lam + gse / bm_2

    return lam


# ---------------------------------------------------------------------------
# ED ground state energy
# ---------------------------------------------------------------------------

def ed_energy(H_fermion):
    H_sparse = get_sparse_operator(H_fermion)
    vals, _ = scipy.sparse.linalg.eigsh(H_sparse, k=1, which='SA')
    return float(vals[0])


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    L   = 2
    t   = 1.0
    r   = 3          # replication factor
    Vs  = np.arange(0.0, 3.6, 0.25)   # V/t values, matching Fig. 14

    # periodic boundary conditions, half filling (L/2 particles)
    # For L=2 with periodic BC, the ring has 2 sites.
    periodic = True

    E0_pipeline = []
    E0_ed       = []

    # Precompute Q at V=0 just to get n (qubit count is V-independent)
    Q_test = bravyi_kitaev(tV_hamiltonian_ph(L=L, t=t, V=0.0, periodic=periodic))
    n = op.count_qubits(Q_test)
    print(f"L={L}  n={n} qubits  r={r}", flush=True)

    # Precompute C for all sectors — C does not depend on V
    p_values = list(range(r // 2 + 1))
    Cs = []
    for p in p_values:
        print(f"  Building C[p={p}]...", flush=True)
        Cs.append(construct(n, r, p))

    for V in Vs:
        print(f"\nV/t = {V/t:.2f}", flush=True)

        H_ferm = tV_hamiltonian_ph(L=L, t=t, V=V, periodic=periodic)
        Q      = bravyi_kitaev(H_ferm)

        # --- ED ---
        e_ed = ed_energy(H_ferm)
        E0_ed.append(e_ed)
        print(f"  ED:       E0 = {e_ed:.6f}", flush=True)

        # --- Pipeline ---
        gseig = []
        for p, C in zip(p_values, Cs):
            Q_diag = XBK(Q, n, r, p)
            lam    = dinkelbach(Q_diag, C)
            gseig.append(lam)
            print(f"  p={p}  lam = {lam:.6f}", flush=True)

        e_pipe = min(gseig)
        E0_pipeline.append(e_pipe)
        print(f"  Pipeline: E0 = {e_pipe:.6f}  error = {abs(e_pipe-e_ed):.6f}",
              flush=True)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    Vs_over_t = Vs / t
    data = np.column_stack([Vs_over_t, E0_ed, E0_pipeline,
                            np.abs(np.array(E0_pipeline) - np.array(E0_ed))])
    np.savetxt("tV_output/tV_data_L2.txt", data,
               header="V/t  E0_ED  E0_pipeline  abs_error", fmt="%.8f")
    print("\nData saved to tV_output/tV_data_L2.txt")

    # -----------------------------------------------------------------------
    # Plot: E0 vs V/t
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Vs_over_t, E0_ed,       'k-',  linewidth=2,   label="ED (exact)")
    ax.plot(Vs_over_t, E0_pipeline, 'o--', linewidth=1.5,
            markersize=5, color='tab:blue', label=f"XBK+SA ($r={r}$)")
    ax.set_xlabel("$V/t$", fontsize=13)
    ax.set_ylabel("$E_0$", fontsize=13)
    ax.set_title(f"1D $t$–$V$ model, $L={L}$ (periodic ring)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("tV_output/tV_E0_vs_Vt_L2.png", dpi=150)
    plt.close(fig)
    print("Plot saved to tV_output/tV_E0_vs_Vt_L2.png")

    # -----------------------------------------------------------------------
    # Plot: absolute error
    # -----------------------------------------------------------------------
    errors = np.abs(np.array(E0_pipeline) - np.array(E0_ed))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(Vs_over_t, errors + 1e-12, 'o-', color='tab:red',
                linewidth=1.5, markersize=5)
    ax.set_xlabel("$V/t$", fontsize=13)
    ax.set_ylabel(r"$|E_\mathrm{pipeline} - E_\mathrm{ED}|$", fontsize=12)
    ax.set_title(f"Absolute error, $L={L}$, $r={r}$", fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig("tV_output/tV_error_L2.png", dpi=150)
    plt.close(fig)
    print("Plot saved to tV_output/tV_error_L2.png")