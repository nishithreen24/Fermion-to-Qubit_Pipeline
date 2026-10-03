"""
model_tv1d.py
=============
1D spinless t-V model, verification run.

Hamiltonian (particle-hole symmetric form):
    H = -t  sum_{i} (c†_i c_{i+1} + h.c.)
        + V  sum_{i} (n_i - 1/2)(n_{i+1} - 1/2)

Sweep: V/t in {0.0, 1.0, 2.0, 3.0, 4.0}
L values: 2, 4   (small, so ED ground state is computable and exact)

For each (L, V/t) point this script computes:
  1. E0_xbk  — pipeline result (BK -> XBK -> quadratise -> Dinkelbach + SA)
  2. E0_ed   — exact ground state via scipy sparse eigensolver
  3. error   = |E0_xbk - E0_ed|

All results are written to output/tv1d/tv1d_L<L>.csv on rank 0.
A PNG comparing E0_xbk vs E0_ed is also saved.

Usage:
    mpirun -np 16 python model_tv1d.py
"""

import os
import csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpi4py import MPI
from openfermion import get_sparse_operator

from pipeline_base import (
    bk_transform,
    build_counting_operator_parallel,
    run_one,
    comm, rank, nranks,
)
from Models import tV_hamiltonian_1d

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR      = "/lustre/shassan/models/output/tv1d"
MODEL_NAME   = "tv1d"

T            = 1.0                         # hopping amplitude (sets energy scale)
L_VALUES     = [2, 4]                      # lattice sizes
VT_VALUES    = [0.0, 1.0, 2.0, 3.0, 4.0]  # V/t ratios to sweep

N_SWEEPS     = 5000    # SA sweeps per Dinkelbach iteration
PENALTY_SCALE = 12.0  # M = PENALTY_SCALE * max|coefficient|


# ─────────────────────────────────────────────────────────────────────────────
# Exact diagonalisation (ED) ground state
# ─────────────────────────────────────────────────────────────────────────────

def ed_ground_state(L, V, t=1.0, periodic=False):
    """
    Compute the exact ground state energy of the 1D t-V model via
    sparse diagonalisation.

    Returns E0 as a float.
    """
    H_fermion = tV_hamiltonian_1d(L=L, t=t, V=V, mu=0.0, periodic=periodic)
    H_sparse  = get_sparse_operator(H_fermion)
    # k=1 requests only the lowest eigenvalue; which='SA' = smallest algebraic
    vals = spla.eigsh(H_sparse.real, k=1, which='SA', return_eigenvectors=False)
    return float(vals[0])


# ─────────────────────────────────────────────────────────────────────────────
# MPI job distribution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # All (L, V/t) pairs
    all_jobs = [(L, vt) for L in L_VALUES for vt in VT_VALUES]

    # Each rank takes a round-robin slice
    my_jobs = [job for i, job in enumerate(all_jobs) if i % nranks == rank]

    local_results = []   # list of (L, vt, E0_xbk, E0_ed)

    for L, vt in my_jobs:
        V = vt * T

        # ── ED ground state ──────────────────────────────────────────────────
        E0_ed = ed_ground_state(L=L, V=V, t=T, periodic=False)

        # ── Pipeline: BK -> XBK -> quadratise -> Dinkelbach ─────────────────
        H_fermion       = tV_hamiltonian_1d(L=L, t=T, V=V, mu=0.0, periodic=False)
        bk_terms, n_q   = bk_transform(H_fermion)
        C_ising         = build_counting_operator_parallel(n_q)
        seed            = int(abs(vt * 100)) + L * 997 + rank * 9973
        E0_xbk          = run_one(bk_terms, n_q, C_ising,
                                  PENALTY_SCALE, N_SWEEPS, seed)

        error = abs(E0_xbk - E0_ed)
        local_results.append((L, vt, E0_xbk, E0_ed, error))

        # Live progress visible in the SLURM log
        print(f"[rank {rank:3d}] L={L}, V/t={vt:.1f}: "
              f"E0_xbk={E0_xbk:.8f}  E0_ed={E0_ed:.8f}  "
              f"error={error:.2e}",
              flush=True)

    # ── Gather all results on rank 0 ─────────────────────────────────────────
    all_parts = comm.gather(local_results, root=0)

    if rank != 0:
        return

    # Flatten and sort
    flat = []
    for part in all_parts:
        flat.extend(part)
    flat.sort(key=lambda r: (r[0], r[1]))   # sort by (L, V/t)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    # One CSV per L value
    from collections import defaultdict
    by_L = defaultdict(list)
    for row in flat:
        by_L[row[0]].append(row)

    for L, rows in by_L.items():
        path = os.path.join(OUT_DIR, f"{MODEL_NAME}_L{L}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["V/t", "E0_xbk", "E0_ed", "error"])
            for _, vt, E0_xbk, E0_ed, error in rows:
                w.writerow([f"{vt:.4f}", f"{E0_xbk:.10f}",
                            f"{E0_ed:.10f}", f"{error:.4e}"])
        print(f"Saved: {path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    colours = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (L, rows) in zip(axes, sorted(by_L.items())):
        vts     = [r[1] for r in rows]
        xbk_es  = [r[2] for r in rows]
        ed_es   = [r[3] for r in rows]
        errors  = [r[4] for r in rows]

        ax.plot(vts, ed_es,  "k-",  linewidth=2.0, label="ED (exact)")
        ax.plot(vts, xbk_es, "o--", color=colours[0],
                linewidth=1.5, markersize=6, label="XBK + SA")
        ax.set_xlabel(r"$V/t$", fontsize=13)
        ax.set_ylabel(r"$E_0$", fontsize=13)
        ax.set_title(f"1D t-V,  $L={L}$", fontsize=13)
        ax.legend(fontsize=11)

        # Inset: log error
        ax2 = ax.inset_axes([0.55, 0.55, 0.42, 0.38])
        ax2.semilogy(vts, [max(e, 1e-16) for e in errors],
                     "s-", color=colours[1], markersize=4)
        ax2.set_xlabel(r"$V/t$", fontsize=9)
        ax2.set_ylabel(r"$|E_{\rm XBK} - E_{\rm ED}|$", fontsize=9)
        ax2.tick_params(labelsize=8)

    fig.suptitle("1D t-V model: XBK pipeline vs exact diagonalisation",
                 fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{MODEL_NAME}_verification.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png}")

    # ── Summary to stdout ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"{'L':>4}  {'V/t':>6}  {'E0_xbk':>14}  {'E0_ed':>14}  {'error':>10}")
    print("-" * 60)
    for _, vt, E0_xbk, E0_ed, error in flat:
        print(f"{_:>4}  {vt:>6.1f}  {E0_xbk:>14.8f}  {E0_ed:>14.8f}  {error:>10.2e}")
    print("=" * 60)
    max_err = max(r[4] for r in flat)
    print(f"Maximum error across all points: {max_err:.2e}")
    if max_err < 1e-4:
        print("PASS: all errors below 1e-4.")
    else:
        print("FAIL: at least one error exceeds 1e-4 — check pipeline.")


if __name__ == "__main__":
    main()