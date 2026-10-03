"""
parallel_siam_v1.py
===================
MPI-parallel XBK pipeline for the 2-site (L=2) periodic spinless t-V model.

Sweeps V/t from 0.0 to 4.0 in steps of 0.2 (21 points).
No output is printed to stdout.
Results are written to  siam_output/results.csv
Plot is saved to        siam_output/ground_state_energy.png

Resource layout (siam_v4.slurm):
    30 MPI ranks  x  2 CPUs/rank

Parallelism:
  1. build_counting_operator  — MPI scatter/gather over 2^n_qubits basis states
  2. V/t sweep                — 21 V values distributed round-robin over ranks;
                                each rank handles all sectors for its subset of V
  3. simulated_annealing      — multiprocessing.Pool over SA chains within each rank,
                                one chain per SLURM_CPUS_PER_TASK (default 2)
"""

import os
import csv
import math
import random
import warnings
from collections import defaultdict
from itertools import product as iproduct
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpi4py import MPI

from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator
from openfermion.utils import count_qubits

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# MPI globals
# ─────────────────────────────────────────────────────────────────────────────

comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nranks = comm.Get_size()

# SA chains per rank = CPUs allocated per task by SLURM
SA_CHAINS = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))

# ─────────────────────────────────────────────────────────────────────────────
# Model: 2-site periodic t-V Hamiltonian
# ─────────────────────────────────────────────────────────────────────────────

def build_tV_hamiltonian(t: float, V: float, L: int) -> FermionOperator:
    """
    Spinless t-V Hamiltonian on a 1D ring of L sites with periodic boundaries.

    H = -t sum_{j=0}^{L-1} (c_j^dag c_{j+1} + h.c.)
        + V sum_{j=0}^{L-1} (n_j - 1/2)(n_{j+1} - 1/2)

    Expanded interaction:
        (n_j - 1/2)(n_{j+1} - 1/2) = n_j n_{j+1} - n_j/2 - n_{j+1}/2 + 1/4

    Indices are taken mod L (periodic boundary conditions).
    """
    H = FermionOperator()
    for j in range(L):
        jp = (j + 1) % L
        H += FermionOperator(((j, 1), (jp, 0)), -t)
        H += FermionOperator(((jp, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (j, 0), (jp, 1), (jp, 0)), V)
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator(((jp, 1), (jp, 0)), -V / 2)
        H += FermionOperator((), V / 4)
    return H


def exact_ground_state(H_fermion: FermionOperator) -> float:
    """Lowest eigenvalue of H_fermion by dense diagonalisation."""
    H_mat = get_sparse_operator(H_fermion).toarray()
    return float(np.linalg.eigvalsh(H_mat)[0])


# ─────────────────────────────────────────────────────────────────────────────
# BK transformation helpers
# ─────────────────────────────────────────────────────────────────────────────

def bk_transform(H_fermion):
    """Return (bk_terms, n_qubits).
    bk_terms: list of (float_coeff, {qubit: pauli_char}).
    """
    H_bk = bravyi_kitaev(H_fermion)
    n = count_qubits(H_bk)
    terms = []
    for pauli_tuple, coeff in H_bk.terms.items():
        c = float(np.real(coeff))
        if abs(c) > 1e-14:
            terms.append((c, {idx: op for idx, op in pauli_tuple}))
    return terms, n


# ─────────────────────────────────────────────────────────────────────────────
# Ising polynomial arithmetic
# dict { frozenset(qubit_indices) -> float }
# Evaluation at spin config s: sum_S coeff_S * prod_{i in S} s_i
# Z_a Z_a = I  ->  product of monomials = symmetric difference of their sets
# ─────────────────────────────────────────────────────────────────────────────

def _poly_mul(p1: dict, p2: dict) -> dict:
    out = defaultdict(complex)
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            out[s1.symmetric_difference(s2)] += c1 * c2
    return dict(out)


def _poly_clean(poly: dict, tol: float = 1e-12) -> dict:
    return {m: v.real for m, v in poly.items() if abs(v.real) > tol}


# ─────────────────────────────────────────────────────────────────────────────
# XBK: H'_p = sum_{j,k=1}^{r=2} S'(j) S'(k) H_(j,k)
#
# Physical qubit layout (r=2):
#   logical qubit i, copy c (1-based) -> physical qubit  2*i + (c-1)
# ─────────────────────────────────────────────────────────────────────────────

def _phys(logical: int, copy: int) -> int:
    return 2 * logical + (copy - 1)


def _xbk_single(pauli: str, qi: int, j: int, k: int, sp: int) -> dict:
    """
    Ising polynomial for one Pauli on logical qubit qi, copies j and k.

    Same copy (j==k):  X,Y->0;  Z->sp Z_{qi,j};  I->1
    Cross copy (j!=k):
        X ->  sp (1 - Z_{qi,j} Z_{qi,k}) / 2
        Y ->  i sp (Z_{qi,k} - Z_{qi,j}) / 2
        Z ->  sp (Z_{qi,j} + Z_{qi,k}) / 2
        I ->  (1 + Z_{qi,j} Z_{qi,k}) / 2
    """
    pj, pk = _phys(qi, j), _phys(qi, k)
    if j == k:
        if pauli in ('X', 'Y'):  return {}
        if pauli == 'Z':         return {frozenset([pj]): sp + 0j}
        return {frozenset(): 1.0 + 0j}
    if pauli == 'X':
        return {frozenset(): sp*0.5+0j, frozenset([pj, pk]): -sp*0.5+0j}
    if pauli == 'Y':
        return {frozenset([pk]): 1j*sp*0.5, frozenset([pj]): -1j*sp*0.5}
    if pauli == 'Z':
        return {frozenset([pj]): sp*0.5+0j, frozenset([pk]): sp*0.5+0j}
    return {frozenset(): 0.5+0j, frozenset([pj, pk]): 0.5+0j}  # I


def build_Hprime(bk_terms, n_qubits: int, sector_signs: list) -> dict:
    """
    Build H'_p = sum_{j,k} S'(j) S'(k) H_(j,k)   (r=2, S'(j)=+1 always).

    For each Pauli string: multiply single-qubit XBK polynomials,
    then accumulate into the full H' polynomial.
    """
    H_prime = defaultdict(complex)
    for j in range(1, 3):
        for k in range(1, 3):
            for coeff, pauli_dict in bk_terms:
                term = {frozenset(): 1.0 + 0j}
                for qi in range(n_qubits):
                    sp = sector_signs[qi]
                    single = _xbk_single(pauli_dict.get(qi, 'I'), qi, j, k, sp)
                    term = _poly_mul(term, single)
                for mono, val in term.items():
                    H_prime[mono] += coeff * val
    return _poly_clean(dict(H_prime))


# ─────────────────────────────────────────────────────────────────────────────
# Counting operator — parallel build via MPI scatter/gather
#
# C(s) = sum_b (P_{1,b}(s) + P_{2,b}(s))^2
# P_{j,b}(s) = prod_k (1 + (-1)^{b_k} s_{k,j}) / 2
# ─────────────────────────────────────────────────────────────────────────────

def _counting_partial(basis_list: list, n_qubits: int) -> dict:
    """Accumulate C contributions for a subset of basis states."""
    partial = defaultdict(complex)
    for basis in basis_list:
        inner = defaultdict(complex)
        for copy_j in range(1, 3):
            p = {frozenset(): 1.0 + 0j}
            for k, b_k in enumerate(basis):
                phys = _phys(k, copy_j)
                sign_k = float((-1) ** b_k)
                factor = {frozenset(): 0.5+0j, frozenset([phys]): sign_k*0.5+0j}
                p = _poly_mul(p, factor)
            for m, v in p.items():
                inner[m] += v
        sq = _poly_mul(dict(inner), dict(inner))
        for m, v in sq.items():
            partial[m] += v
    return dict(partial)


def build_counting_operator_parallel(n_qubits: int) -> dict:
    """
    Build C_p using all MPI ranks.

    Rank 0 distributes the 2^n_qubits basis states across ranks via scatter.
    Each rank computes its partial Ising polynomial.
    Results are gathered on rank 0, merged, and broadcast to all ranks.
    """
    all_bases = list(iproduct([0, 1], repeat=n_qubits))

    if rank == 0:
        chunks = [[] for _ in range(nranks)]
        for i, b in enumerate(all_bases):
            chunks[i % nranks].append(b)
    else:
        chunks = None

    my_bases = comm.scatter(chunks, root=0)
    local_C  = _counting_partial(my_bases, n_qubits)

    all_parts = comm.gather(local_C, root=0)
    if rank == 0:
        merged = defaultdict(complex)
        for part in all_parts:
            for m, v in part.items():
                merged[m] += v
        C = _poly_clean(dict(merged))
    else:
        C = None
    return comm.bcast(C, root=0)


# ─────────────────────────────────────────────────────────────────────────────
# Quadratisation: k-local -> 2-local
#
# x_i = (1 - s_i)/2;  s_i = 1 - 2 x_i
# Rosenberg substitution: introduce w = x_a x_b with penalty
#   M (x_a x_b - 2 w x_a - 2 w x_b + 3 w) >= 0
# ─────────────────────────────────────────────────────────────────────────────

def _ising_to_binary(ising: dict) -> dict:
    """prod_{i in S} s_i = sum_{T subset S} (-2)^|T| prod_{i in T} x_i"""
    out = defaultdict(float)
    for S, c in ising.items():
        sl = sorted(S)
        k = len(sl)
        for tb in range(1 << k):
            T = frozenset(sl[m] for m in range(k) if (tb >> m) & 1)
            out[T] += c * ((-2) ** len(T))
    return dict(out)


def _binary_to_ising(binary: dict) -> dict:
    """prod_{i in T} x_i = (1/2^k) sum_{U subset T} (-1)^|U| prod_{i in U} s_i"""
    out = defaultdict(float)
    for T, c in binary.items():
        tl = sorted(T)
        k = len(tl)
        for ub in range(1 << k):
            U = frozenset(tl[m] for m in range(k) if (ub >> m) & 1)
            out[U] += c * ((-1) ** len(U)) / (2 ** k)
    return dict(out)


def quadratize(binary: dict, M: float, next_id: int):
    """Reduce binary polynomial to degree <= 2 via Rosenberg substitution."""
    poly = defaultdict(float, binary)
    changed = True
    while changed:
        changed = False
        for mono in list(poly.keys()):
            if len(mono) < 3:
                continue
            c = poly.pop(mono, 0.0)
            if abs(c) < 1e-14:
                continue
            sl = sorted(mono)
            a, b, rest = sl[0], sl[1], frozenset(sl[2:])
            w = next_id;  next_id += 1
            poly[frozenset([w]) | rest] += c
            poly[frozenset([a, b])]     += M
            poly[frozenset([w, a])]     += -2.0 * M
            poly[frozenset([w, b])]     += -2.0 * M
            poly[frozenset([w])]        += 3.0 * M
            changed = True
            break
    return dict(poly), next_id


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing — parallel chains via multiprocessing.Pool
# ─────────────────────────────────────────────────────────────────────────────

def _eval(ising: dict, spins: dict) -> float:
    E = 0.0
    for mono, c in ising.items():
        p = 1.0
        for i in mono:
            p *= spins[i]
        E += c * p
    return E


def _dE(ising: dict, spins: dict, flip: int) -> float:
    si = spins[flip]
    field = 0.0
    for mono, c in ising.items():
        if flip not in mono:
            continue
        p = 1.0
        for j in mono:
            if j != flip:
                p *= spins[j]
        field += c * p
    return -2.0 * si * field


def _sa_chain(args):
    """Single SA chain (called from Pool). Returns (spins, energy)."""
    ising, n_spins, n_sweeps, T_start, T_end, seed = args
    rng = random.Random(seed)
    spins = {i: rng.choice([-1, 1]) for i in range(n_spins)}
    E = _eval(ising, spins)
    best_s, best_E = dict(spins), E
    alpha = (T_end / T_start) ** (1.0 / max(n_sweeps, 1))
    T = T_start
    for _ in range(n_sweeps):
        for __ in range(n_spins):
            i = rng.randint(0, n_spins - 1)
            d = _dE(ising, spins, i)
            if d < 0 or rng.random() < math.exp(-d / T):
                spins[i] = -spins[i]
                E += d
                if E < best_E:
                    best_E = E;  best_s = dict(spins)
        T *= alpha
    return best_s, best_E


def sa_parallel(ising: dict, n_spins: int, n_sweeps: int,
                T_start: float, T_end: float, base_seed: int) -> tuple:
    """
    Run SA_CHAINS independent SA chains in parallel using multiprocessing.Pool.
    Each chain gets a distinct seed: base_seed + chain_index.
    Returns (best_spins, best_energy) across all chains.
    """
    args = [(ising, n_spins, n_sweeps, T_start, T_end, base_seed + c)
            for c in range(SA_CHAINS)]
    if SA_CHAINS == 1:
        results = [_sa_chain(args[0])]
    else:
        with Pool(processes=SA_CHAINS) as pool:
            results = pool.map(_sa_chain, args)
    return min(results, key=lambda r: r[1])


# ─────────────────────────────────────────────────────────────────────────────
# Dinkelbach iteration (uses parallel SA)
# ─────────────────────────────────────────────────────────────────────────────

def dinkelbach(H_ising: dict, C_ising: dict, n_spins: int,
               n_sweeps: int, M: float, base_seed: int,
               max_iter: int = 25, tol: float = 1e-6) -> float:
    """
    Solve E0 = min_s H'(s)/C(s) via Dinkelbach fixed-point + parallel SA.
    Returns the best E0 found, or +inf if C is always zero.
    """
    lam = 5.0
    best_E0 = float('inf')
    T0 = max(3.0 * abs(M), 5.0)

    for it in range(max_iter):
        combined = defaultdict(float, H_ising)
        for m, c in C_ising.items():
            combined[m] -= lam * c

        spins, _ = sa_parallel(dict(combined), n_spins,
                                n_sweeps, T0, 1e-4,
                                base_seed=base_seed + it * 1000)

        H_val = _eval(H_ising, spins)
        C_val = _eval(C_ising, spins)

        if abs(C_val) < 1e-8:
            continue

        new_lam = H_val / C_val
        if new_lam < best_E0:
            best_E0 = new_lam

        if abs(new_lam - lam) < tol:
            break
        lam = new_lam

    return best_E0


# ─────────────────────────────────────────────────────────────────────────────
# Per-V-value pipeline: BK -> XBK -> quadratise -> Dinkelbach
# ─────────────────────────────────────────────────────────────────────────────

def compute_one_V(t: float, V: float, C_ising: dict,
                  n_qubits: int, penalty_scale: float,
                  n_sweeps: int, base_seed: int) -> tuple:
    """
    Run the full pipeline for a single (t, V) value.
    Returns (E_xbk, E_exact).
    C_ising is pre-built and shared across all V values.
    """
    H_f       = build_tV_hamiltonian(t, V, L = 6)
    bk_terms, _ = bk_transform(H_f)
    n_phys    = n_qubits * 2
    M         = penalty_scale * max(abs(t), abs(V), 1.0)

    best_E0 = float('inf')

    for sec in range(2 ** n_qubits):
        sector_signs = [+1 if (sec >> i) & 1 == 0 else -1
                        for i in range(n_qubits)]

        H_ising = build_Hprime(bk_terms, n_qubits, sector_signs)

        next_id = n_phys
        H_quad, next_id = quadratize(_ising_to_binary(H_ising), M, next_id)
        C_quad, next_id = quadratize(_ising_to_binary(C_ising),  M, next_id)
        n_total = next_id

        H_iq = _binary_to_ising(H_quad)
        C_iq = _binary_to_ising(C_quad)

        E0 = dinkelbach(H_iq, C_iq, n_total, n_sweeps, M,
                        base_seed=base_seed + sec * 37)
        if E0 < best_E0:
            best_E0 = E0

    E_exact = exact_ground_state(H_f)
    return best_E0, E_exact


# ─────────────────────────────────────────────────────────────────────────────
# Main: MPI-parallel sweep over V/t
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t             = 1.0
    V_values      = np.round(np.arange(0.0, 4.01, 0.2), 6)   # 21 points
    n_sweeps      = 2000
    penalty_scale = 12.0
    out_dir       = "siam_output"

    # n_qubits is fixed for L=2 (independent of V)
    _, n_qubits = bk_transform(build_tV_hamiltonian(t, 1.0, L = 6))

    # ── Step 1: Build counting operator once (MPI parallel) ──────────────
    C_ising = build_counting_operator_parallel(n_qubits)

    # ── Step 2: Distribute V values round-robin across ranks ─────────────
    my_V = [V for i, V in enumerate(V_values) if i % nranks == rank]

    local_results = {}
    for V in my_V:
        seed = int(round(V * 100)) + rank * 9973
        E_xbk, E_exact = compute_one_V(
            t, V, C_ising, n_qubits, penalty_scale, n_sweeps, seed
        )
        local_results[float(V)] = (E_xbk, E_exact)

    # ── Step 3: Gather all results on rank 0 ─────────────────────────────
    all_dicts = comm.gather(local_results, root=0)

    if rank != 0:
        return

    # Merge and sort by V
    merged = {}
    for d in all_dicts:
        merged.update(d)
    rows = sorted(merged.items(), key=lambda x: x[0])

    V_arr  = np.array([r[0]    for r in rows])
    E_xbk  = np.array([r[1][0] for r in rows])
    E_ed   = np.array([r[1][1] for r in rows])
    errors = np.abs(E_xbk - E_ed)

    # ── Save CSV ──────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["V_over_t", "E_exact", "E_xbk_sa", "abs_error"])
        for V, (E_x, E_e) in rows:
            writer.writerow([f"{V/t:.2f}", f"{E_e:.10f}",
                             f"{E_x:.10f}", f"{abs(E_x-E_e):.4e}"])

    # ── Plot ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family"        : "DejaVu Serif",
        "mathtext.fontset"   : "dejavuserif",
        "font.size"          : 12,
        "axes.linewidth"     : 1.2,
        "xtick.direction"    : "in",  "ytick.direction"    : "in",
        "xtick.top"          : True,  "ytick.right"        : True,
        "xtick.major.size"   : 5,     "ytick.major.size"   : 5,
        "xtick.minor.size"   : 3,     "ytick.minor.size"   : 3,
        "xtick.minor.visible": True,  "ytick.minor.visible": True,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05},
        sharex=True,
    )

    ax1.plot(V_arr, E_ed,
             color="#1a1a2e", linewidth=2.2, linestyle="-", zorder=3,
             label="Exact diagonalisation (ED)")
    ax1.scatter(V_arr, E_xbk,
                color="#e63946", s=60, marker="o", linewidths=0.9,
                edgecolors="white", zorder=4,
                label=r"XBK + SA  ($r\!=\!2$, QUBO)")

    ax1.axvline(2.0, color="#888888", linestyle="--", linewidth=1.1, alpha=0.75)
    ax1.text(2.07, -2.05, r"$V/t = 2$  (critical)",
             color="#666666", fontsize=9.5, va="top", style="italic")

    ax1.text(0.45, -3.75, "Metallic\n" + r"($V \ll t$)",
             ha="center", va="center", fontsize=10, color="#3d6b9e",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf0f9",
                       edgecolor="#3d6b9e", alpha=0.7, linewidth=0.8))
    ax1.text(3.35, -2.25, "CDW\n" + r"($V \gg t$)",
             ha="center", va="center", fontsize=10, color="#b5451b",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdf0ec",
                       edgecolor="#b5451b", alpha=0.7, linewidth=0.8))

    ax1.set_ylabel(r"Ground-state energy $E_0$  (units of $t$)", fontsize=13)
    ax1.set_ylim(-4.25, -1.75)
    ax1.legend(frameon=True, framealpha=0.95, fontsize=11,
               loc="upper left", handlelength=2.2)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.set_title(
        r"2-site periodic spinless $t$-$V$ model  ---  "
        r"$L = 2$,  $r = 2$  (MPI parallel)",
        fontsize=12.5, pad=10,
    )

    safe_err = np.where(errors < 1e-15, 1e-16, errors)
    ax2.semilogy(V_arr, safe_err,
                 color="#457b9d", linewidth=1.6, linestyle="-",
                 marker="s", markersize=5, markerfacecolor="white",
                 markeredgewidth=1.3,
                 label=r"$|E_\mathrm{XBK} - E_\mathrm{ED}|$")
    ax2.axvline(2.0, color="#888888", linestyle="--", linewidth=1.1, alpha=0.75)
    ax2.axhline(1e-13, color="#cccccc", linestyle=":", linewidth=0.9)
    ax2.set_ylabel(r"$|E_\mathrm{XBK} - E_\mathrm{ED}|$", fontsize=12)
    ax2.set_xlabel(r"$V/t$  (interaction / hopping)", fontsize=13)
    ax2.set_ylim(1e-17, 1e-10)
    ax2.legend(frameon=True, framealpha=0.95, fontsize=10, loc="upper left")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    for ax in (ax1, ax2):
        ax.set_xlim(-0.1, 4.2)

    png_path = os.path.join(out_dir, "ground_state_energy.png")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
