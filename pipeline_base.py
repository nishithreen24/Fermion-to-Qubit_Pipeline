"""
pipeline_base.py
================
Shared XBK + Quadratisation + SA + Dinkelbach pipeline.
Imported by all model scripts.  No model-specific code here.

Sweep structure
---------------
For each L in L_values, sweep ratio over ratio_values.
All (L, ratio) pairs distributed round-robin across MPI ranks.
Results gathered on rank 0 as:
    dict { L: [ {'x': ratio, 'E0': float}, ... ] }

Public API
----------
bk_transform(H_fermion)                         -> (bk_terms, n_qubits)
parse_qubit_op(H_qubit)                         -> bk_terms
build_Hprime(bk_terms, n_qubits, sector_signs)  -> ising_poly
build_counting_operator_parallel(n_qubits)       -> ising_poly
quadratize(binary_poly, M, next_id)             -> (binary_poly, next_id)
sa_parallel(ising, n_spins, ...)                -> (spins, energy)
dinkelbach(H_ising, C_ising, n_spins, ...)      -> float
run_one(bk_terms, n_qubits, C_ising, ...)       -> float
run_sweep(hamiltonian_fn, ratio_values, L_values, ...)   -> dict|None
run_sweep_qubit(qubit_fn, ratio_values, L_values, ...)   -> dict|None
save_results(results, out_dir, model_name)
save_plot(results, model_name, ratio_label, out_dir)
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
from openfermion import bravyi_kitaev
from openfermion.utils import count_qubits

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# MPI globals
# ─────────────────────────────────────────────────────────────────────────────

comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nranks = comm.Get_size()

SA_CHAINS = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))

# ─────────────────────────────────────────────────────────────────────────────
# BK transformation
# ─────────────────────────────────────────────────────────────────────────────

def bk_transform(H_fermion):
    """Apply BK to a FermionOperator. Returns (bk_terms, n_qubits)."""
    H_bk = bravyi_kitaev(H_fermion)
    n    = count_qubits(H_bk)
    terms = []
    for pauli_tuple, coeff in H_bk.terms.items():
        c = float(np.real(coeff))
        if abs(c) > 1e-14:
            terms.append((c, {idx: op for idx, op in pauli_tuple}))
    return terms, n


def parse_qubit_op(H_qubit):
    """
    QubitOperator -> bk_terms list of (float_coeff, {qubit: pauli_char}).
    Used by spin models that bypass BK entirely.
    """
    terms = []
    for pauli_tuple, coeff in H_qubit.terms.items():
        c = float(np.real(coeff))
        if abs(c) > 1e-14:
            terms.append((c, {idx: op for idx, op in pauli_tuple}))
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Ising polynomial arithmetic
# dict { frozenset(qubit_indices) -> float }
# Z_a Z_a = I  ->  product of monomials = symmetric difference of index sets
# ─────────────────────────────────────────────────────────────────────────────

def _poly_mul(p1, p2):
    out = defaultdict(complex)
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            out[s1.symmetric_difference(s2)] += c1 * c2
    return dict(out)


def _poly_clean(poly, tol=1e-12):
    return {m: v.real for m, v in poly.items() if abs(v.real) > tol}


# ─────────────────────────────────────────────────────────────────────────────
# XBK:  H'_p = sum_{j,k=1}^{2} S'(j) S'(k) H_(j,k)
#
# Physical qubit layout (r=2):
#   logical qubit i, copy c (1-based)  ->  2*i + (c-1)
#
# Same-copy rules (j==k):  X,Y->0;  Z->sp*Z_{qi,j};  I->1
# Cross-copy rules (j!=k):
#   X -> sp*(1 - Z_{qi,j}*Z_{qi,k})/2
#   Y -> i*sp*(Z_{qi,k} - Z_{qi,j})/2
#   Z -> sp*(Z_{qi,j} + Z_{qi,k})/2
#   I -> (1 + Z_{qi,j}*Z_{qi,k})/2
# ─────────────────────────────────────────────────────────────────────────────

def _phys(logical, copy):
    return 2 * logical + (copy - 1)


def _xbk_single(pauli, qi, j, k, sp):
    pj, pk = _phys(qi, j), _phys(qi, k)
    if j == k:
        if pauli in ('X', 'Y'):  return {}
        if pauli == 'Z':         return {frozenset([pj]): sp + 0j}
        return {frozenset(): 1.0 + 0j}
    if pauli == 'X':
        return {frozenset(): sp*0.5+0j, frozenset([pj,pk]): -sp*0.5+0j}
    if pauli == 'Y':
        return {frozenset([pk]): 1j*sp*0.5, frozenset([pj]): -1j*sp*0.5}
    if pauli == 'Z':
        return {frozenset([pj]): sp*0.5+0j, frozenset([pk]): sp*0.5+0j}
    return {frozenset(): 0.5+0j, frozenset([pj,pk]): 0.5+0j}


def build_Hprime(bk_terms, n_qubits, sector_signs):
    """Build H'_p as a Z-only Ising polynomial for a given sector."""
    H_prime = defaultdict(complex)
    for j in range(1, 3):
        for k in range(1, 3):
            for coeff, pd in bk_terms:
                term = {frozenset(): 1.0 + 0j}
                for qi in range(n_qubits):
                    term = _poly_mul(
                        term,
                        _xbk_single(pd.get(qi, 'I'), qi, j, k, sector_signs[qi])
                    )
                for mono, val in term.items():
                    H_prime[mono] += coeff * val
    return _poly_clean(dict(H_prime))


# ─────────────────────────────────────────────────────────────────────────────
# Counting operator — MPI scatter/gather over 2^n_qubits basis states
#
# C(s) = sum_b (P_{1,b}(s) + P_{2,b}(s))^2
# P_{j,b}(s) = prod_k (1 + (-1)^{b_k} s_{phys(k,j)}) / 2
# ─────────────────────────────────────────────────────────────────────────────

def _counting_partial(basis_list, n_qubits):
    partial = defaultdict(complex)
    for basis in basis_list:
        inner = defaultdict(complex)
        for copy_j in range(1, 3):
            p = {frozenset(): 1.0 + 0j}
            for k, b_k in enumerate(basis):
                ph     = _phys(k, copy_j)
                sign_k = float((-1) ** b_k)
                p      = _poly_mul(p, {frozenset(): 0.5+0j,
                                       frozenset([ph]): sign_k*0.5+0j})
            for m, v in p.items():
                inner[m] += v
        for m, v in _poly_mul(dict(inner), dict(inner)).items():
            partial[m] += v
    return dict(partial)


def build_counting_operator_parallel(n_qubits):
    """Build C_p in parallel; returns identical result on all ranks."""
    all_bases = list(iproduct([0, 1], repeat=n_qubits))

    if rank == 0:
        chunks = [[] for _ in range(nranks)]
        for i, b in enumerate(all_bases):
            chunks[i % nranks].append(b)
    else:
        chunks = None

    my_bases  = comm.scatter(chunks, root=0)
    local_C   = _counting_partial(my_bases, n_qubits)
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
# Quadratisation: k-local -> 2-local via Rosenberg substitution
# x_i = (1-s_i)/2;  penalty M*(x_a x_b - 2w x_a - 2w x_b + 3w) >= 0
# ─────────────────────────────────────────────────────────────────────────────

def _ising_to_binary(ising):
    out = defaultdict(float)
    for S, c in ising.items():
        sl = sorted(S);  k = len(sl)
        for tb in range(1 << k):
            T = frozenset(sl[m] for m in range(k) if (tb >> m) & 1)
            out[T] += c * ((-2) ** len(T))
    return dict(out)


def _binary_to_ising(binary):
    out = defaultdict(float)
    for T, c in binary.items():
        tl = sorted(T);  k = len(tl)
        for ub in range(1 << k):
            U = frozenset(tl[m] for m in range(k) if (ub >> m) & 1)
            out[U] += c * ((-1) ** len(U)) / (2 ** k)
    return dict(out)


def quadratize(binary, M, next_id):
    """Reduce binary polynomial to degree <= 2. Returns (poly, next_id)."""
    poly    = defaultdict(float, binary)
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
            changed = True;  break
    return dict(poly), next_id


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing — parallel chains via multiprocessing.Pool
# ─────────────────────────────────────────────────────────────────────────────

def _eval(ising, spins):
    E = 0.0
    for mono, c in ising.items():
        p = 1.0
        for i in mono:
            p *= spins[i]
        E += c * p
    return E


def _dE(ising, spins, flip):
    si = spins[flip];  field = 0.0
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
    ising, n_spins, n_sweeps, T_start, T_end, seed = args
    rng   = random.Random(seed)
    spins = {i: rng.choice([-1, 1]) for i in range(n_spins)}
    E     = _eval(ising, spins)
    best_s, best_E = dict(spins), E
    alpha = (T_end / T_start) ** (1.0 / max(n_sweeps, 1))
    T     = T_start
    for _ in range(n_sweeps):
        for __ in range(n_spins):
            i = rng.randint(0, n_spins - 1)
            d = _dE(ising, spins, i)
            if d < 0 or rng.random() < math.exp(-d / T):
                spins[i] = -spins[i];  E += d
                if E < best_E:
                    best_E = E;  best_s = dict(spins)
        T *= alpha
    return best_s, best_E


def sa_parallel(ising, n_spins, n_sweeps, T_start, T_end, base_seed):
    """Run SA_CHAINS chains in parallel. Returns (best_spins, best_energy)."""
    args = [(ising, n_spins, n_sweeps, T_start, T_end, base_seed + c)
            for c in range(SA_CHAINS)]
    if SA_CHAINS == 1:
        results = [_sa_chain(args[0])]
    else:
        with Pool(processes=SA_CHAINS) as pool:
            results = pool.map(_sa_chain, args)
    return min(results, key=lambda r: r[1])


# ─────────────────────────────────────────────────────────────────────────────
# Dinkelbach: E0 = min_s H'(s)/C(s)
# ─────────────────────────────────────────────────────────────────────────────

def dinkelbach(H_ising, C_ising, n_spins, n_sweeps, M, base_seed,
               max_iter=25, tol=1e-6):
    """Returns best E0, or +inf if C is always zero."""
    lam     = 5.0
    best_E0 = float('inf')
    T0      = max(3.0 * abs(M), 5.0)

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
# Single-instance pipeline: given bk_terms -> E0
# ─────────────────────────────────────────────────────────────────────────────

def run_one(bk_terms, n_qubits, C_ising, penalty_scale, n_sweeps, base_seed):
    """XBK + quadratise + Dinkelbach over all sectors. Returns minimum E0."""
    n_phys = n_qubits * 2
    M      = penalty_scale * max((abs(c) for c, _ in bk_terms), default=1.0)
    M      = max(M, 1.0)

    best_E0 = float('inf')
    for sec in range(2 ** n_qubits):
        sector_signs = [+1 if (sec >> i) & 1 == 0 else -1
                        for i in range(n_qubits)]

        H_ising = build_Hprime(bk_terms, n_qubits, sector_signs)

        next_id = n_phys
        H_quad, next_id = quadratize(_ising_to_binary(H_ising), M, next_id)
        C_quad, next_id = quadratize(_ising_to_binary(C_ising),  M, next_id)

        E0 = dinkelbach(_binary_to_ising(H_quad), _binary_to_ising(C_quad),
                        next_id, n_sweeps, M,
                        base_seed=base_seed + sec * 37)
        if E0 < best_E0:
            best_E0 = E0

    return best_E0


# ─────────────────────────────────────────────────────────────────────────────
# MPI sweep drivers
# ─────────────────────────────────────────────────────────────────────────────

def _gather_results(local_results, L_values):
    """Gather (L, ratio, E0) tuples from all ranks; organise by L on rank 0."""
    all_parts = comm.gather(local_results, root=0)
    if rank != 0:
        return None
    results = {L: [] for L in L_values}
    for part in all_parts:
        for L, ratio, E0 in part:
            results[L].append({'x': ratio, 'E0': E0})
    for L in L_values:
        results[L].sort(key=lambda r: r['x'])
    return results


def run_sweep(hamiltonian_fn, ratio_values, L_values,
              n_sweeps=2000, penalty_scale=12.0):
    """
    For each L in L_values, sweep ratio over ratio_values.
    hamiltonian_fn(ratio, L) must return a FermionOperator.
    Returns dict {L: [{'x': ratio, 'E0': float}]} on rank 0; None elsewhere.
    """
    all_jobs = [(L, r) for L in L_values for r in ratio_values]
    my_jobs  = [job for i, job in enumerate(all_jobs) if i % nranks == rank]

    local = []
    for L, ratio in my_jobs:
        seed       = int(abs(ratio * 100)) + L * 997 + rank * 9973
        bk_terms, n_qubits = bk_transform(hamiltonian_fn(ratio, L))
        C_ising    = build_counting_operator_parallel(n_qubits)
        E0         = run_one(bk_terms, n_qubits, C_ising,
                             penalty_scale, n_sweeps, seed)
        local.append((L, float(ratio), E0))

    return _gather_results(local, L_values)


def run_sweep_qubit(qubit_hamiltonian_fn, ratio_values, L_values,
                    n_sweeps=2000, penalty_scale=12.0):
    """
    Same as run_sweep but for spin models that skip BK.
    qubit_hamiltonian_fn(ratio, L) returns (bk_terms, n_qubits) directly.
    """
    all_jobs = [(L, r) for L in L_values for r in ratio_values]
    my_jobs  = [job for i, job in enumerate(all_jobs) if i % nranks == rank]

    local = []
    for L, ratio in my_jobs:
        seed               = int(abs(ratio * 100)) + L * 997 + rank * 9973
        bk_terms, n_qubits = qubit_hamiltonian_fn(ratio, L)
        C_ising            = build_counting_operator_parallel(n_qubits)
        E0                 = run_one(bk_terms, n_qubits, C_ising,
                                     penalty_scale, n_sweeps, seed)
        local.append((L, float(ratio), E0))

    return _gather_results(local, L_values)


# ─────────────────────────────────────────────────────────────────────────────
# Output: CSV + PNG (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────

_COLOURS = [
    "#1a1a2e", "#e63946", "#457b9d", "#2a9d8f",
    "#e9c46a", "#f4a261", "#6a4c93", "#264653",
]


def save_results(results, out_dir, model_name):
    """Write one CSV per L: columns ratio, E0_xbk_sa."""
    os.makedirs(out_dir, exist_ok=True)
    for L, rows in results.items():
        path = os.path.join(out_dir, f"{model_name}_L{L}.csv")
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['ratio', 'E0_xbk_sa'])
            for r in rows:
                w.writerow([f"{r['x']:.6f}", f"{r['E0']:.10f}"])


def save_plot(results, model_name, ratio_label, out_dir):
    """Single-axes plot, one curve per L value."""
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family"        : "DejaVu Serif",
        "mathtext.fontset"   : "dejavuserif",
        "font.size"          : 12,
        "axes.linewidth"     : 1.2,
        "xtick.direction"    : "in",   "ytick.direction"    : "in",
        "xtick.top"          : True,   "ytick.right"        : True,
        "xtick.major.size"   : 5,      "ytick.major.size"   : 5,
        "xtick.minor.size"   : 3,      "ytick.minor.size"   : 3,
        "xtick.minor.visible": True,   "ytick.minor.visible": True,
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, (L, rows) in enumerate(sorted(results.items())):
        colour = _COLOURS[idx % len(_COLOURS)]
        x = [r['x']  for r in rows]
        y = [r['E0'] for r in rows]
        ax.plot(x, y, color=colour, linewidth=1.8,
                marker='o', markersize=4,
                markerfacecolor=colour, markeredgecolor='white',
                markeredgewidth=0.6,
                label=f"$L = {L}$")

    ax.set_xlabel(ratio_label, fontsize=13)
    ax.set_ylabel(r"$E_0$  (XBK + SA)", fontsize=13)
    ax.set_title(model_name, fontsize=13)
    ax.legend(frameon=True, framealpha=0.95, fontsize=10,
              loc='best', handlelength=2.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model_name}_ground_state.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
