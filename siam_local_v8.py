"""
siam_local.py
=============
Ground-state pipeline for the Single-Impurity Anderson Model (SIAM),
N_bath = 1, 2, 3.

Two solvers run side by side:
  1. Exact diagonalisation (sector-projected scipy.linalg.eigh)
     — ground truth for all observables.
  2. XBK + Dinkelbach + SA  (prev_code.py / pipeline_base.py lineage)
     — local reproduction of the HPC pipeline column E0_SA.

==========================================================================
HAMILTONIAN
==========================================================================
H_SIAM = U n_{0↑} n_{0↓}  −  (U/2)(n_{0↑}+n_{0↓})  +  U/4
       + Σ_{b=1}^{N}  ε_b (n_{b↑}+n_{b↓})
       + Σ_{b=1}^{N}  V_b Σ_σ (a†_{0σ} a_{bσ} + h.c.)

Mode index:  2·site + spin  (spin 0=↑, 1=↓; site 0 = impurity)

Bath parameters (star / Wilson-chain discretisation):
  ε_b = −2t cos(bπ/(N+1))
  V_b =  V √(2/(N+1))  √(1 − (ε_b/2t)²)

==========================================================================
XBK TRANSFORM  (prev_code.py convention, block indexing: i + n·j)
==========================================================================
r replicas, p negative-sector replicas.
Physical spin for logical qubit i in replica j:  index = i + n·j

Same-copy rules (j == k):
  X, Y  →  0
  Z     →  Z_{i+n·j}
  I     →  1

Cross-copy rules (j ≠ k), with sign s = +1 if (j<p)==(k<p) else −1:
  X  →  s · ( 1/2  −  Z_{i+n·j} Z_{i+n·k} / 2 )
  Y  →  s · ( Z_{i+n·k}·i/2  −  Z_{i+n·j}·i/2 )
  Z  →  s · ( Z_{i+n·j}/2  +  Z_{i+n·k}/2 )
  I  →      ( 1/2  +  Z_{i+n·j} Z_{i+n·k} / 2 )

C(s) = Σ_b  [ Σ_{j=0}^{r-1} sign(j,p) · P_{j,b}(s) ]²

where  P_{j,b}(s) = Π_i  ( 1 + (−1)^{b_i} s_{i+n·j} ) / 2
and    sign(j,p)  = −1 if j < p,  +1 otherwise.

The Rayleigh quotient  E_p(s) = H'_p(s) / C_p(s)  converges to
λ_min(H_BK)  as r → ∞.  For r=3 (N=1) and r=3 (N=2) SA finds the
lowest accessible eigenvalue; exact E0 requires the exact solver.

==========================================================================
SECTOR PROJECTION  (mandatory for correct exact observables)
==========================================================================
H commutes with total electron number N_tot and total S_z.
Half-filling:  N_tot = N+1.
  N=1:  (N_tot=2, Sz=0)     unique ground state
  N=2:  (N_tot=3, Sz=+½)   non-degenerate within sector
  N=3:  (N_tot=4, Sz=0)     unique ground state

Without projection, scipy.eigsh mixes the 2-fold degenerate Sz sectors
for N=2, making fidelity / Wasserstein / spin_corr meaningless.

==========================================================================
OBSERVABLES
==========================================================================
E0_sa     XBK + Dinkelbach + SA  (replicates HPC pipeline output)
E0_exact  sector-projected exact diagonalisation
error     |E0_sa − E0_exact|
docc      ⟨n↑n↓⟩  (double occupancy of impurity)
docc_fd   dE0/dU  by finite difference
d2E0      d²E0/dU²  (peaks locate Kondo crossover scale)
spin_corr ⟨S_z^{imp} · S_z^{bath1}⟩
fidelity  |⟨ψ(U_k)|ψ(U_{k+1})⟩|²   (consecutive U/V steps)
d_WGS     W₁(|ψ(U_k)|², |ψ(U_{k+1})|²) with Hamming ground metric
"""

from __future__ import annotations
import os, csv, itertools, math, sys, time as _time
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openfermion as of
from openfermion import FermionOperator, bravyi_kitaev, QubitOperator
from openfermion.utils import count_qubits
from scipy.optimize import linprog
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dwave.samplers import SimulatedAnnealingSampler

# ── timestamped logger (works from worker processes) ─────────────────────────
_T0 = _time.time()

def _log(msg: str) -> None:
    """Print with elapsed time, flushed immediately (visible from subprocesses)."""
    elapsed = _time.time() - _T0
    m, s    = divmod(int(elapsed), 60)
    print(f"  [{m:02d}:{s:02d}] {msg}", flush=True)

# ── output ────────────────────────────────────────────────────────────────────
OUT = "output/siam_local"
os.makedirs(OUT, exist_ok=True)

# ── sweep parameters ──────────────────────────────────────────────────────────
T_HOP   = 1.0
V_HYB   = 1.0
EPS_IMP = 0.0

UV_VALS     = np.arange(0.0, 8.0 + 1e-9, 0.4)   # 21 U/V points
N_BATH_LIST = [1, 2, 3]

# Physical sector per N_bath: (N_tot, 2·Sz)
SECTOR = {1: (2, 0), 2: (3, 1), 3: (4, 0)}

# Finite-difference step: dU = max(FD_REL·U, FD_ABS)·V
FD_REL, FD_ABS = 0.05, 0.05

# XBK / SA settings
R_XBK       = 3          # number of replicas  (r=3 tractable for N=1,2,3)
SA_READS    = 300        # independent SA runs per Dinkelbach iteration
SA_MAX_ITER = 30
SA_TOL      = 1e-5
SA_LAM0     = 5.0
PENALTY_SCALE = 12.0     # Rosenberg penalty multiplier

# Parallelism (4-core i5)
OUTER_WORKERS = 3        # one per N_bath
INNER_WORKERS = 2        # per-N pool for U/V points


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SIAM HAMILTONIAN
# ══════════════════════════════════════════════════════════════════════════════

def _mode(site: int, spin: int) -> int:
    return 2 * site + spin


def build_siam(N: int, U: float,
               t: float = T_HOP, V: float = V_HYB,
               eps_imp: float = EPS_IMP) -> FermionOperator:
    b_idx = np.arange(1, N + 1)
    eps   = -2.0 * t * np.cos(b_idx * np.pi / (N + 1))
    Vb    =  V * np.sqrt(2.0 / (N + 1)) * np.sqrt(1.0 - (eps / (2.0 * t))**2)
    H = FermionOperator()
    H += FermionOperator(((0,1),(1,1),(1,0),(0,0)), U)
    for s in range(2):
        H += FermionOperator(((_mode(0,s),1),(_mode(0,s),0)), -U/2.0 + eps_imp)
    H += FermionOperator((), U / 4.0)
    for b, (e, v) in enumerate(zip(eps, Vb), 1):
        bm0, bm1 = _mode(b, 0), _mode(b, 1)
        H += FermionOperator(((bm0,1),(bm0,0)), e)
        H += FermionOperator(((bm1,1),(bm1,0)), e)
        H += FermionOperator(((0,1),(bm0,0)), v);   H += FermionOperator(((bm0,1),(0,0)), v)
        H += FermionOperator(((1,1),(bm1,0)), v);   H += FermionOperator(((bm1,1),(1,0)), v)
    return H


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EXACT DIAGONALISATION  (sector-projected)
# ══════════════════════════════════════════════════════════════════════════════

def _sector_basis(n_modes: int, n_elec: int, sz2: int) -> np.ndarray:
    """
    Integer array of Fock-space states with n_elec electrons and
    (n_up − n_dn) = sz2.  Bit k = occupation of mode k.
    Even modes = ↑, odd modes = ↓.
    """
    mask_up = sum(1 << k for k in range(0, n_modes, 2))
    mask_dn = sum(1 << k for k in range(1, n_modes, 2))
    return np.array(
        [s for s in range(2**n_modes)
         if (bin(s & mask_up).count('1') + bin(s & mask_dn).count('1') == n_elec
             and bin(s & mask_up).count('1') - bin(s & mask_dn).count('1') == sz2)],
        dtype=int)


def exact_gs(N: int, U: float):
    """Returns (E0, psi, basis) in the physical (N_tot, Sz) sector."""
    n_modes     = 2 * (N + 1)
    n_elec, sz2 = SECTOR[N]
    Hsp   = of.linalg.get_sparse_operator(build_siam(N, U), n_qubits=n_modes)
    basis = _sector_basis(n_modes, n_elec, sz2)
    Hsec  = Hsp[np.ix_(basis, basis)].toarray()
    vals, vecs = scipy.linalg.eigh(Hsec)
    psi = vecs[:, 0];  psi /= np.linalg.norm(psi)
    return float(vals[0]), psi, basis


# ══════════════════════════════════════════════════════════════════════════════
# 3.  XBK TRANSFORM  (prev_code.py convention, block indexing i + n·j)
# ══════════════════════════════════════════════════════════════════════════════

def xbk_transform(op: QubitOperator, r: int, p: int) -> QubitOperator:
    """
    Apply XBK to a QubitOperator.
    Physical spin for logical qubit i in replica j = i + n·j  (block layout).
    p replicas have sector sign −1; the remaining r−p have +1.
    Only p in range(ceil(r/2)+1) need to be tried (symmetry).
    """
    n       = count_qubits(op)
    new_op  = QubitOperator()

    for key, coeff in op.terms.items():
        term = QubitOperator()
        for j in range(r):
            for k in range(r):
                sign      = 1 if (j < p) == (k < p) else -1
                sub_term  = QubitOperator('', 1)
                spot      = 0

                for i in range(n):
                    # determine Pauli character at qubit i
                    try:
                        if key[spot][0] == i:
                            char = key[spot][1];  spot += 1
                        else:
                            char = 'I'
                    except IndexError:
                        char = 'I'

                    ij = i + n * j
                    ik = i + n * k

                    if char == 'X':
                        if j == k:
                            sub_term = QubitOperator('', 0);  break
                        sub_term *= (QubitOperator('', 0.5)
                                     - QubitOperator(f'Z{ij} Z{ik}', 0.5))
                    elif char == 'Y':
                        if j == k:
                            sub_term = QubitOperator('', 0);  break
                        sub_term *= (QubitOperator(f'Z{ik}',  0.5j)
                                     - QubitOperator(f'Z{ij}', 0.5j))
                    elif char == 'Z':
                        if j == k:
                            sub_term *= QubitOperator(f'Z{ij}', 1)
                        else:
                            sub_term *= (QubitOperator(f'Z{ij}', 0.5)
                                         + QubitOperator(f'Z{ik}', 0.5))
                    else:   # Identity
                        if j == k:
                            pass   # multiply by 1
                        else:
                            sub_term *= (QubitOperator('', 0.5)
                                         + QubitOperator(f'Z{ij} Z{ik}', 0.5))

                term += sign * sub_term

        new_op += coeff * term

    new_op.compress()
    return new_op


def counting_op(n: int, r: int, p: int) -> QubitOperator:
    """
    C_p(s) = Σ_b [ Σ_j sign(j,p) · P_{j,b}(s) ]²

    P_{j,b}(s) = Π_i  ( 1 + (−1)^{b_i} · Z_{i+n·j} ) / 2
    Iterates over all 2^n bitstrings b.
    """
    C = QubitOperator('', 0)
    for perm in itertools.product([1, -1], repeat=n):
        term = QubitOperator('', 0)
        for j in range(r):
            product = QubitOperator('', 1)
            for i in range(n):
                product *= (QubitOperator('', 0.5)
                            + QubitOperator(f'Z{i + n*j}', perm[i] * 0.5))
            sign = -1 if j < p else 1
            term += sign * product
        C += term ** 2
    return C


def _qubit_op_to_ising(qop: QubitOperator) -> dict:
    """QubitOperator (Z-only) → dict {(i,j,...): float}."""
    d: dict = {}
    for terms, coeff in qop.terms.items():
        key = tuple(idx for idx, _ in terms)
        d[key] = d.get(key, 0.0) + coeff.real
    return {k: v for k, v in d.items() if abs(v) > 1e-12}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  QUADRATISATION  (Rosenberg binary substitution)
# ══════════════════════════════════════════════════════════════════════════════

def _ising_to_binary(ising: dict) -> dict:
    """
    Convert spin (±1) polynomial to binary (0/1) polynomial.
    Substitution:  s_i = 1 − 2·x_i.
    Monomial s_{i1}·…·s_{ik} →  Σ_{T ⊆ {i1,…,ik}} (−2)^|T| · Π_{j∈T} x_j.
    """
    out: dict = defaultdict(float)
    for S, c in ising.items():
        sl = sorted(S);  k = len(sl)
        for tb in range(1 << k):
            T = frozenset(sl[m] for m in range(k) if (tb >> m) & 1)
            out[T] += c * ((-2) ** len(T))
    return dict(out)


def _binary_to_ising(binary: dict) -> dict:
    """Inverse of _ising_to_binary."""
    out: dict = defaultdict(float)
    for T, c in binary.items():
        tl = sorted(T);  k = len(tl)
        for ub in range(1 << k):
            U = frozenset(tl[m] for m in range(k) if (ub >> m) & 1)
            out[U] += c * ((-1) ** len(U)) / (2 ** k)
    return dict(out)


def _quadratize(binary: dict, M: float, next_id: int):
    """
    Reduce binary polynomial to degree ≤ 2 via Rosenberg's substitution.
    For each degree-3+ monomial with leading pair (a, b) and remainder R:
      Replace  x_a · x_b · R  by  w · R  and add penalty
        M · (x_a x_b − 2w x_a − 2w x_b + 3w) ≥ 0   (w is a new variable)
    Returns (quadratised_dict, next_available_index).
    """
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
            changed = True;  break
    return dict(poly), next_id


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DINKELBACH + SA
# ══════════════════════════════════════════════════════════════════════════════

def _eval_ising(ising: dict, spins: dict) -> float:
    """E = Σ_S c_S · Π_{i∈S} s_i.  Missing spins treated as +1."""
    E = 0.0
    for mono, c in ising.items():
        p = 1.0
        for i in mono:
            p *= spins.get(i, 1)
        E += c * p
    return E


import dimod as _dimod

def dinkelbach_sa(H_ising: dict, C_ising: dict, n_phys: int, M: float,
                  seed: int = 0, label: str = '') -> float:
    """
    Minimise  E0 = min_s H'(s)/C(s)  via Dinkelbach's algorithm.

    Quadratisation: dimod.make_quadratic directly on the Ising polynomial.
    Fast even for thousands of high-degree terms (no manual binary expansion).

    H'/C evaluated on PRE-quadratisation polynomials (physical spins only).
    """
    sampler = SimulatedAnnealingSampler()
    lam = SA_LAM0;  best = np.inf

    for it in range(SA_MAX_ITER):
        combined: dict = defaultdict(float, H_ising)
        for m, c in C_ising.items():
            combined[m] -= lam * c

        bqm = _dimod.make_quadratic(dict(combined), M, _dimod.SPIN)

        _log(f"{label}  Dinkelbach iter {it+1}/{SA_MAX_ITER}  λ={lam:.4f}"
             f"  SA({SA_READS} reads, {bqm.num_variables} vars) ...")

        ss    = sampler.sample(bqm, num_reads=SA_READS, seed=seed + it*1000)
        spins = dict(ss.first.sample)

        Hv = _eval_ising(H_ising, spins)
        Cv = _eval_ising(C_ising, spins)
        if abs(Cv) < 1e-8:
            _log(f"{label}  C=0, skipping")
            continue

        new_lam = Hv / Cv
        if new_lam < best:
            best = new_lam
        _log(f"{label}  Dinkelbach iter {it+1} done  λ={new_lam:.6f}  best={best:.6f}")
        if abs(new_lam - lam) < SA_TOL:
            lam = new_lam;  break
        lam = new_lam

    return best


def run_xbk_sa(N: int, U: float) -> float:
    """
    Full XBK + SA pipeline for one (N, U) point.
    Builds H_qubit once, builds C once per N (cached externally),
    sweeps p in range(ceil(r/2)+1) and returns the minimum E0_SA.
    """
    tag = f"N={N} U/V={U/V_HYB:.2f}"
    H_ferm   = build_siam(N, U)
    H_qubit  = bravyi_kitaev(H_ferm)
    nq       = count_qubits(H_qubit)
    n_phys   = R_XBK * nq
    M        = max(PENALTY_SCALE * max(abs(c) for c, _ in
                   [(float(np.real(c)), t) for t, c in H_qubit.terms.items()
                    if abs(np.real(c)) > 1e-14] or [(1.0, None)]), 1.0)
    seed     = int(abs(U * 100)) + N * 997

    # C is p-independent: build once
    _log(f"{tag}  building C  (nq={nq}, r={R_XBK}, 2^nq={2**nq} bitstrings) ...")
    C_op     = counting_op(nq, r=R_XBK, p=0)
    C_ising  = _qubit_op_to_ising(C_op)
    _log(f"{tag}  C built  ({len(C_ising)} terms)")

    n_p = int(math.ceil(R_XBK / 2 + 1))
    best = np.inf
    for p in range(n_p):
        _log(f"{tag}  XBK p={p}/{n_p-1}  building H' ...")
        H_op    = xbk_transform(H_qubit, r=R_XBK, p=p)
        H_ising = _qubit_op_to_ising(H_op)
        _log(f"{tag}  H' built  ({len(H_ising)} terms)  → Dinkelbach")
        E       = dinkelbach_sa(H_ising, C_ising, n_phys, M,
                                seed=seed + p*37,
                                label=f"{tag} p={p}")
        _log(f"{tag}  p={p} done  E_SA={E:.5f}")
        if E < best:
            best = E

    return best


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EXACT OBSERVABLES
# ══════════════════════════════════════════════════════════════════════════════

def _double_occupancy(psi: np.ndarray, basis: np.ndarray) -> float:
    """⟨n_{0↑} n_{0↓}⟩ = Σ_state |ψ|² · bit0 · bit1."""
    return float(np.sum(np.abs(psi)**2
                        * ((basis >> 0) & 1).astype(float)
                        * ((basis >> 1) & 1).astype(float)))


def _spin_corr(psi: np.ndarray, basis: np.ndarray) -> float:
    """⟨S_z^{imp} · S_z^{bath1}⟩ = ¼⟨(n_{0↑}−n_{0↓})(n_{1↑}−n_{1↓})⟩."""
    sz_imp  = ((basis>>0)&1).astype(float) - ((basis>>1)&1).astype(float)
    sz_bath = ((basis>>2)&1).astype(float) - ((basis>>3)&1).astype(float)
    return float(np.sum(np.abs(psi)**2 * sz_imp * sz_bath)) * 0.25


def _fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    return float(abs(np.dot(psi1.conj(), psi2))**2)


def _wasserstein(p: np.ndarray, q: np.ndarray, basis: np.ndarray) -> float:
    """
    1-Wasserstein with Hamming ground metric, solved as LP.
    Drops last column-marginal constraint (redundant, prevents HiGHS
    infeasibility from floating-point inconsistency).
    """
    p = np.array(p, dtype=np.float64);  p /= p.sum()
    q = np.array(q, dtype=np.float64);  q /= q.sum()
    n = len(p)
    C_mat = np.array([[bin(int(basis[i])^int(basis[j])).count('1')
                       for j in range(n)] for i in range(n)], dtype=np.float64)
    Aeq = np.zeros((2*n - 1, n*n));  beq = np.zeros(2*n - 1)
    for i in range(n):
        Aeq[i, i*n:(i+1)*n] = 1.0;  beq[i] = p[i]
    for j in range(n - 1):
        Aeq[n+j, j::n] = 1.0;  beq[n+j] = q[j]
    res = linprog(C_mat.flatten(), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)]*(n*n), method='highs')
    return float(res.fun) if res.success else float('nan')


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PER-POINT WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _compute_point(args):
    """Compute all quantities for one (N, uv) pair. Runs in worker process."""
    N, uv = args
    U  = uv * V_HYB
    dU = max(FD_REL * U, FD_ABS) * V_HYB
    tag = f"N={N} U/V={uv:.2f}"

    _log(f"{tag}  exact diag (E0, E+, E-) ...")
    E0,  psi,  basis = exact_gs(N, U)
    Ep,  _,    _     = exact_gs(N, U + dU)
    Em,  _,    _     = exact_gs(N, U - dU)
    _log(f"{tag}  exact done  E0={E0:.5f}")

    _log(f"{tag}  XBK + SA ...")
    E0_sa = run_xbk_sa(N, U)
    _log(f"{tag}  ✓ DONE  E0_sa={E0_sa:.5f}  E0_exact={E0:.5f}  err={abs(E0_sa-E0):.3f}")

    return {
        'uv':        float(uv),
        'E0':        E0,
        'E0_sa':     E0_sa,
        'error':     abs(E0_sa - E0),
        'docc':      _double_occupancy(psi, basis),
        'docc_fd':   (Ep - Em) / (2 * dU),
        'd2E0':      (Ep - 2*E0 + Em) / dU**2,
        'spin_corr': _spin_corr(psi, basis),
        'psi':       psi,
        'basis':     basis,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(N: int) -> dict:
    n_pts = len(UV_VALS)
    n_p   = int(math.ceil(R_XBK / 2 + 1))
    _log(f"━━━ N_bath={N}  starting sweep  "
         f"({n_pts} U/V points × {n_p} sectors × ≤{SA_MAX_ITER} Dinkelbach iters) ━━━")

    args = [(N, uv) for uv in UV_VALS]
    point_results = [None] * n_pts
    done = 0

    with ProcessPoolExecutor(max_workers=INNER_WORKERS) as pool:
        futures = {pool.submit(_compute_point, a): idx for idx, a in enumerate(args)}
        for fut in as_completed(futures):
            idx = futures[fut]
            point_results[idx] = fut.result()
            done += 1
            r = point_results[idx]
            _log(f"N={N}  [{done}/{n_pts} points done]  "
                 f"U/V={r['uv']:.2f}  E0_sa={r['E0_sa']:.4f}  "
                 f"E0_exact={r['E0']:.5f}  err={r['error']:.3f}")

    point_results.sort(key=lambda r: r['uv'])

    # Consecutive-pair observables (serial — need adjacent psi vectors)
    _log(f"N={N}  computing fidelity + Wasserstein for {n_pts-1} consecutive pairs ...")
    fid_vals  = [1.0]
    dwgs_vals = [0.0]
    for i in range(1, len(point_results)):
        fid_vals.append(_fidelity(point_results[i-1]['psi'], point_results[i]['psi']))
        dwgs_vals.append(_wasserstein(
            np.abs(point_results[i-1]['psi'])**2,
            np.abs(point_results[i]['psi'])**2,
            point_results[i]['basis']))
        if i % 5 == 0 or i == n_pts - 1:
            _log(f"N={N}  Wasserstein {i}/{n_pts-1} done")

    res = {
        'uv':        [r['uv']        for r in point_results],
        'E0_sa':     [r['E0_sa']     for r in point_results],
        'E0_exact':  [r['E0']        for r in point_results],
        'error':     [r['error']     for r in point_results],
        'docc':      [r['docc']      for r in point_results],
        'docc_fd':   [r['docc_fd']   for r in point_results],
        'd2E0':      [r['d2E0']      for r in point_results],
        'spin_corr': [r['spin_corr'] for r in point_results],
        'fidelity':  fid_vals,
        'd_WGS':     dwgs_vals,
    }

    _log(f"━━━ N_bath={N}  sweep complete ━━━")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _plot_all(all_res: dict) -> None:
    colors  = {1:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c'}
    markers = {1:'o', 2:'s', 3:'^'}
    panels = [
        ('E0_exact',  r'$E_0$ (exact)',
                      'Ground-state energy  (sector-projected exact)'),
        ('E0_sa',     r'$E_0$ (SA, $r=%d$)' % R_XBK,
                      f'XBK + SA ground-state energy  (r={R_XBK})'),
        ('error',     r'$|E_0^{\rm SA}-E_0^{\rm exact}|$',
                      'SA absolute error'),
        ('docc',      r'$\langle n_\uparrow n_\downarrow\rangle$',
                      'Double occupancy'),
        ('docc_fd',   r'$dE_0/dU$',
                      r'$dE_0/dU$  (finite difference)'),
        ('d2E0',      r'$d^2E_0/dU^2$',
                      r'$d^2E_0/dU^2$  (Kondo crossover scale)'),
        ('spin_corr', r'$\langle S_z^{\rm imp}S_z^{\rm bath}\rangle$',
                      'Impurity–bath spin correlation'),
        ('fidelity',  r'$|\langle\psi_k|\psi_{k+1}\rangle|^2$',
                      'Quantum fidelity  (consecutive $U/V$ steps)'),
        ('d_WGS',     r'$W_1$  (Wasserstein, Hamming)',
                      r'$d_{\rm WGS}$  (consecutive $U/V$ steps)'),
    ]
    for key, ylabel, title in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        for N, r in all_res.items():
            ax.plot(r['uv'], r[key], color=colors[N], marker=markers[N],
                    markersize=4, linewidth=1.5, label=f'$N_{{\\rm bath}}={N}$')
        ax.set_xlabel(r'$U/V$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'{key}.pdf'), dpi=150)
        plt.close(fig)
        print(f"  Saved {key}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CSV
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(N: int, res: dict) -> None:
    cols = ['uv','E0_sa','E0_exact','error','docc','docc_fd',
            'd2E0','spin_corr','fidelity','d_WGS']
    with open(os.path.join(OUT, f'siam_N{N}.csv'), 'w', newline='') as f:
        w = csv.writer(f);  w.writerow(cols)
        for i in range(len(res['uv'])):
            w.writerow([res[c][i] for c in cols])
    print(f"  Saved siam_N{N}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time
    t0 = time.time()

    n_pts   = len(UV_VALS)
    n_p     = int(math.ceil(R_XBK / 2 + 1))
    n_total = len(N_BATH_LIST) * n_pts * n_p * SA_MAX_ITER
    print(f"\nSIAM local pipeline  —  r={R_XBK}, {n_pts} U/V points, "
          f"{len(N_BATH_LIST)} N_bath values")
    print(f"Max Dinkelbach iters per sector: {SA_MAX_ITER}  "
          f"(converges early when |Δλ| < {SA_TOL})")
    print(f"SA reads per iter: {SA_READS}")
    print(f"Outputs: {OUT}/\n")

    all_res: dict = {}
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as pool:
        futures = {pool.submit(run_sweep, N): N for N in N_BATH_LIST}
        for fut in as_completed(futures):
            N = futures[fut]
            all_res[N] = fut.result()
            elapsed = time.time() - t0
            m, s = divmod(int(elapsed), 60)
            print(f"\n[{m:02d}:{s:02d}] N_bath={N} sweep finished.\n", flush=True)

    print("\nWriting CSV and figures ...")
    for N in N_BATH_LIST:
        _save_csv(N, all_res[N])
    _plot_all(all_res)

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"\n[{m:02d}:{s:02d}] All done  —  outputs in {OUT}/")
