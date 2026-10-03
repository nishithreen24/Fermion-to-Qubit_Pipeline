"""
siam_local.py
=============
Self-contained SIAM pipeline for N_bath = 1, 2, 3.
No MPI.  Uses neal (simulated annealing) and scipy exact diagonalisation.

PHYSICAL SETUP
--------------
The Single-Impurity Anderson Model (SIAM) Hamiltonian is:

    H = U n_{imp,up} n_{imp,dn}
        - (U/2)(n_{imp,up} + n_{imp,dn})          [particle-hole symmetry shift]
        + U/4                                       [constant]
        + sum_{b=1}^{N} eps_b (n_{b,up} + n_{b,dn})
        + sum_{b=1}^{N} V_b sum_sigma
              (a†_{imp,sigma} a_{b,sigma} + h.c.)

where a† / a are fermionic creation / annihilation operators.
n_{i,sigma} = a†_{i,sigma} a_{i,sigma} is the occupation-number operator
(eigenvalue 0 or 1).

Mode indexing:  mode = 2*site + spin   (spin 0=up, 1=dn)
  site 0   -> impurity  (modes 0, 1)
  site b   -> bath b    (modes 2b, 2b+1),   b = 1..N

Bath discretisation (star geometry / Wilson chain correspondence):
  eps_b = -2t cos(b pi / (N+1))
  V_b   =  V  sqrt(2/(N+1)) sqrt(1 - (eps_b / 2t)^2)

PIPELINE
--------
1.  Build H_fermion as FermionOperator.
2.  Bravyi-Kitaev (BK) transform -> H_qubit (QubitOperator, Pauli strings).
    The BK map replaces each fermionic mode with a qubit such that the parity
    information is stored in O(log n) qubits instead of O(n) (Jordan-Wigner).
3.  XBK diagonalisation with r=2 replicas:
    Each logical qubit i is replaced by two physical spins s_{i,1}, s_{i,2} in {+/-1}.
    The replacement rules are:
      X_i -> (s_{i,1} s_{i,2} - 1) / 2       * s_p   (sector sign s_p)
      Y_i -> (s_{i,2} - s_{i,1}) / 2 * i     * s_p
      Z_i -> (s_{i,1} + s_{i,2}) / 2          * s_p
      I_i -> (1 + s_{i,1} s_{i,2}) / 2
    After substituting all qubits this way, H_qubit becomes a polynomial H'(s)
    in 2*n_qubits binary (±1) spin variables.
4.  Counting operator C(s): C = sum_b (P_{1,b} + P_{2,b})^2
    where P_{j,b} = prod_i [(s_{j,i} * sigma_b,i + 1) / 2]  is the projector
    onto bitstring b in replica j.  This is the denominator of the Rayleigh
    quotient.
5.  Quadratisation: replace each monomial of degree > 2 by auxiliary binary
    variables using Rosenberg's substitution with penalty M.
    This turns H'(s) and C(s) into quadratic unconstrained binary (QUBO) forms.
6.  Dinkelbach iteration:
      lambda_0 = 0
      loop:
        s* = argmin_s  [ H'(s) - lambda_k * C(s) ]   (solved by SA on QUBO)
        lambda_{k+1} = H'(s*) / C(s*)                 (evaluated on PRE-QUAD spins)
        if |lambda_{k+1} - lambda_k| < tol: break
    The minimiser lambda* = E0, the ground-state energy.

    CRITICAL (bug fix): H'(s*) and C(s*) must be evaluated on the
    pre-quadratisation polynomials using only physical spin indices 0..n_phys-1.
    The auxiliary variables introduced by quadratisation are at indices >= n_phys.
    Using the full (augmented) spin vector gives wrong energies.

OBSERVABLES
-----------
For each U/V value and each N_bath we compute:

  E0          Ground-state energy (Dinkelbach converged value).
  E0_exact    Exact E0 from scipy sparse eigensolver (ground truth).
  error       |E0 - E0_exact|
  docc        Double occupancy <n_up n_down> = dE0/dU  (Hellmann-Feynman)
              approximated by finite difference: (E0(U+dU) - E0(U-dU)) / (2 dU)
  d2E0        d^2 E0 / dU^2  (second derivative, finite difference)
              peaks/dips mark the Kondo crossover scale
  spin_corr   Impurity-bath spin correlation: <S_z^imp * S_z^b1>
              where S_z = (n_up - n_dn)/2.
              Computed from exact ground state vector.
  fidelity    Quantum fidelity between ground states at consecutive U/V values.
              F(psi_1, psi_2) = |<psi_1|psi_2>|^2.
              Drops sharply near a phase transition / crossover.
  d_WGS       Wasserstein distance (1-Wasserstein on bitstrings, using Hamming
              distance as the ground metric) between the probability distributions
              |psi(U/V)|^2 and |psi(U/V + step)|^2 over the 2^n computational
              basis states.  This is the key geometric observable in the paper.
"""

import os
import csv
import itertools
from collections import defaultdict
from math import prod

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openfermion as of
from openfermion import FermionOperator, bravyi_kitaev
from neal import SimulatedAnnealingSampler
import dimod

# ===========================================================================
# OUTPUT DIRECTORY
# ===========================================================================
OUT = "output/siam_local"
os.makedirs(OUT, exist_ok=True)

# ===========================================================================
# PARAMETERS
# ===========================================================================
T_HOP   = 1.0          # bath hopping amplitude (sets half-bandwidth W = 2t)
V_HYB   = 1.0          # impurity-bath hybridisation
EPS_IMP = 0.0          # impurity on-site energy (0 = particle-hole symmetric)

UV_MIN  = 0.0
UV_MAX  = 8.0
UV_STEP = 0.4
UV_VALS = np.arange(UV_MIN, UV_MAX + 1e-9, UV_STEP)

N_BATH_LIST = [1, 2, 3]

# SA settings
SA_SWEEPS = 20_000
SA_READS  = 200

# Dinkelbach settings
DK_TOL      = 1e-6
DK_MAXITER  = 60

# Finite-difference step for dE0/dU and d^2E0/dU^2
FD_DU = 0.05

# Quadratisation penalty multiplier
PENALTY_MULT = 12.0

# ===========================================================================
# 1.  SIAM HAMILTONIAN  (from Models.py, reproduced here for self-containment)
# ===========================================================================

def _mode(site: int, spin: int) -> int:
    """mode index = 2*site + spin.  site 0 = impurity, site b = bath b."""
    return 2 * site + spin


def build_siam(N: int, U: float, t: float = T_HOP,
               V: float = V_HYB, eps_imp: float = EPS_IMP) -> FermionOperator:
    """
    Build SIAM FermionOperator for N bath sites.
    Total spin-orbitals: 2*(N+1).
    """
    b_idx = np.arange(1, N + 1)
    eps_bath = -2.0 * t * np.cos(b_idx * np.pi / (N + 1))
    V_bath   =  V * np.sqrt(2.0 / (N + 1)) * np.sqrt(
                    1.0 - (eps_bath / (2.0 * t))**2)

    H = FermionOperator()

    # --- impurity ---
    up_imp = _mode(0, 0)
    dn_imp = _mode(0, 1)

    # U n_up n_dn  (normal ordered)
    H += FermionOperator(((up_imp,1),(dn_imp,1),(dn_imp,0),(up_imp,0)), U)
    # -(U/2)(n_up + n_dn) + eps_imp*(n_up + n_dn)
    one_body = -U / 2.0 + eps_imp
    for spin in range(2):
        H += FermionOperator(((_mode(0, spin),1), (_mode(0, spin),0)), one_body)
    # constant U/4
    H += FermionOperator((), U / 4.0)

    # --- bath on-site energies ---
    for b, eps_b in enumerate(eps_bath, start=1):
        for spin in range(2):
            H += FermionOperator(((_mode(b, spin),1), (_mode(b, spin),0)), eps_b)

    # --- hybridisation ---
    for b, V_b in enumerate(V_bath, start=1):
        for spin in range(2):
            imp_m  = _mode(0, spin)
            bath_m = _mode(b, spin)
            H += FermionOperator(((imp_m,1),(bath_m,0)),  V_b)
            H += FermionOperator(((bath_m,1),(imp_m,0)),  V_b)

    return H


# ===========================================================================
# 2.  XBK DIAGONALISATION  (r=2 replicas, sector p)
# ===========================================================================

def xbk(H_qubit: of.QubitOperator, n: int,
        r: int = 2, p: int = 0) -> of.QubitOperator:
    """
    XBK transform with r replicas and p replicas having sector sign -1.

    Each logical qubit i -> physical spins (j*n+i) for replica j.
    The result H_ising is a QubitOperator in Z-only form
    (representing a polynomial in ±1 spin variables).

    Parameters
    ----------
    H_qubit : QubitOperator   (Pauli strings)
    n       : number of logical qubits
    r       : number of replicas  (use r=2)
    p       : number of replicas with sector sign -1  (0 <= p <= r)

    Returns
    -------
    H_ising : QubitOperator (Z-only = polynomial in s_{j,i} ∈ {±1})
    """
    H_ising = of.QubitOperator()
    for terms, coeff in H_qubit.terms.items():
        for j in range(r):
            Sj = -1 if j < p else 1
            for k in range(r):
                Sk = -1 if k < p else 1
                qc   = -1          # pointer into `terms` list (reversed)
                Qbar = of.QubitOperator(())
                for i in range(n - 1, -1, -1):
                    Zj = of.QubitOperator(f'Z{j * n + i}')
                    Zk = of.QubitOperator(f'Z{k * n + i}')
                    if len(terms) == 0 or i < terms[0][0] or i > terms[qc][0]:
                        # Identity position: I_i -> (1 + Z_{i,1} Z_{i,2}) / 2
                        pauliop = (Zj * Zk + of.QubitOperator(())) * 0.5
                    elif i == terms[qc][0]:
                        gate = terms[qc][1]
                        if gate == 'X':
                            # X_i -> s_p (Z_{i,1} Z_{i,2} - 1) / 2
                            pauliop = (Zj * Zk - of.QubitOperator(())) * (-0.5)
                        elif gate == 'Y':
                            # Y_i -> i*s_p (Z_{i,2} - Z_{i,1}) / 2
                            pauliop = (Zk - Zj) * 0.5j
                        else:  # 'Z'
                            # Z_i -> s_p (Z_{i,1} + Z_{i,2}) / 2
                            pauliop = (Zj + Zk) * 0.5
                        qc -= 1
                    Qbar *= pauliop
                H_ising += coeff * Sj * Sk * Qbar
    return H_ising


# ===========================================================================
# 3.  COUNTING OPERATOR  C(s)
# ===========================================================================

def counting_op(n: int, r: int = 2, p: int = 0) -> of.QubitOperator:
    """
    Build C = sum_b (P_{1,b} + P_{2,b})^2

    where P_{j,b} = prod_i [(s_{j,i} * sigma_{b,i} + 1) / 2]
    is the projector onto bitstring b in replica j.
    sigma_{b,i} in {+1,-1} is the i-th spin of bitstring b.

    This iterates over all 2^n bitstrings b.
    """
    C = of.QubitOperator()
    for signs in itertools.product([1, -1], repeat=n):
        bitstring_op = of.QubitOperator()
        for j in range(r):
            Sj      = -1 if j < p else 1
            prod_op = of.QubitOperator(())
            for i in range(n):
                Zji    = of.QubitOperator(f'Z{j * n + i}')
                prod_op *= (Zji * signs[i] + of.QubitOperator(())) * 0.5
            bitstring_op += prod_op * Sj
        C += bitstring_op * bitstring_op
    return C


# ===========================================================================
# 4.  CONVERT QubitOperator -> coefficient dict  {(idx,...): float}
# ===========================================================================

def to_dict(qop: of.QubitOperator) -> dict:
    """
    Convert a Z-only QubitOperator to a plain dict of the form
      { (i, j, ...): float_coefficient }
    where each key is a sorted tuple of qubit indices.
    The empty tuple () represents the constant term.
    """
    d = defaultdict(float)
    for terms, coeff in qop.terms.items():
        key = tuple(idx for idx, _ in terms)
        d[key] += coeff.real
    return dict(d)


# ===========================================================================
# 5.  QUADRATISATION  (Rosenberg k->2 reduction)
# ===========================================================================

def quadratise(poly: dict, penalty_mult: float = PENALTY_MULT) -> dict:
    """
    Reduce a higher-order polynomial over {±1} spins to degree ≤ 2
    using Rosenberg's substitution.

    For each pair (a, b) appearing in a monomial of degree > 2:
      Introduce auxiliary variable w_{ab}.
      Replace s_a * s_b by w_{ab} everywhere.
      Add penalty  M * (s_a * s_b - 2*w*s_a - 2*w*s_b + 3*w) >= 0
      where M = penalty_mult * max|coefficient|.

    Returns the quadratised dict ready for dimod / neal.
    """
    # We use dimod's built-in higher-order reduction which implements exactly
    # this Rosenberg penalty.
    M = penalty_mult * max(abs(v) for v in poly.values()) if poly else 1.0
    bqm = dimod.higherorder.utils.make_quadratic(poly, M, dimod.SPIN)
    return bqm


# ===========================================================================
# 6.  EVALUATE pre-quadratisation polynomial on a spin assignment
# ===========================================================================

def eval_poly(poly: dict, spin: dict, n_phys: int) -> float:
    """
    Evaluate polynomial `poly` (keys = tuples of variable indices) at the
    spin assignment `spin`, using ONLY indices 0..n_phys-1.

    This is the CRITICAL fix: we ignore auxiliary variables (index >= n_phys)
    introduced by quadratisation. The polynomial `poly` is the pre-quadratisation
    one so all its indices are < n_phys already; but we check explicitly.
    """
    val = 0.0
    for key, coeff in poly.items():
        # skip any term that references an auxiliary index
        if any(idx >= n_phys for idx in key):
            continue
        monomial = prod(spin.get(idx, 0) for idx in key) if key else 1
        val += coeff * monomial
    return val


# ===========================================================================
# 7.  DINKELBACH ITERATION
# ===========================================================================

def dinkelbach(H_poly: dict, C_poly: dict, n_phys: int,
               lam0: float = 0.0,
               tol: float = DK_TOL,
               max_iter: int = DK_MAXITER) -> tuple:
    """
    Minimise  E0 = min_s  H'(s) / C(s)  using Dinkelbach's algorithm.

    At each step k:
      s*_k = argmin_s  [ H'(s) - lambda_k * C(s) ]   <- SA on quadratised QUBO
      lambda_{k+1} = H'(s*_k) / C(s*_k)               <- on PRE-QUAD polynomials

    Parameters
    ----------
    H_poly  : pre-quadratisation dict for H' (physical spins 0..n_phys-1)
    C_poly  : pre-quadratisation dict for C  (physical spins 0..n_phys-1)
    n_phys  : number of physical spins (indices that matter for energy eval)
    lam0    : initial guess for E0
    tol, max_iter : convergence settings

    Returns
    -------
    (lam, spins, n_iter)
      lam    : converged ground-state energy
      spins  : best spin assignment {idx: ±1}
      n_iter : number of Dinkelbach iterations taken
    """
    sampler = SimulatedAnnealingSampler()
    lam    = lam0
    best_s = None

    for k in range(max_iter):
        # --- build shifted polynomial H'(s) - lambda_k * C(s) ---
        shifted = defaultdict(float)
        for key, val in H_poly.items():
            shifted[key] += val
        for key, val in C_poly.items():
            shifted[key] -= lam * val
        shifted_plain = dict(shifted)

        # --- quadratise and run SA ---
        bqm = quadratise(shifted_plain)
        try:
            sampleset = sampler.sample(bqm,
                                       num_sweeps=SA_SWEEPS,
                                       num_reads=SA_READS)
        except Exception:
            continue

        gse  = sampleset.first.energy
        spins = dict(sampleset.first.sample)

        # --- check convergence: min is non-negative -> done ---
        if gse >= -tol:
            break

        # --- evaluate ratio on PRE-QUAD physical spins ONLY ---
        H_val = eval_poly(H_poly, spins, n_phys)
        C_val = eval_poly(C_poly, spins, n_phys)

        if abs(C_val) < 1e-12:
            break

        lam_new = H_val / C_val
        best_s  = spins

        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new

    return lam, best_s, k + 1


# ===========================================================================
# 8.  EXACT DIAGONALISATION via scipy sparse
# ===========================================================================

def exact_gs(H_fermion: FermionOperator, n_modes: int):
    """
    Return (E0, psi0) where E0 is the ground-state energy and psi0 is
    the ground-state vector in the 2^n_modes dimensional Fock space.

    The Fock space basis is ordered as |b_{n-1} ... b_1 b_0> where b_i in {0,1}
    is the occupation of mode i.
    """
    H_sparse = of.linalg.get_sparse_operator(H_fermion, n_qubits=n_modes)
    if H_sparse.shape[0] <= 4:
        vals, vecs = np.linalg.eigh(H_sparse.toarray())
        E0  = vals[0]
        psi = vecs[:, 0]
    else:
        vals, vecs = spla.eigsh(H_sparse, k=1, which='SA')
        E0  = vals[0]
        psi = vecs[:, 0]
    # normalise
    psi = psi / np.linalg.norm(psi)
    return float(E0), psi


# ===========================================================================
# 9.  OBSERVABLES FROM EXACT GROUND STATE
# ===========================================================================

def double_occupancy_exact(psi: np.ndarray, N: int) -> float:
    """
    <n_{imp,up} n_{imp,dn}> = <a†_0 a_0 a†_1 a_1>
    computed directly from exact psi.
    """
    n_modes = 2 * (N + 1)
    nop_up = of.linalg.get_sparse_operator(
        FermionOperator(((0,1),(0,0))), n_qubits=n_modes)
    nop_dn = of.linalg.get_sparse_operator(
        FermionOperator(((1,1),(1,0))), n_qubits=n_modes)
    docc_op = nop_up.dot(nop_dn)
    return float(np.real(psi.conj() @ docc_op.dot(psi)))


def spin_corr_exact(psi: np.ndarray, N: int) -> float:
    """
    <S_z^imp * S_z^bath1>  where S_z = (n_up - n_dn) / 2.

    This quantity measures the correlation between the impurity spin
    and the first bath site.  In the Kondo-screened phase it becomes
    large and negative (singlet formation).
    """
    if N < 1:
        return 0.0
    n_modes = 2 * (N + 1)

    def sz_op(site: int):
        nup = of.linalg.get_sparse_operator(
            FermionOperator(((_mode(site,0),1),(_mode(site,0),0))),
            n_qubits=n_modes)
        ndn = of.linalg.get_sparse_operator(
            FermionOperator(((_mode(site,1),1),(_mode(site,1),0))),
            n_qubits=n_modes)
        return (nup - ndn) * 0.5

    Sz_imp   = sz_op(0)
    Sz_bath1 = sz_op(1)
    prod_op  = Sz_imp.dot(Sz_bath1)
    return float(np.real(psi.conj() @ prod_op.dot(psi)))


def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    Quantum fidelity F = |<psi1|psi2>|^2.
    psi1 and psi2 must live in the same Hilbert space (same dimension).
    """
    return float(np.abs(np.dot(psi1.conj(), psi2))**2)


def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """
    1-Wasserstein distance between two discrete probability distributions
    p and q over the 2^n computational basis, using Hamming distance
    as the ground metric.

    Formula (exact, O(2^{2n}) cost):
      W_1(p, q) = min_{pi} sum_{x,y} pi(x,y) * d_H(x, y)
    where d_H(x, y) = number of differing bits between x and y
    and pi is a joint distribution with marginals p, q.

    We use scipy's linear programming to solve the transport problem exactly.
    For large n this is expensive; for N_bath=1,2,3 the Hilbert space has
    2^4, 2^6, 2^8 = 16, 64, 256 states, which is tractable.
    """
    from scipy.optimize import linprog

    n = len(p)
    assert len(q) == n

    # Cost matrix: C[x,y] = Hamming distance between x and y
    n_bits = int(np.log2(n))
    C = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            C[x, y] = bin(x ^ y).count('1')

    # Flatten variables pi_{x,y} -> vector of length n^2
    c_vec = C.flatten()

    # Equality constraints:
    #   sum_y pi_{x,y} = p[x]  for each x   (row marginal)
    #   sum_x pi_{x,y} = q[y]  for each y   (col marginal)
    A_eq = np.zeros((2 * n, n * n))
    b_eq = np.zeros(2 * n)
    for x in range(n):
        A_eq[x, x*n:(x+1)*n] = 1.0
        b_eq[x] = p[x]
    for y in range(n):
        A_eq[n + y, y::n] = 1.0
        b_eq[n + y] = q[y]

    bounds = [(0, None)] * (n * n)
    result = linprog(c_vec, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if result.success:
        return float(result.fun)
    return float('nan')


# ===========================================================================
# 10.  FULL SWEEP FOR ONE N_BATH
# ===========================================================================

def run_siam_sweep(N: int) -> dict:
    """
    Sweep U/V from UV_MIN to UV_MAX for a fixed N bath sites.
    Returns a dict of lists, one entry per U/V value:
      uv_vals, E0_sa, E0_exact, error, docc_fd, d2E0, spin_corr, fidelity, d_WGS
    """
    print(f"\n{'='*60}")
    print(f"  SIAM  N_bath = {N}")
    print(f"{'='*60}")

    n_modes  = 2 * (N + 1)   # total spin-orbital modes
    r        = 2              # XBK replicas

    results = {k: [] for k in [
        'uv', 'E0_sa', 'E0_exact', 'error',
        'docc_fd', 'docc_exact', 'd2E0',
        'spin_corr', 'fidelity', 'd_WGS']}

    # We store exact psi for fidelity / Wasserstein between consecutive points
    prev_psi = None
    prev_probs = None

    for uv in UV_VALS:
        U  = uv * V_HYB
        print(f"\n  U/V = {uv:.2f}  (U = {U:.3f})")

        # ---- exact ground state ----
        H_ferm    = build_siam(N, U)
        E0_ex, psi0 = exact_gs(H_ferm, n_modes)
        probs       = np.abs(psi0)**2   # probability over 2^n_modes states

        # ---- finite-difference dE0/dU and d^2E0/dU^2 ----
        H_plus  = build_siam(N, U + FD_DU * V_HYB)
        H_minus = build_siam(N, U - FD_DU * V_HYB)
        E_plus,  _ = exact_gs(H_plus,  n_modes)
        E_minus, _ = exact_gs(H_minus, n_modes)
        docc_fd  = (E_plus - E_minus) / (2 * FD_DU * V_HYB)
        # second derivative needs E0 and E±
        d2E0 = (E_plus - 2 * E0_ex + E_minus) / (FD_DU * V_HYB)**2

        # ---- exact double occupancy ----
        docc_ex = double_occupancy_exact(psi0, N)

        # ---- spin correlation ----
        scorr   = spin_corr_exact(psi0, N)

        # ---- fidelity and Wasserstein wrt previous point ----
        if prev_psi is not None:
            F    = fidelity(prev_psi, psi0)
            dWGS = wasserstein_1d(prev_probs, probs)
        else:
            F    = 1.0
            dWGS = 0.0

        # ---- XBK pipeline (SA) ----
        H_qubit = bravyi_kitaev(H_ferm)
        n_q     = of.count_qubits(H_qubit)
        n_phys  = r * n_q    # total physical spins for r=2

        # find best sector (sweep p in {0,1,...,r})
        best_lam = np.inf
        best_spins = None
        for p in range(r + 1):
            H_ising = xbk(H_qubit, n_q, r=r, p=p)
            C_op    = counting_op(n_q, r=r, p=p)
            H_poly  = to_dict(H_ising)
            C_poly  = to_dict(C_op)
            lam, spins, n_iter = dinkelbach(H_poly, C_poly, n_phys)
            print(f"    p={p}  E0_SA = {lam:.6f}  ({n_iter} iters)")
            if lam < best_lam:
                best_lam   = lam
                best_spins = spins

        error = abs(best_lam - E0_ex)
        print(f"    E0_exact={E0_ex:.6f}  E0_SA={best_lam:.6f}  err={error:.2e}")
        print(f"    docc_fd={docc_fd:.4f}  docc_ex={docc_ex:.4f}  "
              f"spin_corr={scorr:.4f}  F={F:.4f}  dWGS={dWGS:.4f}")

        # ---- store ----
        results['uv'].append(float(uv))
        results['E0_sa'].append(float(best_lam))
        results['E0_exact'].append(float(E0_ex))
        results['error'].append(float(error))
        results['docc_fd'].append(float(docc_fd))
        results['docc_exact'].append(float(docc_ex))
        results['d2E0'].append(float(d2E0))
        results['spin_corr'].append(float(scorr))
        results['fidelity'].append(float(F))
        results['d_WGS'].append(float(dWGS))

        prev_psi   = psi0
        prev_probs = probs

    return results


# ===========================================================================
# 11.  PLOTTING
# ===========================================================================

def plot_all(all_results: dict):
    """
    all_results[N] = results dict from run_siam_sweep(N).
    Produces one figure per observable, each with N_bath curves.
    Figures saved to OUT/.
    """
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
    MARKERS = {1: 'o', 2: 's', 3: '^'}

    observables = [
        ('E0_exact',   r'$E_0$ (exact)',             'Ground-state energy'),
        ('E0_sa',      r'$E_0$ (SA)',                 'SA ground-state energy'),
        ('error',      r'$|E_0^{SA} - E_0^{exact}|$','SA absolute error'),
        ('docc_exact', r'$\langle n_\uparrow n_\downarrow \rangle$',
                                                       'Double occupancy'),
        ('d2E0',       r'$d^2 E_0/dU^2$',            'Second derivative of E0 (Kondo scale)'),
        ('spin_corr',  r'$\langle S_z^{imp} S_z^{bath} \rangle$',
                                                       'Impurity-bath spin correlation'),
        ('fidelity',   r'Fidelity $|\langle\psi(U_1)|\psi(U_2)\rangle|^2$',
                                                       'Quantum fidelity (consecutive U/V)'),
        ('d_WGS',      r'$W_1$ (Wasserstein, Hamming)',
                                                       'Wasserstein distance (consecutive U/V)'),
    ]

    for key, ylabel, title in observables:
        fig, ax = plt.subplots(figsize=(7, 4))
        for N, res in all_results.items():
            ax.plot(res['uv'], res[key],
                    color=colors[N], marker=MARKERS[N], markersize=4,
                    label=f'$N_{{bath}}={N}$', linewidth=1.5)
        ax.set_xlabel(r'$U/V$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = os.path.join(OUT, f'{key}.pdf')
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")
        plt.close(fig)


# ===========================================================================
# 12.  CSV EXPORT
# ===========================================================================

def save_csv(N: int, res: dict):
    fname = os.path.join(OUT, f'siam_N{N}.csv')
    cols  = ['uv','E0_sa','E0_exact','error',
             'docc_fd','docc_exact','d2E0','spin_corr','fidelity','d_WGS']
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(res['uv'])):
            w.writerow([res[c][i] for c in cols])
    print(f"  Saved {fname}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    all_results = {}

    for N in N_BATH_LIST:
        res = run_siam_sweep(N)
        all_results[N] = res
        save_csv(N, res)

    plot_all(all_results)
    print(f"\nAll outputs written to {OUT}/")
