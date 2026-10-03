#!/usr/bin/env python3
"""
tv1d_L2_pipeline.py
===================
Full pipeline for the 1D spinless t-V model at L = 2 sites with periodic
boundary conditions (PBC).  Self-contained: numpy, scipy, openfermion only.

NOTATION (every term is defined where it first appears):
  L        : number of lattice sites
  t        : hopping amplitude  (t > 0 favours delocalisation)
  V        : nearest-neighbour density-density interaction strength
  c†_j     : fermionic creation operator on site j
               (creates one particle on site j)
  c_j      : fermionic annihilation operator on site j
  n_j      : number operator c†_j c_j  (eigenvalues 0 or 1, spinless fermions)
  N_tot    : total particle number = ∑_j n_j
  half-fill: N_tot = L/2  (here N_tot = 1 particle in L = 2 sites)
  PBC      : periodic boundary conditions  (site L ≡ site 0)

FERMIONIC HAMILTONIAN:
  H_F = -t ∑_{<i,j>} (c†_i c_j + c†_j c_i)
       + V ∑_{<i,j>} (n_i - 1/2)(n_j - 1/2)

  where <i,j> denotes nearest-neighbour pairs.
  The shifted form  (n_i - 1/2)(n_j - 1/2)  implements particle-hole
  symmetry and cancels constant and linear-Z terms in the qubit Hamiltonian.

PIPELINE STAGES:
  1. FermionOperator  (OpenFermion)
  2. Bravyi-Kitaev (BK) transform  ->  qubit Hamiltonian (Pauli strings)
  3. Z2 symmetry tapering           ->  fewer qubits (project onto sector)
  4. XBK transformation  (r = 2)   ->  Z-only Ising polynomial H'_p(s)
  5. Counting operator C_p(s)       ->  normalises the Rayleigh quotient
  6. Rosenberg quadratisation       ->  2-local QUBO  (k -> 2 reduction)
  7. Dinkelbach iteration           ->  E_0 = min_p min_s H'_p(s) / C_p(s)
  8. Exact diagonalisation (ED)     ->  benchmark / verification

GROUND-STATE ENERGY (analytical, L = 2, PBC):
  In the 1-particle sector the BK + tapered Hamiltonian is:
    H_tap = -2t X_0 - (V/2) I
  whose eigenvalues are -V/2 ± 2t.
  Therefore E_0 = -(V/2 + 2t)  for t > 0.
"""

import numpy as np
from scipy.linalg import eigh          # real symmetric / Hermitian eigenvalues
from collections import defaultdict
from itertools import product as iproduct
import random
import math

from openfermion import FermionOperator, bravyi_kitaev, count_qubits

# ─────────────────────────────────────────────────────────────────────────────
# Physical parameters
# ─────────────────────────────────────────────────────────────────────────────

L        = 3     # number of lattice sites
t        = 1.0   # hopping amplitude  (sets the energy scale)
V        = 2.0   # nearest-neighbour interaction  (V/t = 2 is the critical point)
r        = 2     # XBK replication factor (number of copies per logical qubit)
PERIODIC = True  # close the chain into a ring

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: 1D t-V Fermionic Hamiltonian
# ─────────────────────────────────────────────────────────────────────────────

def tv1d_hamiltonian(L: int, t: float, V: float, periodic: bool = True) -> FermionOperator:
    """
    Return the second-quantised 1D spinless t-V Hamiltonian.

    H_F = -t ∑_{j} (c†_j c_{j+1} + h.c.)
         + V ∑_{j} (n_j - 1/2)(n_{j+1} - 1/2)

    where j runs from 0 to L-1 and index arithmetic is mod L if periodic.

    The interaction is written in particle-hole symmetric form:
      (n_i - 1/2)(n_j - 1/2) = n_i n_j - (n_i + n_j)/2 + 1/4

    This cancels constant and single-Z terms in the BK representation,
    leaving only ZZ couplings of strength V/4.

    Args:
        L       : number of sites
        t       : hopping amplitude
        V       : interaction strength
        periodic: if True, include the bond (L-1, 0) to close the ring

    Returns:
        H : FermionOperator
    """
    H = FermionOperator()
    n_bonds = L if periodic else L - 1      # number of bonds

    for j in range(n_bonds):
        a = j                               # left site of bond
        b = (j + 1) % L                    # right site of bond  (mod L for PBC)

        # ── Hopping: -t (c†_a c_b + c†_b c_a) ──────────────────────────────
        H += FermionOperator(f'{a}^ {b}', -t)   # c†_a c_b
        H += FermionOperator(f'{b}^ {a}', -t)   # c†_b c_a  = Hermitian conjugate

        # ── Interaction: V (n_a - 1/2)(n_b - 1/2) ───────────────────────────
        #   = V n_a n_b  - (V/2) n_a  - (V/2) n_b  + V/4
        #
        # n_a n_b in normal order: c†_a c†_b c_b c_a  (for a < b, up to sign)
        # OpenFermion applies normal ordering automatically.
        if a < b:
            H += FermionOperator(f'{a}^ {b}^ {b} {a}', V)     # V n_a n_b
        else:
            H += FermionOperator(f'{b}^ {a}^ {a} {b}', V)
        H += FermionOperator(f'{a}^ {a}', -V / 2.0)            # -(V/2) n_a
        H += FermionOperator(f'{b}^ {b}', -V / 2.0)            # -(V/2) n_b
        H += FermionOperator('', V / 4.0)                       # +V/4

    return H


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: BK Transform -> bk_terms
# ─────────────────────────────────────────────────────────────────────────────

def bk_to_terms(H_fermion: FermionOperator):
    """
    Apply the Bravyi-Kitaev (BK) transformation to a fermionic Hamiltonian.

    BK encodes occupation / parity information in a Fenwick-tree (binary
    indexed tree) structure, reducing typical operator weights from O(L)
    (Jordan-Wigner) to O(log L).

    Returns:
        bk_terms   : list of (float coeff, dict {qubit_index: pauli_char})
                     where pauli_char ∈ {'X', 'Y', 'Z'}; identity = absent key.
        n_qubits   : number of qubits (equals L for spinless fermions)
    """
    H_bk     = bravyi_kitaev(H_fermion)      # OpenFermion BK transform
    n_qubits = count_qubits(H_bk)

    bk_terms = []
    for pauli_tuple, coeff in H_bk.terms.items():
        c = float(np.real(coeff))
        if abs(c) > 1e-14:
            # pauli_tuple = ((q0, 'X'), (q1, 'Z'), ...) or () for identity
            pd = {idx: op for idx, op in pauli_tuple}
            bk_terms.append((c, pd))

    return bk_terms, n_qubits


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Z2 Symmetry Tapering
# ─────────────────────────────────────────────────────────────────────────────
#
# A 'Z2 symmetry' of H is a Pauli operator τ with eigenvalues ±1 such that
# [τ, H] = 0.  We find ALL such operators via the symplectic representation.
#
# SYMPLECTIC ENCODING:
#   A Pauli string P = σ_{q_0} ⊗ σ_{q_1} ⊗ … on n qubits is encoded as
#   a binary vector v = (x_0, …, x_{n-1} | z_0, …, z_{n-1}) ∈ F_2^{2n}
#   with the encoding rule:
#     I  →  (0 | 0),   X  →  (1 | 0),   Z  →  (0 | 1),   Y  →  (1 | 1)
#
# COMMUTATIVITY:
#   Two Pauli strings P_a, P_b commute iff their symplectic inner product
#     v_a · J · v_b^T = 0  (mod 2)
#   where J = [[0_n, I_n], [I_n, 0_n]] ∈ F_2^{2n × 2n}.
#
# Z2 SYMMETRIES:
#   Stack all m Pauli terms of H into E ∈ F_2^{m × 2n}.
#   A symmetry generator corresponds to a vector u ∈ F_2^{2n} with
#     E · J · u^T = 0  (mod 2)
#   i.e., u is in the null space of (E J)^T over the binary field F_2 = {0,1}.
#
# CLIFFORD REDUCTION:
#   Given a symmetry τ = Z_{q_0} Z_{q_1} … Z_{q_{k-1}} (pure-Z type):
#   Apply CNOT_{q_j → q_{k-1}} for j = 0, …, k-2.
#   Each CNOT_{c→t} acts on symplectic vectors as:
#     x_t  ←  x_t ⊕ x_c   (X on target picks up X from control)
#     z_c  ←  z_c ⊕ z_t   (Z on control picks up Z from target)
#   After all CNOTs: τ → Z_{q_{k-1}}.
#
# SECTOR PROJECTION:
#   Set Z_{q_{k-1}} = λ (eigenvalue ±1) to fix the symmetry sector.
#   Every term in H then has z_{q_{k-1}} = 0 or 1, and the term picks up
#   a factor of λ^{z_{q_{k-1}}}.  Qubit q_{k-1} is removed.
# ─────────────────────────────────────────────────────────────────────────────

def _pauli_to_symp(pd: dict, n: int) -> np.ndarray:
    """
    Convert a Pauli string {qubit: char} to a binary symplectic vector.

    v = (x_0, …, x_{n-1}, z_0, …, z_{n-1})  ∈  {0,1}^{2n}
    """
    v = np.zeros(2 * n, dtype=int)
    for q, op in pd.items():
        if op in ('X', 'Y'):
            v[q] = 1          # x-component
        if op in ('Z', 'Y'):
            v[n + q] = 1      # z-component
    return v


def _symp_to_pauli(v: np.ndarray, n: int) -> dict:
    """
    Convert a binary symplectic vector back to a Pauli string dict.

    Returns {qubit: char} with char ∈ {'X', 'Y', 'Z'}.
    Qubits with identity are omitted.
    """
    pd = {}
    for q in range(n):
        xb, zb = int(v[q]), int(v[n + q])
        if   (xb, zb) == (1, 0):  pd[q] = 'X'
        elif (xb, zb) == (0, 1):  pd[q] = 'Z'
        elif (xb, zb) == (1, 1):  pd[q] = 'Y'
        # (0,0) → identity, omit
    return pd


def _gf2_rref(M: np.ndarray):
    """
    Reduced row echelon form (RREF) of an integer matrix over GF(2).

    All arithmetic is mod 2 (binary field).

    Args:
        M : 2D numpy array of 0s and 1s, shape (m, n)

    Returns:
        rref       : RREF matrix (same shape as M)
        pivot_cols : list of pivot column indices (sorted)
    """
    M          = M.copy().astype(int) % 2
    nrows, ncols = M.shape
    pivot_cols = []
    row        = 0

    for col in range(ncols):
        # Find the first row at or below 'row' with a 1 in this column
        pivot = None
        for r in range(row, nrows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue                      # no pivot in this column

        M[[row, pivot]] = M[[pivot, row]] # swap rows

        # Eliminate all other 1s in this column
        for r in range(nrows):
            if r != row and M[r, col] == 1:
                M[r] = (M[r] + M[row]) % 2

        pivot_cols.append(col)
        row += 1

    return M, pivot_cols


def _gf2_nullspace(M: np.ndarray):
    """
    Compute the null space of M over GF(2).

    Returns a list of binary vectors {0,1}^n; each is a null-space basis vector.
    The null vector u satisfies  M · u^T = 0  (mod 2).
    """
    nrows, ncols = M.shape
    rref, pivot_cols = _gf2_rref(M)
    pivot_set  = set(pivot_cols)
    free_cols  = [c for c in range(ncols) if c not in pivot_set]
    pivot_row  = {pc: i for i, pc in enumerate(pivot_cols)}  # col -> RREF row

    null_vecs = []
    for fc in free_cols:
        v = np.zeros(ncols, dtype=int)
        v[fc] = 1
        # For each pivot column pc (at RREF row i), set v[pc] = RREF[i, fc]
        for pc in pivot_cols:
            v[pc] = int(rref[pivot_row[pc], fc])
        null_vecs.append(v)

    return null_vecs


def _cnot_update_symp(v: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    """
    Apply one CNOT_{control → target} to a symplectic vector v ∈ {0,1}^{2n}.

    Transformation rules over F_2:
      x_target  ←  x_target ⊕ x_control
      z_control ←  z_control ⊕ z_target
    Other components are unchanged.
    """
    v = v.copy()
    v[target]      = (v[target]      + v[control]) % 2      # x-block
    v[n + control] = (v[n + control] + v[n + target]) % 2   # z-block
    return v


def _find_pure_z_symmetries(terms_symp: list, n: int) -> list:
    """
    Find all pure-Z Z2 symmetries of a Hamiltonian given in symplectic form.

    A Z2 symmetry is a vector u ∈ F_2^{2n} in the null space of (E·J)^T,
    where E is the symplectic matrix of all Pauli terms and J is the
    symplectic form [[0,I],[I,0]].

    Returns only the null vectors that correspond to PURE-Z operators
    (x-block = 0), which are the ones we can taper without Hadamard gates.
    """
    if not terms_symp:
        return []

    rows = [v for _, v in terms_symp]
    E    = np.array(rows, dtype=int)                     # shape (m, 2n)

    J    = np.zeros((2 * n, 2 * n), dtype=int)
    J[:n, n:] = np.eye(n, dtype=int)
    J[n:, :n] = np.eye(n, dtype=int)

    EJ   = (E @ J) % 2                                   # shape (m, 2n)

    # Right null space of EJ: vectors u ∈ F_2^{2n} with EJ · u = 0.
    # _gf2_nullspace(M) returns vectors v with M · v = 0, so we pass EJ directly.
    null_vecs = _gf2_nullspace(EJ)

    # Keep only pure-Z generators (x-block = 0, non-trivial z-block)
    pure_z = []
    for u in null_vecs:
        x_part = u[:n]
        z_part = u[n:]
        if np.all(x_part == 0) and np.any(z_part == 1):
            pure_z.append(u)
    return pure_z


def z2_taper(bk_terms, n_qubits: int, sector_lambda: float = -1.0):
    """
    Find all Z2 symmetries of the BK Hamiltonian, then remove ONE symmetry
    per iteration by projecting onto the eigenvalue sector λ.

    Only PURE-Z symmetries (τ = product of Z operators) are handled here;
    the t-V BK Hamiltonian's global-parity symmetry is always pure-Z.

    PROCEDURE (one symmetry per iteration, repeated until no pure-Z symmetry remains):
      1. Build symplectic matrix E from the current Pauli terms.
      2. Compute E · J  (J = symplectic form [[0,I],[I,0]]).
      3. Find null space of (E · J)^T over F_2.
         Keep only pure-Z null vectors (x-block = 0).
      4. Take the first pure-Z symmetry u, corresponding to
           τ = ⊗_{q: u[n+q]=1} Z_q
      5. CNOT fan-in: apply CNOT_{q → target} for all Z-support qubits q
         except the last (target), mapping τ → Z_{target}.
      6. Apply the same CNOTs to all Hamiltonian terms (symplectic update).
      7. Fix Z_{target} = λ (sector eigenvalue), remove qubit 'target'.
      8. Relabel remaining qubits and repeat.

    Args:
        bk_terms      : list of (coeff, {qubit: pauli_char})
        n_qubits      : initial number of qubits
        sector_lambda : eigenvalue to project onto (+1 = even parity, -1 = odd)

    Returns:
        tapered_terms : list of (coeff, {qubit: pauli_char}) on fewer qubits
        n_tap         : number of qubits after tapering
        removed_qubits: list of qubit indices that were removed (in original numbering)
    """
    lam         = float(sector_lambda)
    n_current   = n_qubits
    removed     = []

    # Work in symplectic form throughout
    terms_symp = [(c, _pauli_to_symp(pd, n_current)) for c, pd in bk_terms]

    while True:
        # ── Find all pure-Z symmetries of the current (smaller) Hamiltonian ──
        sym_vecs = _find_pure_z_symmetries(terms_symp, n_current)
        if not sym_vecs:
            break                           # no more taperable symmetries

        # ── Take the first pure-Z symmetry ───────────────────────────────────
        u        = sym_vecs[0]
        z_support = [q for q in range(n_current) if u[n_current + q] == 1]
        if not z_support:
            break

        # ── CNOT fan-in: τ → Z_{target_q} ───────────────────────────────────
        target_q  = z_support[-1]
        cnot_list = [(q, target_q) for q in z_support[:-1]]

        new_terms = []
        for c, v in terms_symp:
            v_new = v.copy()
            for ctrl, tgt in cnot_list:
                v_new = _cnot_update_symp(v_new, n_current, ctrl, tgt)
            new_terms.append((c, v_new))
        terms_symp = new_terms

        # ── Project: Z_{target_q} → λ, remove that qubit ────────────────────
        projected = []
        for c, v in terms_symp:
            x_q = int(v[target_q])
            z_q = int(v[n_current + target_q])

            # After the CNOT fan-in, target_q must appear only as I or Z.
            assert x_q == 0, (
                f"Bug: qubit {target_q} still has X or Y component after CNOT fan-in. "
                f"Symplectic vector: {v}"
            )

            # Multiply by λ^{z_q}: +1 if Z absent, λ if Z present
            factor = lam ** z_q

            # Remove qubit target_q: compress the symplectic vector
            x_part = np.concatenate([v[:target_q],
                                      v[target_q + 1: n_current]])
            z_part = np.concatenate([v[n_current: n_current + target_q],
                                      v[n_current + target_q + 1:]])
            v_new  = np.concatenate([x_part, z_part])

            new_c = c * factor
            if abs(new_c) > 1e-14:
                projected.append((new_c, v_new))

        removed.append(target_q)
        n_current -= 1
        terms_symp = projected

        # Merge duplicate Pauli terms (same symplectic pattern)
        merged = defaultdict(float)
        for c, v in terms_symp:
            key = tuple(v.tolist())
            merged[key] += c
        terms_symp = [(c, np.array(list(k), dtype=int))
                      for k, c in merged.items() if abs(c) > 1e-14]

    # ── Convert back to dict representation on n_current qubits ─────────────
    tapered_terms = []
    for c, v in terms_symp:
        pd = _symp_to_pauli(v, n_current)
        tapered_terms.append((c, pd))

    return tapered_terms, n_current, removed


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Ising Polynomial Arithmetic
# ─────────────────────────────────────────────────────────────────────────────
#
# After XBK, the Hamiltonian H'_p and counting operator C_p are both
# polynomials in Ising spin variables s_ℓ ∈ {-1, +1}.
#
# We represent a polynomial as:
#   dict { frozenset(spin_indices) : float_coefficient }
# For example:
#   {frozenset(): 3.0, frozenset([0,1]): -1.5}
# represents  3.0 - 1.5 s_0 s_1.
#
# Key algebraic fact:  s_ℓ^2 = 1  for all ℓ.
# Therefore the product of two monomials (as sets of spin indices) is the
# symmetric difference of their index sets:
#   (∏_{i ∈ A} s_i)(∏_{j ∈ B} s_j) = ∏_{k ∈ A Δ B} s_k
# ─────────────────────────────────────────────────────────────────────────────

def _poly_mul(p1: dict, p2: dict) -> dict:
    """Multiply two Ising polynomials (symmetric-difference algebra)."""
    out = defaultdict(complex)
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            out[s1.symmetric_difference(s2)] += c1 * c2
    return dict(out)


def _poly_clean(poly: dict, tol: float = 1e-12) -> dict:
    """Drop near-zero terms and take real part."""
    return {m: v.real for m, v in poly.items() if abs(v.real) > tol}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: XBK Transformation  (r = 2)
# ─────────────────────────────────────────────────────────────────────────────
#
# The Xia-Bian-Kais (XBK) method embeds a qubit Hamiltonian (with X, Y, Z)
# into a larger space where every Pauli factor is replaced by a Z-bilinear
# across two copies of each qubit.
#
# PHYSICAL QUBIT LAYOUT (r = 2):
#   Logical qubit i, copy j (j ∈ {1, 2})  →  physical index  2i + (j - 1)
#   E.g., for m = 1 logical qubit:
#     physical 0 = (logical 0, copy 1)  = reference copy
#     physical 1 = (logical 0, copy 2)  = replica copy
#
# For a sector p with sign table S_p(i, j):
#   Convention S_p(i, 1) ≡ +1  (reference copy always +1)
#              S_p(i, 2) ∈ {+1, -1}  (replica sign = sector_signs[i])
#
# XBK SUBSTITUTION RULES (r = 2, for copy pair j = 1, k = 2):
#
#   (A) Same copy  (j = k):
#     X_i  →  0              (X / Y vanish in same-copy terms)
#     Y_i  →  0
#     Z_i  →  s_p * Z_{i,j}  (with physical index phys(i, j))
#     I_i  →  1
#
#   (B) Cross copy (j ≠ k, specifically j=1, k=2):
#     X_i  →  s_p * (1 - Z_{i,1} Z_{i,2}) / 2
#     Y_i  →  i * s_p * (Z_{i,2} - Z_{i,1}) / 2
#     Z_i  →  s_p * (Z_{i,1} + Z_{i,2}) / 2
#     I_i  →  (1 + Z_{i,1} Z_{i,2}) / 2
#
#   where  s_p = S_p(i, 1) * S_p(i, 2) = 1 * sector_signs[i] = sector_signs[i]
#   and Z_{i,j} denotes the physical spin s_{phys(i,j)} ∈ {-1, +1}.
#
# The full XBK Hamiltonian is:
#   H'_p(s) = ∑_{j,k=1}^{2} H'_{j,k}(s)
# summing over all (reference, replica) copy pairs.
# ─────────────────────────────────────────────────────────────────────────────

def _phys(logical: int, copy: int) -> int:
    """Physical spin index for (logical qubit, copy index)."""
    return 2 * logical + (copy - 1)


def _xbk_single(pauli: str, qi: int, j: int, k: int, sp: float) -> dict:
    """
    XBK substitution for a single-qubit Pauli on logical qubit qi,
    for the (copy j, copy k) pair, with sector sign sp.

    Returns a partial Ising polynomial (dict).
    """
    pj = _phys(qi, j)
    pk = _phys(qi, k)

    if j == k:                            # same-copy contribution
        if pauli in ('X', 'Y'):
            return {}                     # X, Y vanish in same-copy block
        if pauli == 'Z':
            return {frozenset([pj]): complex(sp)}
        # Identity
        return {frozenset(): 1.0 + 0j}

    # Cross-copy  (j ≠ k)
    if pauli == 'X':
        return {frozenset():       sp * 0.5 + 0j,
                frozenset([pj, pk]): -sp * 0.5 + 0j}
    if pauli == 'Y':
        return {frozenset([pk]):  1j * sp * 0.5,
                frozenset([pj]): -1j * sp * 0.5}
    if pauli == 'Z':
        return {frozenset([pj]):  sp * 0.5 + 0j,
                frozenset([pk]):  sp * 0.5 + 0j}
    # Identity
    return {frozenset():       0.5 + 0j,
            frozenset([pj, pk]): 0.5 + 0j}


def build_hprime(bk_terms, n_qubits: int, sector_signs: list) -> dict:
    """
    Build H'_p as a Z-only Ising polynomial for a given XBK sector.

    H'_p(s) = ∑_{j=1}^{r} ∑_{k=1}^{r} H'_{j,k}(s)

    For r = 2:  four (j, k) pairs: (1,1), (1,2), (2,1), (2,2).

    Args:
        bk_terms     : list of (coeff, {qubit: pauli_char})
        n_qubits     : number of logical qubits (= m after tapering)
        sector_signs : list of length n_qubits; sector_signs[i] = S_p(i, 2)

    Returns:
        ising_poly : dict { frozenset(spin_indices) : float }
    """
    H_prime = defaultdict(complex)

    for j in range(1, r + 1):
        for k in range(1, r + 1):
            for coeff, pd in bk_terms:
                # Start with the scalar 1 (identity monomial)
                term = {frozenset(): 1.0 + 0j}

                # Multiply in the XBK factor for each logical qubit
                for qi in range(n_qubits):
                    pauli = pd.get(qi, 'I')       # 'I' if qubit absent
                    sp    = sector_signs[qi]       # sector sign for this qubit
                    fac   = _xbk_single(pauli, qi, j, k, sp)
                    if not fac:
                        term = {}
                        break
                    term = _poly_mul(term, fac)

                for mono, val in term.items():
                    H_prime[mono] += coeff * val

    return _poly_clean(dict(H_prime))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Counting Operator
# ─────────────────────────────────────────────────────────────────────────────
#
# Because XBK replicates logical qubits into r copies, each logical basis
# state |b_0 … b_{m-1}⟩ appears r times in the physical space.
# The counting operator C_p(s) counts how many times each logical state
# appears, normalising the Rayleigh quotient:
#
#   E_0 = min_s  H'_p(s) / C_p(s)
#
# DEFINITION:
#   C_p(s) = ∑_{b ∈ {0,1}^m} [ ∑_{j=1}^{r} P_{j,b}(s) ]^2
#
#   where  P_{j,b}(s) = ∏_{k=0}^{m-1} [ (1 + (-1)^{b_k} s_{phys(k,j)}) / 2 ]
#
#   P_{j,b} is 1 if copy j agrees with bit-string b, else 0.
#   Summing (∑_j P_{j,b})^2 over all b counts replica coincidences.
#
# For r = 2, m = 1:  C_p(s) = 3 + s_0 s_1  (verified analytically above).
# ─────────────────────────────────────────────────────────────────────────────

def build_counting_op(n_qubits: int) -> dict:
    """
    Build the counting operator C_p as an Ising polynomial.

    C_p(s) = ∑_{b ∈ {0,1}^m} [ ∑_{j=1}^{r} P_{j,b}(s) ]^2

    Works for arbitrary n_qubits (= m) and r = 2.

    Returns:
        ising_poly : dict { frozenset(spin_indices) : float }
    """
    C_total = defaultdict(complex)

    for basis in iproduct([0, 1], repeat=n_qubits):
        # ∑_{j=1}^{r} P_{j,b}(s)  as an Ising polynomial
        inner = defaultdict(complex)
        for copy_j in range(1, r + 1):
            # P_{j,b}(s) = ∏_k (1 + (-1)^{b_k} s_{phys(k,j)}) / 2
            p = {frozenset(): 1.0 + 0j}
            for k, b_k in enumerate(basis):
                ph     = _phys(k, copy_j)
                sign_k = float((-1) ** b_k)     # +1 if b_k=0, -1 if b_k=1
                factor = {frozenset():    0.5 + 0j,
                          frozenset([ph]): sign_k * 0.5 + 0j}
                p = _poly_mul(p, factor)
            for m, v in p.items():
                inner[m] += v

        # Add (∑_j P_{j,b})^2 to C_total
        inner_dict = dict(inner)
        for m, v in _poly_mul(inner_dict, inner_dict).items():
            C_total[m] += v

    return _poly_clean(dict(C_total))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Rosenberg Quadratisation  (k-local → 2-local)
# ─────────────────────────────────────────────────────────────────────────────
#
# XBK may produce Ising terms s_{a} s_{b} s_{c} … with k > 2 spins.
# Quantum annealers (D-Wave) only support 2-local couplings:
#   H = ∑_i h_i s_i + ∑_{i<j} J_{ij} s_i s_j   (Ising form)
#
# ROSENBERG SUBSTITUTION (binary variables x = (1-s)/2 ∈ {0,1}):
#   For a 3-local term  ±x_a x_b x_c:
#   Introduce an auxiliary variable x_w (a new binary variable) and
#   add a penalty term  M (x_a x_b - 2 x_w x_a - 2 x_w x_b + 3 x_w) ≥ 0.
#
#   The penalty forces x_w = x_a x_b at the minimum (for M large enough).
#   Replacing x_a x_b → x_w reduces the term from 3-local to 2-local.
#
# PENALTY SCALE:
#   M must satisfy  M > |coefficient|  for the substitution to be valid.
#   We use M = 12 * max|h| as in pipeline_base.py.
#
# Note: for L = 2 with r = 2 and 1 tapered qubit, H'_p is already 2-local
# (at most s_0 s_1).  The Rosenberg step is a no-op here, but is implemented
# for generality (needed for L ≥ 4).
# ─────────────────────────────────────────────────────────────────────────────

def _ising_to_binary(ising: dict) -> dict:
    """
    Convert an Ising polynomial in s_ℓ ∈ {±1}
    to a QUBO/HUBO polynomial in x_ℓ ∈ {0, 1},
    using the substitution  s_ℓ = 1 - 2 x_ℓ.

    Each monomial ∏_{ℓ ∈ S} s_ℓ expands to:
      ∏_{ℓ ∈ S} (1 - 2x_ℓ) = ∑_{T ⊆ S} (-2)^|T| ∏_{ℓ ∈ T} x_ℓ
    """
    out = defaultdict(float)
    for S, c in ising.items():
        sl = sorted(S)
        k  = len(sl)
        for tb in range(1 << k):           # iterate over 2^k subsets T ⊆ S
            T    = frozenset(sl[m] for m in range(k) if (tb >> m) & 1)
            out[T] += c * ((-2.0) ** len(T))
    return dict(out)


def _binary_to_ising(binary: dict) -> dict:
    """
    Convert a binary polynomial in x_ℓ ∈ {0, 1}
    back to an Ising polynomial in s_ℓ ∈ {±1},
    using  x_ℓ = (1 - s_ℓ) / 2.

    Each monomial ∏_{ℓ ∈ T} x_ℓ expands to:
      ∏_{ℓ ∈ T} (1 - s_ℓ)/2 = ∑_{U ⊆ T} (-1)^|U| ∏_{ℓ ∈ U} s_ℓ / 2^|T|
    """
    out = defaultdict(float)
    for T, c in binary.items():
        tl = sorted(T)
        k  = len(tl)
        for ub in range(1 << k):
            U    = frozenset(tl[m] for m in range(k) if (ub >> m) & 1)
            out[U] += c * ((-1.0) ** len(U)) / (2.0 ** k)
    return dict(out)


def rosenberg_reduce(binary_poly: dict, M: float, next_id: int):
    """
    Iteratively reduce a binary polynomial to degree ≤ 2
    using the Rosenberg substitution.

    For each degree-k (k ≥ 3) monomial {a, b, c, …}:
      Introduce auxiliary variable x_w  (index next_id).
      Replace:  c * x_a x_b x_c …
        →  c * x_w x_c …                 (degree reduced by 1)
           + M * x_a x_b                 (penalty forcing x_w = x_a x_b)
           - 2M * x_w x_a
           - 2M * x_w x_b
           + 3M * x_w
      Repeat until all monomials have degree ≤ 2.

    Args:
        binary_poly : polynomial in x ∈ {0,1} variables
        M           : penalty coefficient  (should be ≥ max|coeff| * 12)
        next_id     : next available auxiliary variable index

    Returns:
        (quadratic_poly, next_id_after)
    """
    poly    = defaultdict(float, binary_poly)
    changed = True

    while changed:
        changed = False
        for mono in list(poly.keys()):
            if len(mono) < 3:
                continue
            c = poly.pop(mono, 0.0)
            if abs(c) < 1e-14:
                continue

            sl           = sorted(mono)
            a, b, rest   = sl[0], sl[1], frozenset(sl[2:])
            w            = next_id;   next_id += 1   # new auxiliary variable

            # Rosenberg substitution for monomial x_a x_b:
            poly[frozenset([w]) | rest] += c           # replace x_a x_b → x_w
            poly[frozenset([a, b])]     += M            # +M x_a x_b
            poly[frozenset([w, a])]     += -2.0 * M    # -2M x_w x_a
            poly[frozenset([w, b])]     += -2.0 * M    # -2M x_w x_b
            poly[frozenset([w])]        += 3.0 * M      # +3M x_w

            changed = True
            break  # restart after each substitution

    return dict(poly), next_id


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Simulated Annealing (SA)
# ─────────────────────────────────────────────────────────────────────────────
#
# Simulated Annealing (SA) is a stochastic optimisation algorithm that finds
# the minimum of a cost function by accepting 'uphill' moves with decreasing
# probability as a 'temperature' T is cooled.
#
# ALGORITHM (Metropolis-Hastings on Ising spins):
#   1. Initialise spins s_ℓ ∈ {±1} randomly.
#   2. For each sweep:
#      a. Propose a random spin flip  s_i → -s_i.
#      b. Compute energy change ΔE = E(s') - E(s).
#      c. Accept if ΔE < 0, else accept with probability exp(-ΔE / T).
#      d. Update best solution if new energy is lower.
#   3. Geometrically decrease T from T_start to T_end.
#
# ENERGY EVALUATION for Ising polynomial:
#   E(s) = ∑_{S ⊆ [n]} c_S ∏_{ℓ ∈ S} s_ℓ
# ─────────────────────────────────────────────────────────────────────────────

def _eval_poly(poly: dict, spins: dict) -> float:
    """Evaluate an Ising polynomial at a spin configuration."""
    E = 0.0
    for mono, c in poly.items():
        p = 1.0
        for i in mono:
            p *= spins[i]
        E += c * p
    return E


def _delta_e(poly: dict, spins: dict, flip: int) -> float:
    """
    Compute the energy change ΔE when spin 'flip' is flipped.

    ΔE = -2 * s_{flip} * (effective field from all monomials involving flip)
    """
    si    = spins[flip]
    field = 0.0
    for mono, c in poly.items():
        if flip not in mono:
            continue
        p = 1.0
        for j in mono:
            if j != flip:
                p *= spins[j]
        field += c * p
    return -2.0 * si * field


def sa_solve(ising_poly: dict, n_spins: int, n_sweeps: int = 8000,
             T_start: float = 5.0, T_end: float = 1e-4,
             seed: int = 42) -> tuple:
    """
    Minimise an Ising polynomial by Simulated Annealing.

    Args:
        ising_poly : dict { frozenset(spin_indices) : float }
        n_spins    : total number of spin variables (0 … n_spins-1)
        n_sweeps   : number of SA sweeps (each sweep = n_spins flip attempts)
        T_start    : initial temperature
        T_end      : final temperature
        seed       : random seed for reproducibility

    Returns:
        (best_spins, best_energy)
        best_spins  : dict {spin_index: ±1}
        best_energy : float
    """
    rng   = random.Random(seed)
    spins = {i: rng.choice([-1, 1]) for i in range(n_spins)}
    E     = _eval_poly(ising_poly, spins)

    best_spins = dict(spins)
    best_E     = E

    # Geometric cooling schedule
    alpha = (T_end / T_start) ** (1.0 / max(n_sweeps, 1))
    T     = T_start

    for _ in range(n_sweeps):
        for __ in range(n_spins):
            i  = rng.randint(0, n_spins - 1)
            dE = _delta_e(ising_poly, spins, i)
            if dE < 0 or rng.random() < math.exp(-dE / T):
                spins[i] = -spins[i]
                E        += dE
                if E < best_E:
                    best_E     = E
                    best_spins = dict(spins)
        T *= alpha

    return best_spins, best_E


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Dinkelbach Loop
# ─────────────────────────────────────────────────────────────────────────────
#
# We want the ground-state energy:
#   E_0 = min_{s ∈ {±1}^{n_phys}} H'_p(s) / C_p(s)
#
# This is a fractional programming problem.  Dinkelbach's algorithm converts
# it into a sequence of unconstrained minimisations:
#
#   At iteration k:
#     s*_k = argmin_{s} [ H'_p(s) - λ_k C_p(s) ]
#     λ_{k+1} = H'_p(s*_k) / C_p(s*_k)
#
# λ_k converges monotonically to E_0 from above.  The stopping criterion is
# |λ_{k+1} - λ_k| < ε.
#
# CRITICAL (from PROJECT_MEMORY.md):
#   SA runs on the QUADRATISED (augmented) Hamiltonian H'_quad - λ C_quad
#   which includes auxiliary variables for Rosenberg reduction.
#   But the Rayleigh quotient λ_{k+1} = H'(s*)/C(s*) MUST be evaluated on
#   the PRE-QUADRATISATION polynomials using ONLY physical spin indices
#   (0 … n_phys - 1).  Using auxiliary spins in the ratio gives wrong results.
# ─────────────────────────────────────────────────────────────────────────────

def dinkelbach_solve(H_pre: dict, C_pre: dict,
                     H_quad: dict, C_quad: dict,
                     n_phys: int, n_aug: int,
                     n_sweeps: int = 8000, base_seed: int = 0,
                     max_iter: int = 30, tol: float = 1e-7) -> float:
    """
    Dinkelbach's algorithm to minimise H'_p(s)/C_p(s).

    Args:
        H_pre    : H'_p as pre-quadratisation Ising polynomial (physical spins only)
        C_pre    : C_p as pre-quadratisation Ising polynomial (physical spins only)
        H_quad   : H'_p after Rosenberg reduction (includes auxiliaries)
        C_quad   : C_p after Rosenberg reduction (includes auxiliaries)
        n_phys   : number of physical spins (0 … n_phys - 1)
        n_aug    : total spins including auxiliaries (= next_id after quadratise)
        n_sweeps : SA sweeps per Dinkelbach iteration
        base_seed: random seed (incremented each iteration)
        max_iter : maximum Dinkelbach iterations
        tol      : convergence threshold |λ_{k+1} - λ_k| < tol

    Returns:
        E_0 : estimated ground-state energy  (float)
    """
    # Estimate penalty scale from the largest coefficient in H'
    M_scale   = max((abs(c) for c in H_pre.values()), default=1.0)
    T0        = max(5.0 * M_scale, 5.0)   # initial SA temperature

    lam       = 0.0          # initial Dinkelbach parameter (arbitrary start)
    best_E0   = float('inf')

    for it in range(max_iter):
        # ── Build shifted cost: H_quad - λ * C_quad ─────────────────────────
        shifted = defaultdict(float, H_quad)
        for mono, c in C_quad.items():
            shifted[mono] -= lam * c

        # ── SA on the shifted quadratic cost ─────────────────────────────────
        best_spins, _ = sa_solve(dict(shifted), n_aug,
                                  n_sweeps=n_sweeps,
                                  T_start=T0, T_end=1e-4,
                                  seed=base_seed + it * 997)

        # ── Rayleigh quotient on PHYSICAL spins ONLY (pre-quad polynomials) ──
        # Extract only the physical spins from the SA solution
        s_phys = {i: best_spins[i] for i in range(n_phys)}

        H_val = _eval_poly(H_pre, s_phys)
        C_val = _eval_poly(C_pre, s_phys)

        if abs(C_val) < 1e-8:
            continue     # degenerate configuration; skip

        new_lam = H_val / C_val
        if new_lam < best_E0:
            best_E0 = new_lam

        if abs(new_lam - lam) < tol:
            lam = new_lam
            break       # converged

        lam = new_lam

    return best_E0


# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Exact Diagonalisation (Benchmark)
# ─────────────────────────────────────────────────────────────────────────────
#
# Build the many-body Hamiltonian matrix in the Fock space {|n_0 n_1 … n_{L-1}⟩}
# and diagonalise it exactly.
#
# For L = 2 the Fock space has dimension 2^L = 4:
#   |00⟩  (no particles),  |10⟩  (particle on site 0),
#   |01⟩  (particle on site 1),  |11⟩  (both sites occupied)
#
# For half-filling (1 particle) we restrict to {|10⟩, |01⟩} and find E_0.
# ─────────────────────────────────────────────────────────────────────────────

def exact_diagonalise(L: int, t: float, V: float, periodic: bool = True) -> float:
    """
    Exact ground-state energy of the 1D t-V model by diagonalising the
    many-body Hamiltonian in the full Fock basis (dimension 2^L).

    Args:
        L        : number of sites
        t        : hopping amplitude
        V        : interaction strength
        periodic : use PBC if True

    Returns:
        E_0 : ground-state energy (lowest eigenvalue)
    """
    dim  = 2 ** L
    H_mat = np.zeros((dim, dim), dtype=float)

    for state in range(dim):
        # Diagonal term: interaction V (n_j - 1/2)(n_{j+1} - 1/2)
        n_bonds = L if periodic else L - 1
        for j in range(n_bonds):
            a = j
            b = (j + 1) % L
            na = (state >> a) & 1         # occupation of site a  (0 or 1)
            nb = (state >> b) & 1
            H_mat[state, state] += V * (na - 0.5) * (nb - 0.5)

        # Off-diagonal: hopping -t (c†_a c_b + h.c.)
        n_bonds = L if periodic else L - 1
        for j in range(n_bonds):
            a = j
            b = (j + 1) % L

            # c†_a c_b: destroy at b, create at a
            na = (state >> a) & 1
            nb = (state >> b) & 1
            if nb == 1 and na == 0:       # can hop b → a
                new_state = state ^ (1 << b) ^ (1 << a)
                # Jordan-Wigner phase: count occupied sites between a and b
                lo, hi    = min(a, b), max(a, b)
                jw_phase  = (-1) ** bin((state >> (lo + 1)) & ((1 << (hi - lo - 1)) - 1)).count('1')
                H_mat[new_state, state] += -t * jw_phase

            # c†_b c_a: destroy at a, create at b  (h.c.)
            if na == 1 and nb == 0:       # can hop a → b
                new_state = state ^ (1 << a) ^ (1 << b)
                lo, hi    = min(a, b), max(a, b)
                jw_phase  = (-1) ** bin((state >> (lo + 1)) & ((1 << (hi - lo - 1)) - 1)).count('1')
                H_mat[new_state, state] += -t * jw_phase

    eigenvalues = eigh(H_mat, eigvals_only=True)
    return float(np.min(eigenvalues))


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(L: int, t: float, V: float, periodic: bool = True,
                 taper_lambda: float = -1.0,
                 n_sweeps: int = 12000, penalty_scale: float = 12.0,
                 seed: int = 0) -> dict:
    """
    Run the complete pipeline for given parameters.

    Returns a dict with intermediate results and final E_0 estimates.
    """
    results = {}

    # ── Stage 1: Fermionic Hamiltonian ───────────────────────────────────────
    print("\n── Stage 1: 1D t-V Fermionic Hamiltonian ──────────────────────────")
    H_F = tv1d_hamiltonian(L, t, V, periodic)
    print(f"  FermionOperator:\n    {H_F}")

    # ── Stage 2: BK Transform ────────────────────────────────────────────────
    print("\n── Stage 2: Bravyi-Kitaev Transform ───────────────────────────────")
    bk_terms, n_bk = bk_to_terms(H_F)
    print(f"  n_qubits (after BK) = {n_bk}")
    print("  BK Pauli strings:")
    for c, pd in bk_terms:
        label = ' '.join(f'{op}{q}' for q, op in sorted(pd.items())) or 'I'
        print(f"    {c:+.6f} * {label}")

    results['bk_terms']  = bk_terms
    results['n_bk']      = n_bk

    # ── Stage 3: Z2 Symmetry Tapering ────────────────────────────────────────
    print("\n── Stage 3: Z2 Symmetry Tapering ──────────────────────────────────")
    print(f"  Projecting onto sector λ = {taper_lambda:.0f}")
    tap_terms, n_tap, removed_q = z2_taper(bk_terms, n_bk, taper_lambda)
    print(f"  Removed qubit(s): {removed_q}")
    print(f"  n_qubits (after tapering) = {n_tap}")
    print("  Tapered Hamiltonian:")
    for c, pd in tap_terms:
        label = ' '.join(f'{op}{q}' for q, op in sorted(pd.items())) or 'I'
        print(f"    {c:+.6f} * {label}")

    results['tap_terms'] = tap_terms
    results['n_tap']     = n_tap

    if n_tap == 0:
        # All qubits removed — Hamiltonian is a constant
        E_tap = sum(c for c, pd in tap_terms if not pd)
        print(f"\n  All qubits tapered. E_0 (direct) = {E_tap:.8f}")
        results['E0_tap']    = E_tap
        results['E0_xbk_sa'] = E_tap
        return results

    # Physical spin count (after XBK replication)
    n_phys = r * n_tap

    # ── Stage 4: XBK + Counting Operator over ALL Sectors ────────────────────
    print(f"\n── Stage 4: XBK Transformation  (r = {r}) ──────────────────────────")
    print(f"  Logical qubits (m) = {n_tap}")
    print(f"  Physical spins after XBK (r·m) = {n_phys}")
    print(f"  Number of XBK sector combinations = 2^{n_tap} = {2**n_tap}")

    print("\n── Stage 5: Counting Operator ──────────────────────────────────────")
    C_ising = build_counting_op(n_tap)
    print("  C_p(s) Ising polynomial:")
    for mono, c in sorted(C_ising.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
        label = ' '.join(f's{i}' for i in sorted(mono)) or '1'
        print(f"    {c:+.6f} * {label}")

    # ── Rosenberg for counting operator (same M for all sectors) ─────────────
    M = penalty_scale * max((abs(c) for c, _ in tap_terms), default=1.0)
    M = max(M, 1.0)
    print(f"\n── Stage 6: Rosenberg Quadratisation  (penalty M = {M:.3f}) ────────")

    C_bin         = _ising_to_binary(C_ising)
    C_quad_bin, _ = rosenberg_reduce(C_bin, M, next_id=n_phys)
    C_quad_ising  = _binary_to_ising(C_quad_bin)

    # ── Stage 5 / 6 / 7: Loop over all XBK sectors ───────────────────────────
    print(f"\n── Stage 7: Dinkelbach Loop (max_sweeps={n_sweeps}) ─────────────────")
    best_E0 = float('inf')

    for sec in range(2 ** n_tap):
        # sector_signs[i] = S_p(i, 2): the replica sign for logical qubit i.
        # We enumerate all 2^{n_tap} sign combinations.
        sector_signs = [+1 if (sec >> i) & 1 == 0 else -1 for i in range(n_tap)]

        print(f"\n  Sector {sec}/{2**n_tap - 1}:  signs = {sector_signs}")

        # Build H'_p (pre-quadratisation, physical spins only)
        H_ising = build_hprime(tap_terms, n_tap, sector_signs)
        print("  H'_p Ising polynomial:")
        for mono, c in sorted(H_ising.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            label = ' '.join(f's{i}' for i in sorted(mono)) or '1'
            print(f"    {c:+.6f} * {label}")

        # ── Rosenberg reduction (H and C share a global auxiliary-index space) ──
        # next_id starts at n_phys (after the physical spins).
        # H is quadratised first, then C continues from H's next_id.
        # This ensures H and C auxiliary variables occupy disjoint index ranges.
        next_id   = n_phys

        H_bin              = _ising_to_binary(H_ising)
        H_quad_bin, next_id = rosenberg_reduce(H_bin, M, next_id=next_id)
        H_quad             = _binary_to_ising(H_quad_bin)

        C_bin              = _ising_to_binary(C_ising)
        C_quad_bin, next_id = rosenberg_reduce(C_bin, M, next_id=next_id)
        C_quad             = _binary_to_ising(C_quad_bin)

        n_aug = next_id   # total spin variables (physical + auxiliaries)

        # Run Dinkelbach
        E0 = dinkelbach_solve(
            H_pre=H_ising,   C_pre=C_ising,
            H_quad=H_quad,   C_quad=C_quad,
            n_phys=n_phys,   n_aug=n_aug,
            n_sweeps=n_sweeps, base_seed=seed + sec * 100,
            max_iter=30, tol=1e-7
        )
        print(f"  Dinkelbach E_0 (sector {sec}) = {E0:.8f}")

        if E0 < best_E0:
            best_E0 = E0

    print(f"\n  Best E_0 across all sectors = {best_E0:.8f}")
    results['E0_xbk_sa'] = best_E0

    # Analytical result (valid for L=2, PBC)
    E0_analytical = -(V / 2.0 + 2.0 * t)
    print(f"\n── Analytical E_0 (L=2, PBC) = {E0_analytical:.8f}")
    results['E0_analytical'] = E0_analytical

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main: run pipeline and benchmark against ED
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)
    random.seed(0)

    print("=" * 65)
    print("1D t-V MODEL  |  L=2  |  PBC  |  BK -> Z2 Tap -> XBK(r=2) -> Dink.")
    print(f"Parameters:  t = {t:.2f},  V = {V:.2f},  r = {r}")
    print("=" * 65)

    # ── Exact Diagonalisation (benchmark) ────────────────────────────────────
    E0_ed = exact_diagonalise(L, t, V, PERIODIC)
    print(f"\nExact Diagonalisation (ED) ground-state energy: {E0_ed:.10f}")

    # ── Analytical result ─────────────────────────────────────────────────────
    # After BK + Z2 tapering (λ=-1 sector):
    #   H_tap = -2t X_0 - V/2 I
    # Eigenvalues of X: ±1  →  energies: -V/2 ± 2t.
    # Ground state: E_0 = -V/2 - 2t.
    E0_analytical = -(V / 2.0 + 2.0 * t)
    print(f"Analytical (L=2, 1-particle sector):           {E0_analytical:.10f}")
    print()

    # ── Pipeline for the half-filling sector (λ = -1, odd parity) ────────────
    print("Running pipeline with taper sector λ = -1 (half-filling):")
    res_neg = run_pipeline(L, t, V, PERIODIC,
                           taper_lambda=-1.0,
                           n_sweeps=15000, seed=42)

    # ── Pipeline for the even-parity sector (λ = +1) for completeness ────────
    print("\nRunning pipeline with taper sector λ = +1 (empty / doubly occupied):")
    res_pos = run_pipeline(L, t, V, PERIODIC,
                           taper_lambda=+1.0,
                           n_sweeps=15000, seed=123)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Exact diagonalisation (ED):          E_0 = {E0_ed:.10f}")
    print(f"  Analytical (L=2, 1-particle):         E_0 = {E0_analytical:.10f}")
    print(f"  Pipeline (λ=-1 taper, XBK+SA):        E_0 = {res_neg.get('E0_xbk_sa', float('inf')):.10f}")
    print(f"  Pipeline (λ=+1 taper, XBK+SA):        E_0 = {res_pos.get('E0_xbk_sa', float('inf')):.10f}")

    best_pipeline = min(res_neg.get('E0_xbk_sa', float('inf')),
                        res_pos.get('E0_xbk_sa', float('inf')))
    print(f"  Best pipeline result:                 E_0 = {best_pipeline:.10f}")
    print(f"  Error vs ED:                        |ΔE| = {abs(best_pipeline - E0_ed):.2e}")

    # ── V/t scan ─────────────────────────────────────────────────────────────
    print("\n── V/t scan: E_0 vs V/t (pipeline λ=-1 vs analytical) ─────────────")
    print(f"  {'V/t':>8}  {'E_0 (anal.)':>14}  {'E_0 (pipeline)':>16}  {'|error|':>10}")
    print("  " + "-" * 54)

    for vt_ratio in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]:
        V_scan    = vt_ratio * t
        E0_an_sc  = -(V_scan / 2.0 + 2.0 * t)
        res_sc    = run_pipeline(L, t, V_scan, PERIODIC,
                                  taper_lambda=-1.0,
                                  n_sweeps=12000, seed=7)
        E0_pl_sc  = res_sc.get('E0_xbk_sa', float('nan'))
        err       = abs(E0_pl_sc - E0_an_sc)
        print(f"  {vt_ratio:>8.2f}  {E0_an_sc:>14.8f}  {E0_pl_sc:>16.8f}  {err:>10.2e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
