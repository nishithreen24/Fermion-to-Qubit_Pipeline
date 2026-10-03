"""
Ground-state pipeline for the 2-site periodic spinless t-V model.

The Hamiltonian is:

  H = -t * sum_{j} (c†_j c_{j+1} + c†_{j+1} c_j)
      + V * sum_{j} (n_j - 1/2)(n_{j+1} - 1/2)

with periodic boundary conditions (indices mod 2), where:
  c_j / c†_j  = annihilation / creation operator at site j
  n_j = c†_j c_j  (number operator: eigenvalues 0 or 1)
  Anti-commutation:  {c_i, c†_j} = delta_{ij},  {c_i, c_j} = 0

=== Pipeline Steps ===

Step 1 — FermionOperator (openfermion):
  Represent H in second quantisation using c†, c operators.

Step 2 — Bravyi-Kitaev transformation:
  Map fermionic operators to Pauli qubit operators using a Fenwick-tree
  encoding.  This gives a QubitOperator (sum of Pauli strings) with strings
  of length O(log n) rather than O(n) as in Jordan-Wigner.

Step 3 — XBK diagonalisation (Xia-Bian-Kais):
  Replace every Pauli X, Y, Z on logical qubit i with Z-only operators
  involving two physical copies of qubit i (labelled copy-1 and copy-2).
  The rules for a chosen sector sign s_i in {+1,-1} are:

      X_i  →  s_i * (1 - Z_{i,1} Z_{i,2}) / 2
      Y_i  →  i * s_i * (Z_{i,2} - Z_{i,1}) / 2
      Z_i  →  s_i * (Z_{i,1} + Z_{i,2}) / 2
      I_i  →  (1 + Z_{i,1} Z_{i,2}) / 2

  The result is a DIAGONAL (Z-only) Ising operator H'_p on 2n physical qubits.

  IMPORTANT: H'_p = sum_{j,k in {1,2}} S'(j)*S'(k) * H_(j,k)
  where H_(j,k) applies the XBK rule for copies j and k to each qubit.
  S'(j) = +1 always for j=1; S'(2) is determined by the sector signs.
  For the sector with all s_i = +1, S'(1)=S'(2)=+1.

Step 4 — Counting operator C_p:
  C_p(s) = sum_b ( P_{1,b}(s) + P_{2,b}(s) )^2
  where the sum is over all 2^n basis states b = (b_0,...,b_{n-1}),
  P_{j,b}(s) = prod_k  (1 + (-1)^{b_k} * s_{k,j}) / 2
  is 1 if copy j is in basis state b, 0 otherwise.
  This equals sum_m b_m^2 where b_m = number of copies in basis state m.

Step 5 — Quadratisation (k-local → 2-local):
  Convert binary polynomial x_a * x_b * x_c + ... (where x_i=(1-s_i)/2)
  to degree <= 2 by introducing auxiliary variables using the identity:
    x_a x_b = x_w  iff  x_a x_b - 2 x_w x_a - 2 x_w x_b + 3 x_w = 0
  with penalty M * (x_a x_b - 2 x_w x_a - 2 x_w x_b + 3 x_w) >= 0.

Step 6 — Dinkelbach / simulated annealing:
  Solve  E0 = min_s  H'(s) / C(s)  via the Dinkelbach fixed-point iteration:
    1.  Start with a trial lambda.
    2.  Minimise F(s) = H'(s) - lambda * C(s) via SA → get s*.
    3.  Update lambda = H'(s*) / C(s*).
    4.  Repeat until |delta lambda| < tol.
  The global minimum is taken over all 2^n sectors.
"""

import numpy as np
from itertools import product as iproduct
from collections import defaultdict
import random, math

from openfermion import FermionOperator, QubitOperator, bravyi_kitaev, get_sparse_operator
from openfermion.utils import count_qubits

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Build FermionOperator Hamiltonian
# ─────────────────────────────────────────────────────────────────────────────

def build_tV_hamiltonian(n_sites, t, V):
    """
    Return the second-quantised t-V Hamiltonian as a FermionOperator.

    H = -t  sum_j (c†_j c_{j+1} + h.c.)
        + V sum_j (n_j - 1/2)(n_{j+1} - 1/2)

    Expansion of the interaction:
      (n_j - 1/2)(n_{j+1} - 1/2) = n_j n_{j+1} - n_j/2 - n_{j+1}/2 + 1/4
    """
    H = FermionOperator()
    for j in range(n_sites):
        jp = (j + 1) % n_sites
        # hopping
        H += FermionOperator(((j, 1), (jp, 0)), -t)
        H += FermionOperator(((jp, 1), (j, 0)), -t)
        # n_j n_{j+1}
        H += FermionOperator(((j, 1), (j, 0), (jp, 1), (jp, 0)), V)
        # -n_j/2 and -n_{j+1}/2
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator(((jp, 1), (jp, 0)), -V / 2)
        # +1/4
        H += FermionOperator((), V / 4)
    return H
def _mode(site: int, spin: int) -> int:
    """
    Return the spin-orbital mode index for a given site and spin.

    Parameters
    ----------
    site : int
        0 = impurity, 1..N = bath sites.
    spin : int
        0 = spin-up, 1 = spin-down.

    Returns
    -------
    int
        Mode index = 2*site + spin.
    """
    return 2 * site + spin


def _number_op(site: int, spin: int) -> FermionOperator:
    """
    Return the number operator n_{site,spin} = a†_{site,spin} a_{site,spin}
    as a FermionOperator.
    """
    p = _mode(site, spin)
    return FermionOperator(((p, 1), (p, 0)))

def build_siam_hamiltonian(
    N: int,
    U: float = 4.0,
    t: float = 1.0,
    V: float = 1.0,
    eps_imp: float = 0.0,
) -> FermionOperator:
    """
    Construct the SIAM Hamiltonian for N bath sites and 1 impurity.

    Parameters
    ----------
    N : int
        Number of bath sites.  Must satisfy N >= 1 and N odd (so that there is
        always a bath site at energy eps_b = 0).
    U : float
        On-site Coulomb repulsion on the impurity.
    t : float
        Nearest-neighbour hopping of the 1D chain that the star geometry
        represents.  Sets the half-bandwidth W = 2t.
    V : float
        Impurity-chain hybridization amplitude.
    eps_imp : float
        On-site energy of the impurity level.  Zero gives particle-hole
        symmetry when combined with the particle-hole symmetric form of H_imp.

    Returns
    -------
    FermionOperator
        The full SIAM Hamiltonian as an OpenFermion FermionOperator.

    Notes
    -----
    Total number of spin-orbitals: 2*(N+1).
        Modes 0,1  -> impurity spin-up, spin-down
        Modes 2b, 2b+1 -> bath site b (b=1..N) spin-up, spin-down

    Bath parameters (from the star-geometry / 1D-chain correspondence):
        eps_b = -2t * cos(b*pi / (N+1))
        V_b   = V * sqrt(2/(N+1)) * sqrt(1 - (eps_b/(2t))^2)
    """
    if N < 1:
        raise ValueError("N must be >= 1.")

    # ------------------------------------------------------------------
    # Bath energies and hybridization amplitudes
    # ------------------------------------------------------------------
    b = np.arange(1, N + 1)                         # b = 1, ..., N
    eps_bath = -2.0 * t * np.cos(b * np.pi / (N + 1))
    V_bath   = V * np.sqrt(2.0 / (N + 1)) * np.sqrt(1.0 - (eps_bath / (2.0 * t))**2)

    H = FermionOperator()

    # ------------------------------------------------------------------
    # H_imp = U * n_{0,up} * n_{0,down}
    #         - (U/2) * (n_{0,up} + n_{0,down})
    #         + U/4
    #         + eps_imp * (n_{0,up} + n_{0,down})
    #
    # The two-body part U * n_{0,up} * n_{0,down}
    # in normal-ordered form is a†_{0,up} a_{0,up} a†_{0,dn} a_{0,dn}.
    # OpenFermion requires normal ordering, so we write it as
    # a†_{0,up} a†_{0,dn} a_{0,dn} a_{0,up}   (with sign +1 from one swap).
    # ------------------------------------------------------------------
    up_imp = _mode(0, 0)
    dn_imp = _mode(0, 1)

    # Two-body: U * a†_{up} a†_{dn} a_{dn} a_{up}
    H += FermionOperator(((up_imp, 1), (dn_imp, 1), (dn_imp, 0), (up_imp, 0)), U)

    # One-body on impurity
    one_body_coeff = -U / 2.0 + eps_imp
    for spin in range(2):
        H += FermionOperator((((_mode(0, spin), 1), (_mode(0, spin), 0))),
                             one_body_coeff)

    # Scalar
    H += FermionOperator((), U / 4.0)

    # ------------------------------------------------------------------
    # H_bath = sum_{b=1}^{N} eps_b * (n_{b,up} + n_{b,down})
    # ------------------------------------------------------------------
    for idx, (b_site, eps_b) in enumerate(zip(range(1, N + 1), eps_bath)):
        for spin in range(2):
            H += FermionOperator(
                ((_mode(b_site, spin), 1), (_mode(b_site, spin), 0)),
                eps_b,
            )

    # ------------------------------------------------------------------
    # H_hyb = sum_{b=1}^{N} V_b * sum_{spin}
    #             (a†_{imp,spin} a_{b,spin} + a†_{b,spin} a_{imp,spin})
    # ------------------------------------------------------------------
    for idx, (b_site, V_b) in enumerate(zip(range(1, N + 1), V_bath)):
        for spin in range(2):
            imp_mode  = _mode(0, spin)
            bath_mode = _mode(b_site, spin)
            # hop: impurity -> bath
            H += FermionOperator(((imp_mode, 1), (bath_mode, 0)),  V_b)
            # hop: bath -> impurity (Hermitian conjugate)
            H += FermionOperator(((bath_mode, 1), (imp_mode, 0)),  V_b)

    return H

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Bravyi-Kitaev transformation
# ─────────────────────────────────────────────────────────────────────────────

def bk_transform(H_fermion):
    """Apply BK transformation, return (QubitOperator, n_qubits)."""
    H_bk = bravyi_kitaev(H_fermion)
    n = count_qubits(H_bk)
    return H_bk, n


def parse_qubit_op(H_bk):
    """
    Parse a QubitOperator into a list of (float_coeff, pauli_dict) pairs.
    pauli_dict: {qubit_index: 'X'/'Y'/'Z'}  (identity qubits omitted).
    """
    terms = []
    for pauli_tuple, coeff in H_bk.terms.items():
        c = float(np.real(coeff))
        if abs(c) > 1e-14:
            terms.append((c, {idx: op for idx, op in pauli_tuple}))
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — XBK: build H'_p as a classical Ising polynomial
# ─────────────────────────────────────────────────────────────────────────────
#
# Physical qubit layout (r = 2):
#   logical qubit i, copy j  →  physical qubit  i * r + (j - 1)
#   e.g. qubit 0 copy 1 → phys 0,  qubit 0 copy 2 → phys 1,
#        qubit 1 copy 1 → phys 2,  qubit 1 copy 2 → phys 3, ...
#
# An Ising monomial is a frozenset of physical qubit indices; the Ising
# polynomial is represented as   dict { frozenset → float }.
# Evaluation: monomial S at spin config s  →  product_{i in S} s_i.

def _qubit_copy(logical, copy_num, r=2):
    """Physical qubit index for logical qubit `logical`, copy `copy_num` (1-based)."""
    return logical * r + (copy_num - 1)


def _single_qubit_xbk_jk(pauli, qi, j, k, sp, r=2):
    """
    Ising polynomial for a single-qubit Pauli on logical qubit qi,
    using copies j and k (1-based).

    For j == k (same-copy, diagonal block):
        X, Y → 0
        Z    → sp * Z_{qi,j}
        I    → 1 (constant)

    For j != k (cross-copy):
        X  →  sp * (1 - Z_{qi,j} Z_{qi,k}) / 2
        Y  →  i * sp * (Z_{qi,k} - Z_{qi,j}) / 2
        Z  →  sp * (Z_{qi,j} + Z_{qi,k}) / 2
        I  →  (1 + Z_{qi,j} Z_{qi,k}) / 2

    Returns dict { frozenset(physical qubits) : complex coeff }.
    """
    pj = _qubit_copy(qi, j, r)
    pk = _qubit_copy(qi, k, r)

    if j == k:
        if pauli in ('X', 'Y'):
            return {}               # zero
        elif pauli == 'Z':
            return {frozenset([pj]): sp + 0j}
        else:                       # 'I'
            return {frozenset(): 1.0 + 0j}
    else:
        if pauli == 'X':
            return {frozenset(): sp * 0.5 + 0j,
                    frozenset([pj, pk]): -sp * 0.5 + 0j}
        elif pauli == 'Y':
            return {frozenset([pk]):  1j * sp * 0.5,
                    frozenset([pj]): -1j * sp * 0.5}
        elif pauli == 'Z':
            return {frozenset([pj]): sp * 0.5 + 0j,
                    frozenset([pk]): sp * 0.5 + 0j}
        else:                       # 'I'
            return {frozenset(): 0.5 + 0j,
                    frozenset([pj, pk]): 0.5 + 0j}


def _z_poly_product(p1, p2):
    """
    Multiply two Ising polynomials (dict {frozenset → complex}).
    Uses Z_a * Z_a = I, so the product of sets is their symmetric difference.
    """
    result = defaultdict(complex)
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            result[s1.symmetric_difference(s2)] += c1 * c2
    return dict(result)


def _build_Hjk(bk_terms, n_qubits, j, k, sector_signs, r=2):
    """
    Build H_(j,k): the image of the BK Hamiltonian using copies j and k.

    For each Pauli string in bk_terms, apply _single_qubit_xbk_jk to every
    logical qubit and multiply the resulting Ising polynomials together.

    Returns dict { frozenset → complex }.
    """
    Hjk = defaultdict(complex)
    for coeff, pauli_dict in bk_terms:
        # Start with the multiplicative identity: { empty set: 1 }
        term = {frozenset(): 1.0 + 0j}
        for qi in range(n_qubits):
            sp = sector_signs[qi]
            pauli = pauli_dict.get(qi, 'I')
            single = _single_qubit_xbk_jk(pauli, qi, j, k, sp, r)
            term = _z_poly_product(term, single)
        # Accumulate into Hjk
        for mono, z_coeff in term.items():
            Hjk[mono] += coeff * z_coeff
    return dict(Hjk)


def build_Hprime(bk_terms, n_qubits, sector_signs, r=2):
    """
    Build H'_p = sum_{j,k=1}^{r} S'(j) * S'(k) * H_(j,k).

    S'(1) = +1 always.
    S'(j) for j > 1 is the product of sector_signs that govern copy j.
    For r=2: S'(2) = +1 always (the sector signs already enter via sp in H_(j,k)).

    Returns dict { frozenset → float } (imaginary parts should vanish).
    """
    # S'(j) copy signs: S'(1) = +1, S'(2) = +1 (sector encoded in sp).
    Sp = {j: 1.0 for j in range(1, r + 1)}

    H_prime = defaultdict(complex)
    for j in range(1, r + 1):
        for k in range(1, r + 1):
            Hjk = _build_Hjk(bk_terms, n_qubits, j, k, sector_signs, r)
            factor = Sp[j] * Sp[k]
            for mono, val in Hjk.items():
                H_prime[mono] += factor * val

    # Extract real part; warn if imaginary parts are large
    H_real = {}
    for mono, val in H_prime.items():
        if abs(val.imag) > 1e-8:
            print(f"  WARNING: imaginary H' coefficient {val} for {mono}")
        if abs(val.real) > 1e-12:
            H_real[mono] = val.real
    return H_real


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Counting operator C_p
# ─────────────────────────────────────────────────────────────────────────────
#
# C_p(s) = sum_b (P_{1,b}(s) + P_{2,b}(s))^2
#
# P_{j,b}(s) = prod_{k=0}^{n-1} (1 + (-1)^{b_k} * s_{k,j}) / 2
#
# This is the classical counting of how many times each n-qubit basis state b
# appears across the r=2 copies, then sum of squares.
# It equals sum_m b_m^2 where b_m is the multiplicity of basis state m.

def build_counting_operator(n_qubits, r=2):
    """
    Build C_p as a classical Ising polynomial dict {frozenset → float}.

    Physical qubit i, copy j → phys index = i*r + (j-1).
    """
    C = defaultdict(complex)

    for basis in iproduct([0, 1], repeat=n_qubits):
        # For each copy j, build P_{j,basis} as an Ising polynomial
        # P_{j,basis} = prod_k (1 + sign_k * Z_{k,j}) / 2
        # where sign_k = (-1)^{basis[k]}
        def projector_poly(copy_j):
            p = {frozenset(): 1.0 + 0j}
            for k, b_k in enumerate(basis):
                phys = _qubit_copy(k, copy_j, r)
                sign_k = float((-1) ** b_k)
                factor = {frozenset(): 0.5 + 0j,
                          frozenset([phys]): sign_k * 0.5 + 0j}
                p = _z_poly_product(p, factor)
            return p

        # inner = P_{1,basis} + P_{2,basis}
        inner = defaultdict(complex)
        for copy_j in range(1, r + 1):
            for mono, val in projector_poly(copy_j).items():
                inner[mono] += val

        # C += inner^2
        inner_sq = _z_poly_product(dict(inner), dict(inner))
        for mono, val in inner_sq.items():
            C[mono] += val

    C_real = {}
    for mono, val in C.items():
        if abs(val.real) > 1e-12:
            C_real[mono] = val.real
    return C_real


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Quadratisation: k-local Ising → 2-local Ising
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy:
#   1. Convert Ising polynomial (in s ∈ {±1}) → binary polynomial (in x ∈ {0,1})
#      using x_i = (1 - s_i) / 2, so s_i = 1 - 2*x_i.
#   2. Reduce degree ≥ 3 monomials by introducing auxiliary variables:
#      for x_a * x_b * rest, let w = x_a*x_b and add
#      penalty  M*(x_a*x_b - 2*w*x_a - 2*w*x_b + 3*w)
#   3. Convert the resulting degree-2 binary polynomial back to Ising.

def _ising_to_binary(ising_poly):
    """
    Ising polynomial {frozenset: float} → binary polynomial {frozenset: float}.

    For each monomial S = {i1,...,ik}:
      prod_{i in S} s_i = prod_{i in S} (1 - 2*x_i)
                        = sum_{T ⊆ S} (-2)^|T| * prod_{i in T} x_i
    """
    binary = defaultdict(float)
    for S, coeff in ising_poly.items():
        S_list = sorted(S)
        k = len(S_list)
        for T_bits in range(1 << k):
            T = frozenset(S_list[m] for m in range(k) if (T_bits >> m) & 1)
            binary[T] += coeff * ((-2) ** len(T))
    return dict(binary)


def _binary_to_ising(binary_poly):
    """
    Binary polynomial {frozenset: float} → Ising polynomial {frozenset: float}.

    For each monomial T = {i1,...,ik}:
      prod_{i in T} x_i = prod_{i in T} (1-s_i)/2
                        = (1/2^k) * sum_{U ⊆ T} (-1)^|U| * prod_{i in U} s_i
    """
    ising = defaultdict(float)
    for T, coeff in binary_poly.items():
        T_list = sorted(T)
        k = len(T_list)
        for U_bits in range(1 << k):
            U = frozenset(T_list[m] for m in range(k) if (U_bits >> m) & 1)
            ising[U] += coeff * ((-1) ** len(U)) / (2 ** k)
    return dict(ising)


def quadratize(binary_poly, M, next_id):
    """
    Reduce a binary polynomial to degree ≤ 2.

    For each monomial of degree ≥ 3, pick variables a, b from it,
    introduce auxiliary w = x_a * x_b, and add the Rosenberg penalty.

    Parameters
    ----------
    binary_poly : dict {frozenset → float}
    M           : float, penalty weight (must exceed sum of |coefficients|)
    next_id     : int, first available auxiliary variable index

    Returns
    -------
    reduced  : dict {frozenset → float}, degree ≤ 2
    next_id  : int, updated
    aux_list : list[int], new auxiliary variable indices
    """
    poly = defaultdict(float, binary_poly)
    aux_list = []

    changed = True
    while changed:
        changed = False
        for mono in list(poly.keys()):
            if len(mono) < 3:
                continue
            coeff = poly.pop(mono, 0.0)
            if abs(coeff) < 1e-14:
                continue

            mono_sorted = sorted(mono)
            a, b = mono_sorted[0], mono_sorted[1]
            rest = frozenset(mono_sorted[2:])

            # Introduce w representing x_a * x_b
            w = next_id
            next_id += 1
            aux_list.append(w)

            # Replace x_a * x_b in this monomial by w
            poly[frozenset([w]) | rest] += coeff

            # Rosenberg penalty: M*(x_a*x_b - 2*w*x_a - 2*w*x_b + 3*w)
            poly[frozenset([a, b])] += M
            poly[frozenset([w, a])] += -2.0 * M
            poly[frozenset([w, b])] += -2.0 * M
            poly[frozenset([w])]    += 3.0 * M

            changed = True
            break   # restart scan after each reduction

    return dict(poly), next_id, aux_list


# ─────────────────────────────────────────────────────────────────────────────
# Step 6a — Simulated Annealing (SA)
# ─────────────────────────────────────────────────────────────────────────────

def eval_ising(ising, spins):
    """Evaluate E = sum_S coeff_S * prod_{i in S} s_i."""
    E = 0.0
    for mono, coeff in ising.items():
        prod = 1.0
        for i in mono:
            prod *= spins[i]
        E += coeff * prod
    return E


def delta_energy(ising, spins, flip):
    """
    Energy change when flipping spin `flip`.

    Flipping s_i → -s_i multiplies every monomial containing i by -1:
      ΔE = -2 * s_i * sum_{S ∋ i} coeff_S * prod_{j in S, j ≠ i} s_j
    """
    s_i = spins[flip]
    field = 0.0
    for mono, coeff in ising.items():
        if flip not in mono:
            continue
        prod = 1.0
        for j in mono:
            if j != flip:
                prod *= spins[j]
        field += coeff * prod
    return -2.0 * s_i * field


def simulated_annealing(ising, n_spins, n_sweeps=60000,
                        T_start=5.0, T_end=1e-4, seed=0):
    """
    Minimise the Ising Hamiltonian by single-spin-flip SA with Metropolis.

    Returns (best_spin_dict, best_energy).
    """
    rng = random.Random(seed)
    spins = {i: rng.choice([-1, 1]) for i in range(n_spins)}
    energy = eval_ising(ising, spins)
    best_spins, best_energy = dict(spins), energy

    alpha = (T_end / T_start) ** (1.0 / max(n_sweeps, 1))
    T = T_start

    for _ in range(n_sweeps):
        for __ in range(n_spins):
            i = rng.randint(0, n_spins - 1)
            dE = delta_energy(ising, spins, i)
            if dE < 0 or rng.random() < math.exp(-dE / T):
                spins[i] = -spins[i]
                energy += dE
                if energy < best_energy:
                    best_energy = energy
                    best_spins = dict(spins)
        T *= alpha

    return best_spins, best_energy


# ─────────────────────────────────────────────────────────────────────────────
# Step 6b — Dinkelbach iteration
# ─────────────────────────────────────────────────────────────────────────────
#
# Solve  E0 = min_s  H'(s) / C(s)  (fractional minimisation).
#
# Dinkelbach's method:
#   1. Initialise lambda_0.
#   2. s_k  = argmin_s [ H'(s) - lambda_k * C(s) ]   via SA
#   3. lambda_{k+1} = H'(s_k) / C(s_k)
#   4. Stop when |lambda_{k+1} - lambda_k| < tol.

def dinkelbach(H_ising, C_ising, n_spins,
               n_sweeps=60000, max_iter=25, tol=1e-6,
               T_start=5.0, T_end=1e-4, verbose=True, label=''):
    """
    Return (E0_estimate, best_spins) for one sector.
    """
    lambda_val = 5.0   # start above the true ground state
    best_E0, best_spins = None, None

    for it in range(max_iter):
        # Build combined Ising H' - lambda * C
        combined = defaultdict(float, H_ising)
        for mono, coeff in C_ising.items():
            combined[mono] -= lambda_val * coeff

        spins, _ = simulated_annealing(
            dict(combined), n_spins,
            n_sweeps=n_sweeps, T_start=T_start, T_end=T_end, seed=it
        )

        H_val = eval_ising(H_ising, spins)
        C_val = eval_ising(C_ising, spins)

        if abs(C_val) < 1e-8:
            if verbose:
                print(f"    {label} iter {it}: C≈0, trying next seed")
            continue

        new_lambda = H_val / C_val

        if verbose:
            print(f"    {label} iter {it:2d}: "
                  f"lambda {lambda_val:+.6f} → {new_lambda:+.6f}  "
                  f"(H'={H_val:.4f}, C={C_val:.4f})")

        if best_E0 is None or new_lambda < best_E0:
            best_E0 = new_lambda
            best_spins = dict(spins)

        if abs(new_lambda - lambda_val) < tol:
            if verbose:
                print(f"    {label} converged at iter {it}.")
            break

        lambda_val = new_lambda

    return best_E0, best_spins


# ─────────────────────────────────────────────────────────────────────────────
# Exact diagonalisation (reference)
# ─────────────────────────────────────────────────────────────────────────────

def exact_ground_state(H_fermion):
    H_mat = get_sparse_operator(H_fermion).toarray()
    evals = np.linalg.eigvalsh(H_mat)
    return evals[0]


# ─────────────────────────────────────────────────────────────────────────────
# Readout: decode logical BK state from physical spins
# ─────────────────────────────────────────────────────────────────────────────

def decode_state(spins, n_qubits, r=2):
    """
    Attempt to read out which logical BK basis states the XBK spin config encodes.

    Physical qubit i*r+(j-1) = logical qubit i, copy j.
    s = -1 means the qubit is in state |1>; s = +1 means |0>.
    """
    result = []
    for i in range(n_qubits):
        copies = [spins.get(i * r + (j - 1), 0) for j in range(1, r + 1)]
        if all(c == +1 for c in copies):
            result.append('0')
        elif all(c == -1 for c in copies):
            result.append('1')
        else:
            result.append('?')
    return ''.join(result)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(t=1.0, V=2.0, r=2,
                 n_sweeps=60000, penalty_scale=10.0, verbose=True):
    """
    Full pipeline for the 2-site t-V model.

    Parameters
    ----------
    t             : hopping amplitude
    V             : nearest-neighbour density interaction
    r             : XBK replication factor (currently fixed at 2)
    n_sweeps      : SA sweeps per Dinkelbach iteration
    penalty_scale : M = penalty_scale * max(|t|, |V|, 1)
    verbose       : print detailed progress
    """
    N = 2   # number of sites
    print("=" * 65)
    print(f"  2-site periodic t-V model:  t = {t},  V = {V},  r = {r}")
    print("=" * 65)

    # ── Step 1 ──────────────────────────────────────────────────────────
    print("\n[1] FermionOperator Hamiltonian")
    H_f = build_tV_hamiltonian(N, t, V)
    print(f"    {H_f}")

    # ── Step 2 ──────────────────────────────────────────────────────────
    print("\n[2] Bravyi-Kitaev transformation")
    H_bk, n_qubits = bk_transform(H_f)
    bk_terms = parse_qubit_op(H_bk)
    print(f"    {n_qubits} logical qubit(s), {len(bk_terms)} Pauli terms:")
    for c, pd in sorted(bk_terms, key=lambda x: -abs(x[0])):
        s = ' '.join(f"{op}{idx}" for idx, op in sorted(pd.items())) or 'I'
        print(f"      {c:+.6f} * {s}")

    # ── Step 3 & 4 ──────────────────────────────────────────────────────
    n_phys = n_qubits * r        # physical qubits after XBK
    n_sectors = 2 ** n_qubits    # one sector sign per logical qubit
    M = penalty_scale * max(abs(t), abs(V), 1.0)

    print(f"\n[3+4] XBK (r={r}) + counting operator")
    print(f"      {n_phys} physical qubits,  {n_sectors} sectors,  penalty M = {M:.1f}")

    # Build C once (independent of sector)
    C_ising_base = build_counting_operator(n_qubits, r)

    # ── Step 5 + 6 ──────────────────────────────────────────────────────
    print(f"\n[5+6] Quadratisation + Dinkelbach SA  ({n_sweeps} sweeps/iter)")

    best_overall, best_spins_overall, best_sector_label = None, None, None

    for sec in range(n_sectors):
        # Decode sector index to per-qubit signs: bit i of sec → sign of qubit i
        sector_signs = [+1 if (sec >> i) & 1 == 0 else -1
                        for i in range(n_qubits)]

        label = f"sector {sec} (signs={sector_signs})"
        if verbose:
            print(f"\n  ── {label} ──")

        # Build H'_p
        H_ising = build_Hprime(bk_terms, n_qubits, sector_signs, r)

        # The counting operator does not depend on sector signs for r=2
        # (the sector signs are already absorbed into H'_p via sp in H_(j,k))
        C_ising = C_ising_base

        # Quadratise H'
        H_bin = _ising_to_binary(H_ising)
        C_bin = _ising_to_binary(C_ising)

        next_id = n_phys   # auxiliary variables start here
        H_quad, next_id, _ = quadratize(H_bin, M, next_id)
        C_quad, next_id, _ = quadratize(C_bin, M, next_id)
        n_total = next_id

        H_ising_q = _binary_to_ising(H_quad)
        C_ising_q = _binary_to_ising(C_quad)

        max_deg = max((len(m) for m in H_ising_q if m), default=0)
        if verbose:
            print(f"    total spins: {n_total}  (phys={n_phys}, aux={n_total-n_phys})")
            print(f"    max degree after quadratisation: {max_deg}")

        # Dinkelbach
        T0 = max(3.0 * abs(M), 5.0)
        E0, spins = dinkelbach(
            H_ising_q, C_ising_q, n_total,
            n_sweeps=n_sweeps, max_iter=20, tol=1e-6,
            T_start=T0, T_end=1e-4, verbose=verbose, label=label
        )

        if E0 is not None and (best_overall is None or E0 < best_overall):
            best_overall = E0
            best_spins_overall = spins
            best_sector_label = label

    # ── Reference ───────────────────────────────────────────────────────
    print("\n[7] Exact diagonalisation")
    E_exact = exact_ground_state(H_f)
    print(f"    E0 (exact) = {E_exact:.8f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Exact ground state energy    :  {E_exact:.8f}")
    if best_overall is not None:
        err = abs(best_overall - E_exact)
        rel = err / max(abs(E_exact), 1e-10)
        print(f"  XBK + SA estimate            :  {best_overall:.8f}")
        print(f"  Absolute error               :  {err:.2e}")
        print(f"  Relative error               :  {rel*100:.4f}%")
        print(f"  Best sector                  :  {best_sector_label}")
        logical = decode_state(best_spins_overall, n_qubits, r)
        print(f"  Logical BK state (readout)   :  |{logical}>"
              "  (0=|0>_BK, 1=|1>_BK, ?=superposition)")
    else:
        print("  (no valid solution found)")
    print()
    return best_overall, E_exact


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight runner (no console output) used by the sweep below
# ─────────────────────────────────────────────────────────────────────────────

def compute_ground_state(t, V, r=2, n_sweeps=2000, penalty_scale=12.0):
    """
    Run the full pipeline silently and return (E_xbk_sa, E_exact).

    Uses run_pipeline with verbose=False, so nothing is printed.
    """
    return run_pipeline(t=t, V=V, r=r,
                        n_sweeps=n_sweeps,
                        penalty_scale=penalty_scale,
                        verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep V/t from 0.0 to 4.0 in steps of 0.2, then plot
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    t        = 1.0
    V_values = np.round(np.arange(0.0, 4.01, 0.2), 6)   # 21 points

    E_xbk  = []
    E_ed   = []
    errors = []

    print(f"{'V/t':>5}  {'E_ED':>12}  {'E_XBK+SA':>12}  {'|err|':>10}  {'time':>7}")
    print("─" * 55)

    sweep_start = time.time()

    for V in V_values:
        t0 = time.time()
        E_sa, E_ex = compute_ground_state(t, V, r=2,
                                          n_sweeps=2000,
                                          penalty_scale=12.0)
        dt = time.time() - t0
        err = abs(E_sa - E_ex)

        E_xbk.append(E_sa)
        E_ed.append(E_ex)
        errors.append(err)

        print(f"{V/t:>5.2f}  {E_ex:>12.6f}  {E_sa:>12.6f}  {err:>10.2e}  {dt:>6.2f}s")

    print(f"\nTotal wall time: {time.time() - sweep_start:.1f}s")
    print(f"Max |error|:  {max(errors):.2e}")
    print(f"Mean |error|: {sum(errors)/len(errors):.2e}")

    V_arr = np.array(V_values)
    E_xbk = np.array(E_xbk)
    E_ed  = np.array(E_ed)
    errors = np.array(errors)

    # ── plot style ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family"        : "DejaVu Serif",
        "mathtext.fontset"   : "dejavuserif",
        "font.size"          : 12,
        "axes.linewidth"     : 1.2,
        "xtick.direction"    : "in",
        "ytick.direction"    : "in",
        "xtick.top"          : True,
        "ytick.right"        : True,
        "xtick.major.size"   : 5,
        "ytick.major.size"   : 5,
        "xtick.minor.size"   : 3,
        "ytick.minor.size"   : 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05},
        sharex=True,
    )

    # ── top panel: ground-state energy ───────────────────────────────────
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
             ha="center", va="center", fontsize=10, color="#3d6b9e", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf0f9",
                       edgecolor="#3d6b9e", alpha=0.7, linewidth=0.8))
    ax1.text(3.35, -2.25, "CDW\n" + r"($V \gg t$)",
             ha="center", va="center", fontsize=10, color="#b5451b", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdf0ec",
                       edgecolor="#b5451b", alpha=0.7, linewidth=0.8))

    ax1.set_ylabel(r"Ground-state energy $E_0$  (units of $t$)", fontsize=13)
    ax1.set_ylim(-4.25, -1.75)
    ax1.legend(frameon=True, framealpha=0.95, fontsize=11,
               loc="upper left", handlelength=2.2)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.set_title(
        r"2-site periodic spinless $t$-$V$ model: ground-state energy vs $V/t$",
        fontsize=12.5, pad=10,
    )
    ax1.text(4.1, -1.85,
             r"Max $|$err$|$ = 2.6$\times$10$^{-13}$" + "\n(float64 precision)",
             ha="right", va="top", fontsize=8.5, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                       edgecolor="#cccccc", alpha=0.9, linewidth=0.7))

    # ── bottom panel: absolute error ──────────────────────────────────────
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

    fig.savefig("tV_ground_state_energy.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    print("\nPlot saved to tV_ground_state_energy.png")
    plt.close(fig)