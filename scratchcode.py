#!/usr/bin/env python3
"""
Quantum Condensed Matter → 2-Local Ising (QUBO) Pipeline
=========================================================

Pipeline stages
---------------
FermionOperator
  → QubitOperator            (Bravyi-Kitaev transform, via openfermion)
  → k-local diagonal HUBO   (XBK transform, r=2 copies)
  → Boolean PUBO             (Ising z_i ↦ Boolean x_i via z_i = 1 − 2x_i)
  → 2-local QUBO             (Rosenberg quadratisation)
  → Ground state             (SimulatedAnnealingSampler, dwave.samplers)

Benchmark
---------
  Exact ground state energy via sparse Lanczos diagonalisation (scipy).

Models implemented
------------------
  tv_1d, tv_2d               — spinless t-V model
  hubbard_1d, hubbard_2d     — Hubbard model
  j1j2_1d, j1j2_2d           — J1-J2 Heisenberg model
  siam                       — Single Impurity Anderson Model
  pam_1d                     — Periodic Anderson Model
  kitaev_chain               — Kitaev p-wave superconductor chain

Notation
--------
  z_i ∈ {−1, +1}  — Ising spin variables
  x_i ∈ {0, 1}    — Boolean (binary) variables
  z_i = 1 − 2 x_i (x_i = 0 ↔ spin up, x_i = 1 ↔ spin down)

  Ising HUBO:  H = Σ_S  h_S · ∏_{i∈S} z_i
  Boolean PUBO: E = Σ_S  e_S · ∏_{i∈S} x_i
  QUBO:         E = Σ_{i≤j}  Q_{ij} · x_i x_j

XBK transform  (Xia, Bian, Kais 2017, arXiv:1706.00271)
---------------------------------------------------------
For each qubit q, two Ising spins are introduced:
  copy 0 → spin index 2q
  copy 1 → spin index 2q + 1

For a Pauli string term c · ⊗_q P_q with S′(0) = S′(1) = +1 and r = 2:

  H′ = H′_(0,0)  +  H′_(1,1)  +  H′_(0,1)  +  H′_(1,0)

Diagonal pieces H′_(j,j)  [non-zero only when term has no X or Y]:
  H′_(j,j) = c · ∏_{q: P_q = Z}  z_{2q+j}

Off-diagonal piece (H′_(0,1) + H′_(1,0) = 2 H′_(0,1) for even-Y-count terms):
  H′_(0,1) = c · ∏_{q active}  f(q, P_q)

where the XBK factors (eq. 3 in the paper) are:
  P_q = I  →  (1 + z_{2q} z_{2q+1}) / 2
  P_q = Z  →  (z_{2q} + z_{2q+1}) / 2
  P_q = X  →  (1 − z_{2q} z_{2q+1}) / 2
  P_q = Y  →  i (z_{2q+1} − z_{2q}) / 2

Implementation note: we use the "active-qubits-only" expansion — identity
factors for qubits absent from each Pauli string are omitted. This is
exact for full-weight terms and a controlled approximation for lower-weight
terms; it is tractable for Bravyi-Kitaev output (Pauli weight O(log n)).

Rosenberg quadratisation  (Rosenberg 1975)
------------------------------------------
For any Boolean cubic monomial x_{v1} x_{v2} x_{v3} introduce auxiliary y:

  x_{v1} x_{v2} x_{v3}  →  y x_{v3}
  + penalty M·(x_{v1} x_{v2} − 2 x_{v1} y − 2 x_{v2} y + 3 y)

where M > Σ |negative coefficients| ensures the penalty term is always
satisfied at a global optimum.  The same auxiliary y is reused whenever
the same pair {v1, v2} appears again.

Ground state energy
-------------------
With r = 2, at the QUBO optimum x*, the Ising spins are z_i = 1 − 2 x_i*.
The HUBO energy at z* satisfies (from XBK eq. 14-15):

  H′(z*) = Σ_m b²_m · λ

where λ is the ground state energy of the original Hamiltonian H and
Σ_m b²_m ≈ r = 2 for a normalised ground state with one basis vector per
copy.  Hence the estimated ground state energy is:

  λ ≈ H′(z*) / 2

This estimate is benchmarked against exact diagonalisation.

Requirements
------------
    pip install openfermion dimod dwave-samplers scipy numpy
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Dict, FrozenSet, List, Optional, Tuple

from openfermion import FermionOperator, QubitOperator
from openfermion.transforms import bravyi_kitaev, jordan_wigner

import dimod
from dwave.samplers import SimulatedAnnealingSampler


# ═══════════════════════════════════════════════════════════════════════════════
# Type aliases
# ═══════════════════════════════════════════════════════════════════════════════

# Ising HUBO:  dict mapping a frozenset of spin indices → float coefficient.
# frozenset()  is the constant (scalar) term.
# The model energy at Ising configuration z is:
#   E(z) = Σ_{S} h_S · ∏_{i∈S} z_i
IsingHUBO = Dict[FrozenSet[int], float]

# Boolean PUBO: same structure but variables x_i ∈ {0,1} and x_i² = x_i.
BoolPUBO  = Dict[FrozenSet[int], float]

# QUBO: dict[(i,j), float] with i ≤ j.
# Energy E = Σ_{i≤j} Q_{ij} x_i x_j
QUBO = Dict[Tuple[int, int], float]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Ising polynomial arithmetic
# ═══════════════════════════════════════════════════════════════════════════════

def _ising_mul(p1: dict, p2: dict) -> dict:
    """
    Multiply two Ising polynomials.

    Key rule: z_i² = 1  ⟹  z_i · z_i = 1.
    Product of monomials S1 and S2:
        ∏_{i∈S1} z_i · ∏_{i∈S2} z_i  =  ∏_{i∈S1 Δ S2} z_i
    where Δ is the symmetric difference (duplicate indices cancel).
    """
    result: dict = {}
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            key = s1.symmetric_difference(s2)
            result[key] = result.get(key, 0.0) + c1 * c2
    return result


def _ising_add(p1: dict, p2: dict) -> dict:
    result = dict(p1)
    for k, v in p2.items():
        result[k] = result.get(k, 0.0) + v
    return result


def _clean(p: dict, tol: float = 1e-12) -> dict:
    """Drop terms whose absolute coefficient is below tol."""
    return {k: v for k, v in p.items() if abs(v) > tol}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — XBK transform  (QubitOperator → Ising HUBO)
# ═══════════════════════════════════════════════════════════════════════════════

def _xbk_factor(q: int, pauli: str) -> dict:
    """
    Off-diagonal XBK factor for qubit q with given Pauli operator.

    Spin index convention:
        qubit q, copy 0  →  spin 2q
        qubit q, copy 1  →  spin 2q + 1

    Mapping (XBK paper eq. 3, r = 2, j = copy 0, k = copy 1):
        I  →  (1 + z_{2q} z_{2q+1}) / 2
        Z  →  (z_{2q} + z_{2q+1}) / 2
        X  →  (1 − z_{2q} z_{2q+1}) / 2
        Y  →  i(z_{2q+1} − z_{2q}) / 2
    """
    s0   = 2 * q          # copy-0 spin
    s1   = 2 * q + 1      # copy-1 spin
    pair = frozenset({s0, s1})
    f0   = frozenset({s0})
    f1   = frozenset({s1})
    nil  = frozenset()

    if pauli == 'I':
        return {nil: 0.5, pair: 0.5}
    elif pauli == 'Z':
        return {f0: 0.5, f1: 0.5}
    elif pauli == 'X':
        return {nil: 0.5, pair: -0.5}
    elif pauli == 'Y':
        # i · (z_{s1} − z_{s0}) / 2
        return {f1: 0.5j, f0: -0.5j}
    else:
        raise ValueError(f"Unknown Pauli character: {pauli!r}")


def xbk_transform(qubit_op: QubitOperator) -> IsingHUBO:
    """
    Apply the XBK transformation with r = 2, S′(0) = S′(1) = +1.

    For each Pauli term c · ⊗_q P_q in qubit_op, the contribution to H′ is:

        H′_(0,0) + H′_(1,1) + 2 H′_(0,1)

    Diagonal terms H′_(j,j)  (non-zero only if term has no X or Y):
        H′_(j,j)  =  c · ∏_{q: P_q=Z}  z_{2q+j}

    Off-diagonal term  2 H′_(0,1)  (active-qubits-only expansion):
        Multiply XBK factors for qubits with non-identity Paulis only.
        Factor 2 comes from H′_(1,0) = H′_(0,1) for even-Y-count terms
        (which is guaranteed for all Hermitian condensed-matter Hamiltonians).

    Returns
    -------
    hubo : IsingHUBO
        Real coefficients; imaginary parts from Y factors cancel for
        Hermitian operators.
    """
    hubo: dict = {}

    for pauli_term, coeff in qubit_op.terms.items():
        c = complex(coeff)
        pauli_map: Dict[int, str] = dict(pauli_term)

        has_xy = any(p in ('X', 'Y') for p in pauli_map.values())

        # ── Diagonal contributions ──────────────────────────────────────────
        # Non-zero only when no X or Y appear in this Pauli string.
        if not has_xy:
            z_qubits = [q for q, p in pauli_map.items() if p == 'Z']
            for copy in (0, 1):
                key = frozenset(2 * q + copy for q in z_qubits)
                hubo[key] = hubo.get(key, 0.0) + c

        # ── Off-diagonal contribution  2 H′_(0,1) ──────────────────────────
        # Product of XBK factors over active (non-identity) qubits only.
        off: dict = {frozenset(): 1.0}
        for q, p in pauli_map.items():
            off = _ising_mul(off, _xbk_factor(q, p))

        for s, d in off.items():
            hubo[s] = hubo.get(s, 0.0) + 2.0 * c * d

    # Extract real parts (imaginary parts cancel for any Hermitian operator)
    # and drop negligibly small terms.
    result: IsingHUBO = {}
    for s, c in hubo.items():
        re = c.real if hasattr(c, 'real') else float(c)
        if abs(re) > 1e-12:
            result[s] = re
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Ising HUBO → Boolean PUBO
# ═══════════════════════════════════════════════════════════════════════════════

def _ising_mono_to_bool(S: FrozenSet[int]) -> BoolPUBO:
    """
    Convert one Ising monomial ∏_{i∈S} z_i to a Boolean polynomial.

    Substitution  z_i = 1 − 2x_i :

        ∏_{i∈S} (1 − 2x_i)  =  Σ_{T⊆S}  (−2)^|T|  ∏_{j∈T} x_j

    Since x_i ∈ {0,1} satisfies  x_i² = x_i, the union rule replaces
    x_i · x_i  →  x_i (set union instead of multiset).
    """
    poly: BoolPUBO = {frozenset(): 1.0}
    for i in S:
        new: BoolPUBO = {}
        for T, c in poly.items():
            # Contribution from the "1" part of (1 − 2x_i)
            new[T] = new.get(T, 0.0) + c
            # Contribution from the "−2x_i" part; x_i² = x_i so T ∪ {i}
            T_new = frozenset(T | {i})
            new[T_new] = new.get(T_new, 0.0) + c * (-2.0)
        poly = new
    return poly


def ising_to_bool(hubo: IsingHUBO) -> Tuple[BoolPUBO, float]:
    """
    Convert an Ising HUBO to a Boolean PUBO.

    Returns
    -------
    pubo   : BoolPUBO  (constant term excluded)
    offset : float     (the constant term; add back to recover true energy)
    """
    pubo: BoolPUBO = {}
    for S, h in hubo.items():
        for T, c in _ising_mono_to_bool(S).items():
            pubo[T] = pubo.get(T, 0.0) + h * c

    offset = pubo.pop(frozenset(), 0.0)
    return {k: v for k, v in pubo.items() if abs(v) > 1e-12}, offset


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Rosenberg quadratisation  (Boolean PUBO → QUBO)
# ═══════════════════════════════════════════════════════════════════════════════

def rosenberg_reduce(pubo: BoolPUBO) -> Tuple[QUBO, float, int]:
    """
    Reduce a Boolean PUBO to a QUBO via Rosenberg's substitution.

    Algorithm
    ---------
    While any monomial has degree ≥ 3:
      1.  Pick the two lowest-index variables {v1, v2} from that monomial.
      2.  Introduce auxiliary variable y ≡ x_{v1} · x_{v2}.
      3.  Replace x_{v1} x_{v2} with y in the monomial.
      4.  Add penalty:  M·(x_{v1}x_{v2} − 2x_{v1}y − 2x_{v2}y + 3y)
          which is ≥ 0 everywhere, and = 0 iff y = x_{v1}x_{v2}.

    Pair caching: if {v1, v2} has already been substituted, the same y is
    reused — no new variable is created and no new penalty is added.

    Penalty strength: M = 1 + Σ_{c<0} |c|  over all coefficients in the
    current PUBO.  This guarantees M > any single negative coefficient, so
    no solution gains from violating a constraint.

    Returns
    -------
    qubo        : QUBO  (dict[(i,j), float], i ≤ j)
    qubo_offset : float (constant term accumulated during reduction)
    n_aux       : int   (number of auxiliary variables added)
    """
    poly: BoolPUBO = dict(pubo)

    # Penalty strength: large enough to suppress any single violated pair.
    M = 1.0 + sum(abs(v) for v in poly.values() if v < 0.0)

    # Original variable index range
    all_orig = set()
    for key in poly:
        all_orig.update(key)
    next_var = (max(all_orig, default=-1) + 1)

    pair_cache: Dict[FrozenSet[int], int] = {}

    def _add(p: BoolPUBO, key: FrozenSet[int], val: float) -> None:
        p[key] = p.get(key, 0.0) + val

    def _get_or_create_aux(v1: int, v2: int) -> Tuple[int, bool]:
        """Return aux variable for {v1,v2}.  bool = True if newly created."""
        nonlocal next_var
        pair = frozenset({v1, v2})
        if pair in pair_cache:
            return pair_cache[pair], False
        pair_cache[pair] = next_var
        next_var += 1
        return pair_cache[pair], True

    changed = True
    while changed:
        changed = False
        new_poly: BoolPUBO = {}
        for monomial, coeff in poly.items():
            if abs(coeff) < 1e-12:
                continue
            if len(monomial) <= 2:
                _add(new_poly, monomial, coeff)
                continue

            # Degree ≥ 3: pick two variables (sorted for reproducibility).
            vlist = sorted(monomial)
            v1, v2  = vlist[0], vlist[1]
            rest    = frozenset(vlist[2:])

            y, is_new = _get_or_create_aux(v1, v2)

            # Substitute x_{v1} x_{v2} → y in this monomial.
            _add(new_poly, rest | frozenset({y}), coeff)

            # Penalty (only add once per new pair).
            if is_new:
                _add(new_poly, frozenset({v1, v2}),  M)
                _add(new_poly, frozenset({v1, y}),  -2.0 * M)
                _add(new_poly, frozenset({v2, y}),  -2.0 * M)
                _add(new_poly, frozenset({y}),       3.0 * M)

            changed = True

        poly = {k: v for k, v in new_poly.items() if abs(v) > 1e-12}

    # Separate residual constant.
    qubo_offset = poly.pop(frozenset(), 0.0)

    # Build QUBO dict.
    qubo: QUBO = {}
    for monomial, coeff in poly.items():
        if len(monomial) == 1:
            i = next(iter(monomial))
            qubo[(i, i)] = qubo.get((i, i), 0.0) + coeff
        elif len(monomial) == 2:
            i, j = sorted(monomial)
            qubo[(i, j)] = qubo.get((i, j), 0.0) + coeff
        else:
            raise RuntimeError(
                f"Rosenberg reduction left a degree-{len(monomial)} monomial. "
                "This should not happen."
            )

    n_aux = next_var - (max(all_orig, default=-1) + 1)
    return qubo, qubo_offset, n_aux


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Exact diagonalisation (benchmark)
# ═══════════════════════════════════════════════════════════════════════════════

_SIGMA = {
    'I': np.eye(2,             dtype=complex),
    'X': np.array([[0, 1],  [1,  0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0],  [0, -1]], dtype=complex),
}


def exact_ground_state_energy(qubit_op: QubitOperator, n_qubits: int) -> float:
    """
    Compute the exact ground state energy via sparse Lanczos (scipy).

    Hilbert space dimension: 2^n_qubits.
    Dense eigensolver is used when dim ≤ 64; sparse Lanczos otherwise.

    Parameters
    ----------
    qubit_op : QubitOperator  Hamiltonian.
    n_qubits : int            Number of qubits.

    Returns
    -------
    ground_energy : float
    """
    dim = 1 << n_qubits
    H_mat = lil_matrix((dim, dim), dtype=complex)

    for pauli_term, coeff in qubit_op.terms.items():
        pauli_map = dict(pauli_term)
        # Full n-qubit Pauli matrix via Kronecker products.
        mat = np.array([[1.0+0j]])
        for q in range(n_qubits):
            p = pauli_map.get(q, 'I')
            mat = np.kron(mat, _SIGMA[p])
        H_mat += complex(coeff) * mat

    H_csr = csr_matrix(H_mat, dtype=complex)

    if dim <= 64:
        evals = np.linalg.eigvalsh(H_csr.toarray())
        return float(evals.min().real)
    else:
        evals, _ = eigsh(H_csr, k=1, which='SA', tol=1e-10, maxiter=10000)
        return float(evals.min().real)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Condensed matter model constructors
# ═══════════════════════════════════════════════════════════════════════════════

def _n_modes(fop: FermionOperator) -> int:
    """Return the number of fermionic modes (highest mode index + 1)."""
    m = -1
    for term in fop.terms:
        for mode, _ in term:
            m = max(m, mode)
    return m + 1


def tv_1d(n_sites: int, t: float = 1.0, v: float = 0.5,
          pbc: bool = False) -> FermionOperator:
    """
    Spinless t-V model on a 1D chain.

        H = −t Σ_i (a†_i a_{i+1} + h.c.)  +  V Σ_i n_i n_{i+1}

    Mode index: site i → mode i.
    """
    H = FermionOperator()
    for i in range(n_sites - 1 + int(pbc)):
        j = (i + 1) % n_sites
        H += FermionOperator(f'{i}^ {j}', -t)
        H += FermionOperator(f'{j}^ {i}', -t)
        H += FermionOperator(f'{i}^ {i} {j}^ {j}', v)
    return H


def tv_2d(nx: int, ny: int, t: float = 1.0, v: float = 0.5,
          pbc: bool = False) -> FermionOperator:
    """
    Spinless t-V model on a 2D nx×ny lattice (row-major ordering).

    Mode index: site (ix, iy) → iy * nx + ix.
    """
    def idx(ix: int, iy: int) -> int:
        return iy * nx + ix

    H = FermionOperator()
    for iy in range(ny):
        for ix in range(nx):
            i = idx(ix, iy)
            # Horizontal bond
            if ix < nx - 1 or pbc:
                jx = (ix + 1) % nx
                j  = idx(jx, iy)
                H += FermionOperator(f'{i}^ {j}', -t)
                H += FermionOperator(f'{j}^ {i}', -t)
                H += FermionOperator(f'{i}^ {i} {j}^ {j}', v)
            # Vertical bond
            if iy < ny - 1 or pbc:
                jy = (iy + 1) % ny
                j  = idx(ix, jy)
                H += FermionOperator(f'{i}^ {j}', -t)
                H += FermionOperator(f'{j}^ {i}', -t)
                H += FermionOperator(f'{i}^ {i} {j}^ {j}', v)
    return H


def hubbard_1d(n_sites: int, t: float = 1.0, U: float = 2.0,
               mu: float = 0.0, pbc: bool = False) -> FermionOperator:
    """
    Hubbard model on a 1D chain.

        H = −t Σ_{i,σ} (a†_{i,σ} a_{i+1,σ} + h.c.)
            + U Σ_i n_{i,↑} n_{i,↓}
            − μ Σ_{i,σ} n_{i,σ}

    Mode index: site i, spin σ → 2i + σ  (σ=0 for ↑, σ=1 for ↓).
    """
    def m(site: int, spin: int) -> int:
        return 2 * site + spin

    H = FermionOperator()
    for i in range(n_sites - 1 + int(pbc)):
        j = (i + 1) % n_sites
        for spin in (0, 1):
            mi, mj = m(i, spin), m(j, spin)
            H += FermionOperator(f'{mi}^ {mj}', -t)
            H += FermionOperator(f'{mj}^ {mi}', -t)

    for i in range(n_sites):
        up, dn = m(i, 0), m(i, 1)
        H += FermionOperator(f'{up}^ {up} {dn}^ {dn}', U)
        if mu != 0.0:
            H += FermionOperator(f'{up}^ {up}', -mu)
            H += FermionOperator(f'{dn}^ {dn}', -mu)
    return H


def hubbard_2d(nx: int, ny: int, t: float = 1.0, U: float = 2.0,
               mu: float = 0.0, pbc: bool = False) -> FermionOperator:
    """
    Hubbard model on an nx×ny lattice.

    Mode index: site (ix,iy), spin σ → 2·(iy·nx+ix) + σ.
    """
    def idx(ix: int, iy: int, spin: int) -> int:
        return 2 * (iy * nx + ix) + spin

    H = FermionOperator()
    for iy in range(ny):
        for ix in range(nx):
            for spin in (0, 1):
                i = idx(ix, iy, spin)
                if ix < nx - 1 or pbc:
                    j = idx((ix + 1) % nx, iy, spin)
                    H += FermionOperator(f'{i}^ {j}', -t)
                    H += FermionOperator(f'{j}^ {i}', -t)
                if iy < ny - 1 or pbc:
                    j = idx(ix, (iy + 1) % ny, spin)
                    H += FermionOperator(f'{i}^ {j}', -t)
                    H += FermionOperator(f'{j}^ {i}', -t)

    for iy in range(ny):
        for ix in range(nx):
            up = idx(ix, iy, 0)
            dn = idx(ix, iy, 1)
            H += FermionOperator(f'{up}^ {up} {dn}^ {dn}', U)
            if mu != 0.0:
                H += FermionOperator(f'{up}^ {up}', -mu)
                H += FermionOperator(f'{dn}^ {dn}', -mu)
    return H


def j1j2_1d(n_sites: int, J1: float = 1.0, J2: float = 0.5,
             pbc: bool = False) -> FermionOperator:
    """
    J1-J2 Heisenberg spin-1/2 chain via Jordan-Wigner mapping.

        H = J1 Σ_i S_i · S_{i+1}  +  J2 Σ_i S_i · S_{i+2}

    Spin operators expressed in fermionic language (Jordan-Wigner):
        S^z_i = n_i − 1/2
        S^+_i S^-_j + S^-_i S^+_j  =  a†_i a_j + a†_j a_i   (j ≠ i)
        S_i · S_j = (a†_i a_j + a†_j a_i)/2 + (n_i−1/2)(n_j−1/2)

    Mode index: site i → mode i.
    """
    H = FermionOperator()

    def _bond(i: int, j: int, J: float) -> FermionOperator:
        """J · S_i · S_j in fermionic language."""
        b  = FermionOperator(f'{i}^ {j}', J / 2)
        b += FermionOperator(f'{j}^ {i}', J / 2)
        b += FermionOperator(f'{i}^ {i}', -J / 4)
        b += FermionOperator(f'{j}^ {j}', -J / 4)
        b += FermionOperator(f'{i}^ {i} {j}^ {j}', J)
        b += FermionOperator((), J / 4)      # constant
        return b

    for i in range(n_sites - 1 + int(pbc)):
        H += _bond(i, (i + 1) % n_sites, J1)
    for i in range(n_sites - 2 + 2 * int(pbc)):
        H += _bond(i, (i + 2) % n_sites, J2)
    return H


def j1j2_2d(nx: int, ny: int, J1: float = 1.0, J2: float = 0.5,
             pbc: bool = False) -> FermionOperator:
    """
    J1-J2 Heisenberg model on an nx×ny square lattice.

    J1 = nearest-neighbour; J2 = next-nearest-neighbour (diagonal).
    Mode index: site (ix,iy) → iy·nx + ix.
    """
    def idx(ix: int, iy: int) -> int:
        return iy * nx + ix

    def _bond(i: int, j: int, J: float) -> FermionOperator:
        b  = FermionOperator(f'{i}^ {j}', J / 2)
        b += FermionOperator(f'{j}^ {i}', J / 2)
        b += FermionOperator(f'{i}^ {i}', -J / 4)
        b += FermionOperator(f'{j}^ {j}', -J / 4)
        b += FermionOperator(f'{i}^ {i} {j}^ {j}', J)
        b += FermionOperator((), J / 4)
        return b

    H = FermionOperator()
    for iy in range(ny):
        for ix in range(nx):
            i = idx(ix, iy)
            # J1 bonds
            if ix < nx - 1 or pbc:
                H += _bond(i, idx((ix + 1) % nx, iy), J1)
            if iy < ny - 1 or pbc:
                H += _bond(i, idx(ix, (iy + 1) % ny), J1)
            # J2 bonds (NNN diagonal)
            if (ix < nx - 1 or pbc) and (iy < ny - 1 or pbc):
                H += _bond(i, idx((ix + 1) % nx, (iy + 1) % ny), J2)
            if (ix > 0 or pbc) and (iy < ny - 1 or pbc):
                H += _bond(i, idx((ix - 1) % nx, (iy + 1) % ny), J2)
    return H


def siam(n_bath: int, eps_d: float = -2.0, U: float = 4.0,
         eps_k: Optional[List[float]] = None,
         V_k:   Optional[List[float]] = None) -> FermionOperator:
    """
    Single Impurity Anderson Model (SIAM).

        H =  Σ_σ ε_d n_{d,σ}  +  U n_{d,↑} n_{d,↓}
           + Σ_{k,σ} ε_k n_{k,σ}
           + Σ_{k,σ} V_k (a†_{k,σ} a_{d,σ} + h.c.)

    Mode ordering:
        Impurity ↑  →  0
        Impurity ↓  →  1
        Bath k, ↑  →  2 + 2k
        Bath k, ↓  →  2 + 2k + 1

    Parameters
    ----------
    n_bath : int    Number of bath sites.
    eps_d  : float  Impurity on-site energy.
    U      : float  Impurity Coulomb repulsion.
    eps_k  : list   Bath energies (length n_bath; defaults to uniform grid).
    V_k    : list   Hybridisation amplitudes (length n_bath; defaults to 0.4).
    """
    if eps_k is None:
        eps_k = list(np.linspace(-2.0, 2.0, n_bath))
    if V_k is None:
        V_k = [0.4] * n_bath

    H = FermionOperator()
    # Impurity terms
    H += FermionOperator('0^ 0', eps_d)          # ↑ on-site
    H += FermionOperator('1^ 1', eps_d)          # ↓ on-site
    H += FermionOperator('0^ 0 1^ 1', U)         # Hubbard U

    for k in range(n_bath):
        ku = 2 + 2 * k      # bath site k, spin ↑
        kd = 2 + 2 * k + 1  # bath site k, spin ↓
        # Bath kinetic energy
        H += FermionOperator(f'{ku}^ {ku}', eps_k[k])
        H += FermionOperator(f'{kd}^ {kd}', eps_k[k])
        # Hybridisation
        H += FermionOperator(f'{ku}^ 0', V_k[k])
        H += FermionOperator(f'0^ {ku}', V_k[k])
        H += FermionOperator(f'{kd}^ 1', V_k[k])
        H += FermionOperator(f'1^ {kd}', V_k[k])
    return H


def pam_1d(n_sites: int, t: float = 1.0, V: float = 0.5,
           eps_f: float = -2.0, U: float = 4.0,
           pbc: bool = False) -> FermionOperator:
    """
    Periodic Anderson Model on a 1D chain.

        H = −t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)
            + ε_f Σ_{i,σ} n^f_{i,σ}
            + U   Σ_i n^f_{i,↑} n^f_{i,↓}
            + V   Σ_{i,σ} (f†_{i,σ} c_{i,σ} + h.c.)

    Mode ordering (4 modes per site):
        c_{i,↑}  →  4i
        c_{i,↓}  →  4i + 1
        f_{i,↑}  →  4i + 2
        f_{i,↓}  →  4i + 3
    """
    def mc(site: int, spin: int) -> int:
        return 4 * site + spin          # conduction

    def mf(site: int, spin: int) -> int:
        return 4 * site + 2 + spin     # localised f

    H = FermionOperator()
    # Conduction hopping
    for i in range(n_sites - 1 + int(pbc)):
        j = (i + 1) % n_sites
        for spin in (0, 1):
            ci, cj = mc(i, spin), mc(j, spin)
            H += FermionOperator(f'{ci}^ {cj}', -t)
            H += FermionOperator(f'{cj}^ {ci}', -t)

    for i in range(n_sites):
        for spin in (0, 1):
            fi = mf(i, spin)
            ci = mc(i, spin)
            # f on-site energy
            H += FermionOperator(f'{fi}^ {fi}', eps_f)
            # f-c hybridisation
            H += FermionOperator(f'{fi}^ {ci}', V)
            H += FermionOperator(f'{ci}^ {fi}', V)
        # f Hubbard U
        fu, fd = mf(i, 0), mf(i, 1)
        H += FermionOperator(f'{fu}^ {fu} {fd}^ {fd}', U)
    return H


def kitaev_chain(n_sites: int, t: float = 1.0, delta: float = 1.0,
                 mu: float = 0.0, pbc: bool = False) -> FermionOperator:
    """
    Kitaev p-wave superconductor chain.

        H = Σ_i [−t(a†_i a_{i+1} + h.c.)
                 + Δ(a†_i a†_{i+1} + h.c.)
                 − μ(n_i − 1/2)]

    Mode index: site i → mode i.
    """
    H = FermionOperator()
    for i in range(n_sites - 1 + int(pbc)):
        j = (i + 1) % n_sites
        H += FermionOperator(f'{i}^ {j}', -t)
        H += FermionOperator(f'{j}^ {i}', -t)
        H += FermionOperator(f'{i}^ {j}^', delta)
        H += FermionOperator(f'{j} {i}',   delta)

    for i in range(n_sites):
        H += FermionOperator(f'{i}^ {i}', -mu)
        H += FermionOperator((), mu / 2)
    return H


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def eval_ising_hubo(hubo: IsingHUBO, z: Dict[int, float]) -> float:
    """
    Evaluate the Ising HUBO at a given spin configuration z.

    Parameters
    ----------
    hubo : IsingHUBO
    z    : dict mapping spin index → ±1

    Returns
    -------
    energy : float
    """
    energy = 0.0
    for S, h in hubo.items():
        prod = 1.0
        for i in S:
            prod *= z.get(i, 1.0)
        energy += h * prod
    return energy


def run_pipeline(
    fermion_op:   FermionOperator,
    transform:    str   = 'bk',
    num_reads:    int   = 1000,
    benchmark_ed: bool  = True,
    verbose:      bool  = True,
) -> dict:
    """
    Run the full fermionic → QUBO → ground state pipeline.

    Parameters
    ----------
    fermion_op   : FermionOperator  Input Hamiltonian.
    transform    : str              'bk' (Bravyi-Kitaev) or 'jw' (Jordan-Wigner).
    num_reads    : int              Number of SA annealing shots.
    benchmark_ed : bool             Also compute exact ground state energy via ED.
    verbose      : bool             Print per-stage diagnostics.

    Returns
    -------
    results : dict
        n_qubits        — qubits after fermionic transform
        n_ising         — Ising spins (= 2 × n_qubits)
        n_qubo          — QUBO variables (Ising + Rosenberg auxiliary)
        n_aux           — number of auxiliary variables
        n_hubo_terms    — HUBO terms
        n_qubo_entries  — nonzero QUBO entries
        qubo_energy     — raw QUBO energy at SA optimum
        hubo_energy     — HUBO energy decoded from SA optimum (= qubo+offsets)
        estimated_gs    — estimated ground state energy (= hubo_energy / r)
        ed_energy       — exact ground state energy from ED (or None)
        qubit_op        — QubitOperator
        hubo            — IsingHUBO
        qubo            — QUBO dict
        best_sample     — best bitstring from SA (dict {var: 0/1})
    """
    _r = 2  # number of XBK copies

    # ── Stage 1: FermionOperator → QubitOperator ──────────────────────────────
    if transform == 'bk':
        qubit_op = bravyi_kitaev(fermion_op)
    elif transform == 'jw':
        qubit_op = jordan_wigner(fermion_op)
    else:
        raise ValueError(f"transform must be 'bk' or 'jw', got {transform!r}")

    # Determine qubit count from the resulting QubitOperator.
    n_qubits = 0
    for term in qubit_op.terms:
        for q, _ in term:
            n_qubits = max(n_qubits, q + 1)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Transform      : {transform.upper()}")
        print(f"  Qubits         : {n_qubits}")
        print(f"  QubitOp terms  : {len(qubit_op.terms)}")

    # ── Stage 2: XBK transform ────────────────────────────────────────────────
    hubo = xbk_transform(qubit_op)
    n_ising = _r * n_qubits

    if verbose:
        max_deg = max((len(k) for k in hubo), default=0)
        print(f"  Ising spins    : {n_ising}  (= {_r} copies × {n_qubits} qubits)")
        print(f"  HUBO terms     : {len(hubo)}")
        print(f"  Max HUBO deg   : {max_deg}")

    # ── Stage 3: Ising → Boolean ──────────────────────────────────────────────
    pubo, ising_offset = ising_to_bool(hubo)

    if verbose:
        max_deg_pubo = max((len(k) for k in pubo), default=0)
        print(f"  PUBO terms     : {len(pubo)}  (max deg {max_deg_pubo})")

    # ── Stage 4: Rosenberg reduction ─────────────────────────────────────────
    qubo, qubo_offset, n_aux = rosenberg_reduce(pubo)
    total_offset = ising_offset + qubo_offset
    n_qubo = n_ising + n_aux

    if verbose:
        print(f"  QUBO variables : {n_qubo}  ({n_aux} auxiliary)")
        print(f"  QUBO entries   : {len(qubo)}")

    # ── Stage 5: Simulated Annealing ──────────────────────────────────────────
    sampler    = SimulatedAnnealingSampler()
    sampleset  = sampler.sample_qubo(qubo, num_reads=num_reads)
    best       = sampleset.first
    qubo_energy = best.energy   # raw QUBO minimum (excludes total_offset)

    # Decode optimal QUBO bitstring to Ising spins.
    # Auxiliary variable values are irrelevant for HUBO evaluation.
    best_x = best.sample                         # dict {var_idx: 0 or 1}
    z_spins = {i: 1.0 - 2.0 * best_x.get(i, 0)
               for i in range(n_ising)}

    # HUBO energy at decoded configuration, including all offsets.
    hubo_energy_raw = eval_ising_hubo(hubo, z_spins)
    # Add back constant offsets stripped during Ising→Bool and Rosenberg.
    # (total_offset represents the difference between QUBO energy and H')
    hubo_energy = hubo_energy_raw

    # Estimated ground state energy of the original Hamiltonian.
    # From XBK eq. 14-15: H'(z*) ≈ Σ b²_m · λ,  with Σ b²_m ≈ r = 2.
    estimated_gs = hubo_energy / _r

    if verbose:
        print(f"  QUBO energy    : {qubo_energy:.6f}")
        print(f"  HUBO energy    : {hubo_energy:.6f}")
        print(f"  Est. GS energy : {estimated_gs:.6f}  (÷ r={_r})")

    # ── Stage 6: Exact diagonalisation ────────────────────────────────────────
    ed_energy = None
    if benchmark_ed:
        ed_energy = exact_ground_state_energy(qubit_op, n_qubits)
        if verbose:
            print(f"  ED  GS energy  : {ed_energy:.6f}")
            if abs(ed_energy) > 1e-12:
                ratio = estimated_gs / ed_energy
                print(f"  Ratio est/ED   : {ratio:.4f}  (ideally 1.00)")
        print(f"{'─'*60}")

    return {
        'n_qubits':       n_qubits,
        'n_ising':        n_ising,
        'n_qubo':         n_qubo,
        'n_aux':          n_aux,
        'n_hubo_terms':   len(hubo),
        'n_qubo_entries': len(qubo),
        'qubo_energy':    qubo_energy,
        'hubo_energy':    hubo_energy,
        'estimated_gs':   estimated_gs,
        'ed_energy':      ed_energy,
        'qubit_op':       qubit_op,
        'hubo':           hubo,
        'qubo':           qubo,
        'best_sample':    best_x,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — Demo runs
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    DEMOS = [
        # (label, FermionOperator, transform, num_reads, run_ed)
        (
            "1D t-V  (4 sites, t=1, V=0.5, OBC)",
            tv_1d(4, t=1.0, v=0.5, pbc=False),
            'bk', 500, True,
        ),
        (
            "1D Hubbard  (2 sites, t=1, U=2)",
            hubbard_1d(2, t=1.0, U=2.0),
            'bk', 500, True,
        ),
        (
            "Kitaev chain  (4 sites, t=1, Δ=1, μ=0)",
            kitaev_chain(4, t=1.0, delta=1.0, mu=0.0, pbc=False),
            'bk', 500, True,
        ),
        (
            "1D J1-J2  (4 sites, J1=1, J2=0.5)",
            j1j2_1d(4, J1=1.0, J2=0.5),
            'bk', 500, True,
        ),
        (
            "SIAM  (1 bath site)",
            siam(n_bath=1, eps_d=-1.0, U=2.0),
            'bk', 500, True,
        ),
        (
            "PAM  (2 sites, t=1, V=0.5, ε_f=-2, U=4)",
            pam_1d(2, t=1.0, V=0.5, eps_f=-2.0, U=4.0),
            'bk', 500, True,
        ),
    ]

    print("=" * 60)
    print("  Quantum Condensed Matter → QUBO Pipeline")
    print("  XBK transform (r=2) + Rosenberg reduction")
    print("=" * 60)

    summary = []
    for label, fop, tr, nr, do_ed in DEMOS:
        print(f"\n>>> {label}")
        res = run_pipeline(fop, transform=tr, num_reads=nr,
                           benchmark_ed=do_ed, verbose=True)
        summary.append((label, res['estimated_gs'], res['ed_energy']))

    print("\n" + "=" * 60)
    print(f"  {'Model':<40}  {'Est GS':>10}  {'ED GS':>10}  {'Err%':>7}")
    print("=" * 60)
    for label, est, ed in summary:
        if ed is not None and abs(ed) > 1e-10:
            err_pct = 100.0 * abs(est - ed) / abs(ed)
            print(f"  {label:<40}  {est:10.4f}  {ed:10.4f}  {err_pct:6.1f}%")
        else:
            print(f"  {label:<40}  {est:10.4f}  {'N/A':>10}")
    print("=" * 60)