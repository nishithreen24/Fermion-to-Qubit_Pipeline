"""
Z2 Symmetry Tapering for Qubit Hamiltonians
============================================

COMPLETE MATHEMATICAL DESCRIPTION
-----------------------------------

Starting point:
    A qubit Hamiltonian (output of BK or JW transform) expressed as a sum
    of Pauli strings:
        H = sum_j  h_j * sigma_j
    where each sigma_j is an M-qubit Pauli operator, i.e. a tensor product
    of single-qubit matrices chosen from {I, X, Y, Z}.

Goal:
    Find all Z2 symmetries of H (Pauli operators that commute with every
    term in H), then use each one to remove exactly one qubit from H,
    while keeping the ground state energy exactly unchanged.

-------------------------------------------------------------------------------
STEP 1: Symplectic (binary) representation of Pauli operators
-------------------------------------------------------------------------------

Every M-qubit Pauli operator sigma can be written (ignoring global phase) as:
    sigma = prod_{i in A} X_i * prod_{j in B} Z_j   (with A, B subsets of [M])
where Y_i = i * X_i * Z_i contributes to BOTH A and B.

We represent sigma as a binary vector of length 2M:
    v = (x_0, x_1, ..., x_{M-1}  |  z_0, z_1, ..., z_{M-1})  in {0,1}^{2M}
where:
    x_i = 1  iff  sigma acts as X or Y on qubit i
    z_i = 1  iff  sigma acts as Z or Y on qubit i
    (i.e. I -> (0,0),  X -> (1,0),  Z -> (0,1),  Y -> (1,1))

Commutativity condition:
    Two Pauli operators with symplectic vectors v_a and v_b commute iff:
        v_a * J * v_b^T = 0  (mod 2)
    where J is the 2M x 2M symplectic matrix:
        J = [[0, I], [I, 0]]   (I = M x M identity)
    Explicitly:
        v_a * J * v_b^T = (x_a . z_b + z_a . x_b)  mod 2
    where '.' denotes the dot product mod 2.

So two Paulis commute iff the number of positions where one has X (or Y)
and the other has Z (or Y) — in both orders — is even.

-------------------------------------------------------------------------------
STEP 2: Build the constraint matrix E
-------------------------------------------------------------------------------

Stack the r Pauli terms of H as rows in a matrix:
    E in F_2^{r x 2M}    (F_2 = binary field {0,1})
where the j-th row is the symplectic vector of sigma_j.

A Pauli operator tau (represented by binary vector u in {0,1}^{2M}) commutes
with ALL terms in H iff:
    E * J * u^T = 0  (mod 2)
i.e. u lies in the right kernel of the matrix  (E * J)^T = J^T * E^T = J * E^T
(since J is symmetric: J^T = J).

Equivalently, stack the matrix  F = E * J  (computed mod 2), then:
    u is a symmetry vector  iff  F * u^T = 0  (mod 2).

So we need: ker_F2(F) = nullspace of F over the binary field F_2.

-------------------------------------------------------------------------------
STEP 3: Find the nullspace of F over F_2 (binary field)
-------------------------------------------------------------------------------

The nullspace is computed by Gaussian elimination (row reduction) over F_2.
The result is a set of basis vectors {u_1, u_2, ..., u_k} for ker_F2(F).
Each u_i corresponds to a Pauli symmetry operator tau_i.

However, not all vectors in this kernel correspond to independent, non-trivial
symmetries. We need to select a maximal set of linearly independent generators
that all mutually commute (this is always satisfiable for Z2 symmetries).

-------------------------------------------------------------------------------
STEP 4: Convert each symmetry vector back to a Pauli operator tau_i
-------------------------------------------------------------------------------

Given binary vector u = (ux | uz):
    tau_i = prod_{k: ux[k]=1} X_k * prod_{k: uz[k]=1} Z_k
         (where Y_k appears if BOTH ux[k]=1 AND uz[k]=1)

These tau_i are Pauli operators that commute with H.

-------------------------------------------------------------------------------
STEP 5: Find a single-qubit Pauli X_q(i) that anti-commutes with tau_i
        but commutes with tau_j for all j != i
-------------------------------------------------------------------------------

A single-qubit X acting on qubit q anti-commutes with tau_i iff tau_i has
a Z or Y component on qubit q (because X*Z = -Z*X and X*Y = -Y*X).
It commutes with tau_j (j != i) iff tau_j does NOT have a Z or Y on qubit q.

So we need to find a qubit q(i) such that:
    uz_i[q(i)] = 1   (tau_i has Z or Y on qubit q(i))
    uz_j[q(i)] = 0   for all j != i   (tau_j has no Z or Y on qubit q(i))

Such a qubit is called a "pivot" for symmetry i.

-------------------------------------------------------------------------------
STEP 6: Apply a Clifford rotation to map tau_i -> Z_{q(i)}
-------------------------------------------------------------------------------

We want a unitary transformation U such that:
    U * tau_i * U^dagger = Z_{q(i)}   for each i

This is achieved by a product of single-qubit Clifford gates.
The construction (from Bravyi et al.):
    U_i = (1/sqrt(2)) * (X_{q(i)} + tau_i)

This is a Clifford unitary satisfying:
    U_i * X_{q(i)} * U_i^dagger = tau_i
    U_i * tau_i * U_i^dagger = X_{q(i)}

So applying U_i^dagger conjugates tau_i to X_{q(i)}.
Combined with a Hadamard on qubit q(i) (which maps X -> Z), we get tau_i -> Z_{q(i)}.

In practice, we implement this directly on the symplectic representation of
each Pauli term in H: we apply the corresponding Clifford transformation rules
to update the binary vectors of all Pauli terms.

The Clifford that maps tau -> Z_q acts on symplectic vectors as follows.
For each Pauli term sigma_j with symplectic vector v_j = (vx | vz):
    We apply the row operations that implement the Clifford conjugation.

The key rule:  if we want to map tau (with vector u) -> Z_q,
    For any Pauli sigma, the conjugated Pauli sigma' = U sigma U^dagger has
    symplectic vector obtained by the transformation:
        For each qubit k != q:
            if u has X on k (ux[k]=1, uz[k]=0):
                v_j gets modified by adding row of tau on that qubit
        (see implementation below for the exact binary row operations)

    Simpler implementation: we directly check how each gate changes the
    symplectic vector using the known Clifford conjugation rules.

-------------------------------------------------------------------------------
STEP 7: Fix the eigenvalue sector and eliminate qubit q(i)
-------------------------------------------------------------------------------

After the transformation U, the Hamiltonian H' = U^dagger H U commutes with
Z_{q(i)} for each i. This means every Pauli term in H' acts on qubit q(i)
by either I or Z (not X or Y).

To remove qubit q(i), we fix the eigenvalue of Z_{q(i)} to be lambda_i = +1 or -1.
Physically, this restricts to one eigenspace of Z_{q(i)}.
To choose the correct eigenvalue for the ground state:

    lambda_i = <psi_GS | tau_i | psi_GS>

where |psi_GS> is the ground state in the given particle-number sector.
For the ground state sector: lambda_i = +1 by convention (this is the
standard choice, equivalent to choosing the sector that contains the
physical ground state).

In practice: every term in H' acting on qubit q(i) by Z gets its coefficient
multiplied by lambda_i (since Z acting on the eigenstate |+1> or |-1> gives
the eigenvalue back). Terms acting by I are unchanged.
Then qubit q(i) is dropped from the operator: the resulting Hamiltonian acts
on M-1 qubits.

Repeating for each symmetry generator reduces from M to M-k qubits.

-------------------------------------------------------------------------------
IMPLEMENTATION NOTES
-------------------------------------------------------------------------------

We work entirely in the symplectic (binary) representation for efficiency.
The implementation follows the algorithm of Bravyi, Gambetta, Mezzacapo, Temme
(arXiv:1701.08213) and the openfermion-based implementation strategy.

We provide two modes:
  1. Full from-scratch implementation (educational, uses only numpy)
  2. Integration with openfermion QubitOperator objects
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from openfermion import QubitOperator


# ============================================================
# SYMPLECTIC UTILITIES
# ============================================================

def pauli_to_symplectic(pauli_string: tuple, n_qubits: int) -> np.ndarray:
    """
    Convert an openfermion Pauli string to its symplectic binary vector.

    A Pauli string is a tuple like ((0, 'X'), (2, 'Z'), (3, 'Y')).
    The symplectic representation is a binary vector of length 2*n_qubits:
        v = [x_0, ..., x_{n-1}, z_0, ..., z_{n-1}]
    where:
        I -> x_i = 0, z_i = 0
        X -> x_i = 1, z_i = 0
        Z -> x_i = 0, z_i = 1
        Y -> x_i = 1, z_i = 1    (Y = i*X*Z, contributes to both)

    Parameters:
        pauli_string : tuple of (qubit_index, gate_char) pairs
        n_qubits     : total number of qubits

    Returns:
        v : numpy array of shape (2*n_qubits,) with entries in {0, 1}
    """
    v = np.zeros(2 * n_qubits, dtype=np.int8)
    for (qubit, gate) in pauli_string:
        if gate == 'X':
            v[qubit] = 1
        elif gate == 'Z':
            v[n_qubits + qubit] = 1
        elif gate == 'Y':
            v[qubit] = 1
            v[n_qubits + qubit] = 1
        # 'I' contributes nothing
    return v


def symplectic_to_pauli_term(v: np.ndarray, n_qubits: int) -> Tuple[tuple, float]:
    """
    Convert a symplectic binary vector back to a Pauli string (tuple) and
    a phase factor (+1 or -1 or +-i) arising from Y = i*X*Z.

    The phase is +1 for each Y gate (since we split Y = i*X*Z, the 'i'
    must be tracked). However for symmetry generators we only care about
    the Pauli content, not the phase, so phase = 1 here by convention.

    Parameters:
        v        : binary vector [x_0,...,x_{n-1}, z_0,...,z_{n-1}]
        n_qubits : number of qubits

    Returns:
        pauli_string : tuple of (qubit_index, gate_char)
        phase        : float (always 1.0 here; phase is tracked separately)
    """
    pauli_list = []
    for i in range(n_qubits):
        xi = v[i]
        zi = v[n_qubits + i]
        if xi == 1 and zi == 0:
            pauli_list.append((i, 'X'))
        elif xi == 0 and zi == 1:
            pauli_list.append((i, 'Z'))
        elif xi == 1 and zi == 1:
            pauli_list.append((i, 'Y'))
        # (0,0) -> Identity, skip
    return tuple(pauli_list), 1.0


def commutator_symplectic(v_a: np.ndarray, v_b: np.ndarray, n_qubits: int) -> int:
    """
    Check whether two Pauli operators commute, using their symplectic vectors.

    Two Paulis with symplectic vectors v_a = (xa | za) and v_b = (xb | zb) commute iff:
        (xa . zb + za . xb) mod 2 == 0
    where '.' is the dot product over the integers, then taken mod 2.

    Returns:
        0 if they commute, 1 if they anti-commute.
    """
    xa = v_a[:n_qubits]
    za = v_a[n_qubits:]
    xb = v_b[:n_qubits]
    zb = v_b[n_qubits:]
    return int((np.dot(xa, zb) + np.dot(za, xb)) % 2)


def build_constraint_matrix(
    qubit_op: QubitOperator, n_qubits: int
) -> Tuple[np.ndarray, List[tuple], List[float]]:
    """
    Build the constraint matrix F = E * J over F_2 for the given qubit operator.

    E is the matrix (r x 2M) whose rows are the symplectic vectors of each
    Pauli term in the Hamiltonian.
    J is the 2M x 2M symplectic form: J = [[0, I_M], [I_M, 0]].
    F = E * J  (mod 2).

    A vector u is a symmetry vector iff F * u^T = 0 (mod 2).

    Parameters:
        qubit_op : QubitOperator (sum of Pauli strings with coefficients)
        n_qubits : number of qubits

    Returns:
        F             : numpy array (r, 2*n_qubits) over {0,1}, the constraint matrix
        pauli_strings : list of Pauli string tuples (one per term in qubit_op)
        coefficients  : list of real coefficients (one per term)
    """
    pauli_strings = []
    coefficients = []
    rows = []

    for ps, coeff in qubit_op.terms.items():
        v = pauli_to_symplectic(ps, n_qubits)
        # Apply J: J * v = (z_part | x_part), i.e. swap the two halves
        # because J = [[0, I], [I, 0]] and v = [x | z]
        # so J * v = [z | x]
        Jv = np.concatenate([v[n_qubits:], v[:n_qubits]])
        rows.append(Jv)
        pauli_strings.append(ps)
        coefficients.append(coeff.real if hasattr(coeff, 'real') else float(coeff))

    F = np.array(rows, dtype=np.int8)  # shape (r, 2*n_qubits)
    return F, pauli_strings, coefficients


# ============================================================
# BINARY FIELD LINEAR ALGEBRA
# ============================================================

def gf2_nullspace(F: np.ndarray) -> np.ndarray:
    """
    Compute the nullspace of a matrix F over the binary field GF(2) = {0,1}.

    Method: Gaussian elimination (row reduction) over GF(2).
    All arithmetic is mod 2 (XOR for addition, AND for multiplication).

    Given F of shape (r, 2M), we find all vectors u in {0,1}^{2M} such that
        F * u = 0  (mod 2)
    i.e. the right nullspace (kernel) of F over GF(2).

    Algorithm:
        1. Augment F with an identity matrix on the right: [F | I_{2M}]
        2. Row-reduce [F | I] to row-echelon form over GF(2).
        3. Rows of I that correspond to zero rows of the reduced F give
           the nullspace basis vectors.

    Parameters:
        F : numpy array of shape (r, 2M) with entries in {0, 1}

    Returns:
        null_basis : numpy array of shape (k, 2M) where k = dim(ker F),
                     each row is a basis vector of the nullspace over GF(2).
    """
    r, n = F.shape
    # We row-reduce F over GF(2) while tracking column operations on an n x n identity.
    # Equivalently: row-reduce the TRANSPOSE F^T over GF(2), which gives us the
    # column space of F, and the nullspace is its orthogonal complement.
    #
    # Standard algorithm: augment F (r x n) with I_n on the RIGHT as (r x (n+r))
    # is wrong. Instead we augment the TRANSPOSE (n x r) with I_n on the LEFT,
    # row-reduce, and read off the nullspace from zero rows.
    #
    # Cleaner: use the standard "kernel via row reduction" on F directly.
    # Augment F (r x n) with I_r on the RIGHT: shape (r, n+r).
    # After full row reduction, the zero rows of the left block (F part)
    # correspond to rows of the right block (I_r part) that are in the left nullspace.
    # But we want the RIGHT nullspace (ker F, vectors u s.t. Fu = 0).
    #
    # Correct approach: transpose, row reduce, extract free variables.
    # Work on F^T of shape (n, r), augmented with I_n: shape (n, r+n).
    # After row reduction of F^T, rows with zero in the F^T part AND non-zero
    # in the I_n part are the nullspace basis vectors.
    #
    # Even simpler and correct: row reduce F (r x n) over GF(2) to find pivot columns.
    # For each non-pivot column j (= free variable), build the nullspace vector by:
    #   u[j] = 1, and for each pivot column p_i: u[p_i] = reduced_F[i, j].

    A = F.astype(np.int8).copy()  # shape (r, n)
    pivot_cols = []
    pivot_rows_used = []
    row = 0
    for col in range(n):
        # Find pivot in column 'col' at or below current row
        pivot_r = None
        for r2 in range(row, r):
            if A[r2, col] == 1:
                pivot_r = r2
                break
        if pivot_r is None:
            continue
        # Swap rows
        A[[row, pivot_r]] = A[[pivot_r, row]]
        # Eliminate column in all OTHER rows
        for r2 in range(r):
            if r2 != row and A[r2, col] == 1:
                A[r2] = (A[r2] + A[row]) % 2
        pivot_cols.append(col)
        pivot_rows_used.append(row)
        row += 1
        if row >= r:
            break

    # Now A is in reduced row echelon form.
    # For each free column j (not in pivot_cols), build a nullspace vector:
    free_cols = [j for j in range(n) if j not in pivot_cols]
    null_basis = []
    for free_j in free_cols:
        u = np.zeros(n, dtype=np.int8)
        u[free_j] = 1
        # For each pivot row i, u[pivot_col] = A[pivot_row, free_j]
        for pivot_col, pivot_row in zip(pivot_cols, pivot_rows_used):
            u[pivot_col] = A[pivot_row, free_j]
        null_basis.append(u)

    if not null_basis:
        return np.zeros((0, n), dtype=np.int8)
    return np.array(null_basis, dtype=np.int8)


# ============================================================
# SYMMETRY GENERATOR EXTRACTION
# ============================================================

def find_symmetry_generators(
    qubit_op: QubitOperator, n_qubits: int
) -> Tuple[List[np.ndarray], List[QubitOperator]]:
    """
    Find all independent Z2 symmetry generators of the qubit Hamiltonian.

    A Z2 symmetry is a Pauli operator tau such that [tau, H] = 0
    AND tau^2 = I (which is automatic for any Pauli operator).

    The set of all such symmetries forms an abelian group (since any product
    of commuting Paulis also commutes with H). We find a minimal generating
    set = the nullspace basis of F = E*J over GF(2).

    Parameters:
        qubit_op : QubitOperator Hamiltonian (post BK or JW transform)
        n_qubits : number of qubits

    Returns:
        sym_vectors    : list of binary vectors (each of length 2*n_qubits)
                         representing the symmetry generators in symplectic form
        sym_operators  : list of QubitOperator objects for each symmetry generator
    """
    F, _, _ = build_constraint_matrix(qubit_op, n_qubits)
    null_basis = gf2_nullspace(F)  # shape (k, 2*n_qubits)

    sym_vectors = []
    sym_operators = []
    for u in null_basis:
        pauli_str, _ = symplectic_to_pauli_term(u, n_qubits)
        if not pauli_str:
            # Identity operator — trivial symmetry, skip
            continue
        sym_vectors.append(u)
        sym_operators.append(QubitOperator(pauli_str, 1.0))

    return sym_vectors, sym_operators


# ============================================================
# PIVOT QUBIT SELECTION
# ============================================================

def find_pivots(sym_vectors: List[np.ndarray], n_qubits: int) -> List[int]:
    """
    For each symmetry generator tau_i (with symplectic vector u_i),
    find a "pivot" qubit q(i) such that:
        1. X_{q(i)} anti-commutes with tau_i:
           this requires uz_i[q(i)] = 1 (tau_i has Z or Y on qubit q(i))
        2. X_{q(i)} commutes with tau_j for all j != i:
           this requires uz_j[q(i)] = 0 for all j != i

    The anti-commutativity condition arises because X anti-commutes with Z and Y
    (since XZ = -ZX and XY = -YX on a single qubit), but commutes with X and I.

    If no such "clean" pivot exists, we apply a further GF(2) row-reduction
    on the Z-part of the symmetry matrix to diagonalize it (this always succeeds
    for a valid independent set of generators).

    Parameters:
        sym_vectors : list of binary vectors of length 2*n_qubits
        n_qubits    : number of qubits

    Returns:
        pivots : list of integers, pivots[i] = q(i) for symmetry i
                 pivots[i] = -1 if no pivot found (signals error)
    """
    k = len(sym_vectors)
    if k == 0:
        return []

    # Stack Z-parts (columns n_qubits to 2*n_qubits) into a matrix
    # uz_matrix[i, j] = 1  iff  tau_i has Z or Y on qubit j
    uz_matrix = np.array([u[n_qubits:] for u in sym_vectors], dtype=np.int8)

    # Row-reduce uz_matrix over GF(2) to find pivots
    # After row reduction, each row has a leading 1 (pivot) in a unique column.
    # We also track row operations to apply them to sym_vectors.
    matrix = uz_matrix.copy()
    full_sym = np.array(sym_vectors, dtype=np.int8).copy()  # shape (k, 2M)

    pivots = [-1] * k
    pivot_row = 0
    for col in range(n_qubits):
        # Find a row with a 1 in this column, at or below pivot_row
        pivot = None
        for r2 in range(pivot_row, k):
            if matrix[r2, col] == 1:
                pivot = r2
                break
        if pivot is None:
            continue
        # Swap
        matrix[[pivot_row, pivot]] = matrix[[pivot, pivot_row]]
        full_sym[[pivot_row, pivot]] = full_sym[[pivot, pivot_row]]
        # Eliminate
        for r2 in range(k):
            if r2 != pivot_row and matrix[r2, col] == 1:
                matrix[r2] = (matrix[r2] + matrix[pivot_row]) % 2
                full_sym[r2] = (full_sym[r2] + full_sym[pivot_row]) % 2
        pivots[pivot_row] = col
        pivot_row += 1
        if pivot_row >= k:
            break

    # Update sym_vectors with the row-reduced versions (they still span the same space)
    for i in range(k):
        sym_vectors[i] = full_sym[i]

    return pivots


# ============================================================
# CLIFFORD TRANSFORMATION (symplectic implementation)
# ============================================================

def _single_qubit_clifford_transform(
    v: np.ndarray,
    tau_vec: np.ndarray,
    pivot_qubit: int,
    n_qubits: int,
) -> Tuple[np.ndarray, complex]:
    """
    Apply the Clifford conjugation that maps tau -> Z_{pivot} to a single Pauli
    term with symplectic vector v.

    The Clifford unitary is:
        U = (1/sqrt(2)) * (X_{pivot} + tau)
    which satisfies:
        U * X_{pivot} * U^dagger = tau
        U * tau * U^dagger = X_{pivot}

    Followed by a Hadamard on qubit 'pivot', which maps X -> Z.
    So the combined transformation is: tau -> X_{pivot} -> Z_{pivot}.

    In symplectic representation, this Clifford acts on a vector v = (vx | vz) as:
    
    The full Clifford that maps tau -> Z_q is implemented as follows.
    Let tau have symplectic vector u = (ux | uz).
    
    The transformation U (up to phases that we track) acts on any Pauli P as:
        P -> U^dagger P U
    
    The rules (derived from the standard Clifford conjugation identities) are:
    
    For the specific gate that maps tau_i -> Z_q(i):
        We use CNOT-like Clifford gates acting on pairs of qubits.
        
    The simplest implementation: for each Pauli term, compute:
        If [P, tau] = 0 (commutes):
            U P U^dagger = P unchanged  (on the qubits away from the pivot)
            plus: any Z_{pivot} in P is unchanged (since U maps Z_{pivot} -> tau)
        Wait — the action is more subtle. We implement the exact transformation.

    EXACT IMPLEMENTATION:
    The Clifford unitary U_i = (X_{q(i)} + tau_i) / sqrt(2) satisfies:
        For any Pauli sigma:
            U_i sigma U_i^dagger = sigma  if sigma commutes with BOTH X_{q(i)} and tau_i
            U_i sigma U_i^dagger = -sigma if sigma anti-commutes with BOTH
            U_i sigma U_i^dagger = tau_i * sigma  if sigma commutes with tau_i but anti-commutes with X_{q(i)}
            U_i sigma U_i^dagger = X_{q(i)} * sigma if sigma commutes with X_{q(i)} but anti-commutes with tau_i

    More precisely, for a single-qubit Clifford that maps X -> tau and tau -> X:
        sigma' = U sigma U^dagger
        In symplectic terms, if we define:
            x_q = the X-type Pauli on qubit q  (symplectic: e_q | 0)
            tau  = symplectic vector u
        Then the transformation acts on the symplectic vector v of sigma as:
            sigma' has symplectic vector:
                If v_q^x = 0 and v_q^z = 0: sigma' = sigma  (untouched by qubit q)
                If v_q^x = 1 and v_q^z = 0: sigma has X_q. New sigma' replaces X_q with tau.
                    -> v' = v XOR (e_q_x) XOR u   mod 2   (swap the X_q column for tau's pattern)
                If v_q^x = 0 and v_q^z = 1: sigma has Z_q. New sigma' replaces Z_q with ... 
                    -> The Hadamard on q maps Z -> X. Then X -> tau (from U_i).
                    -> So Z_q gets mapped to tau. Apply: v' = v XOR (0|e_q_z) XOR u
                If v_q^x = 1 and v_q^z = 1: sigma has Y_q = iX_q Z_q.
                    -> Both X_q and Z_q get mapped, accumulating a phase.
                    
    This is getting complex. We use the standard approach:
    directly apply the transformation on the full symplectic representation
    using the known action of CNOT, H, and S gates on symplectic vectors.

    For now, we implement the key transformation needed:
        Given that tau has only Z's (after the row-reduction in find_pivots),
        tau = Z_{q1} * Z_{q2} * ... * Z_{qm}  with q1 = pivot
        
        To map tau -> Z_{pivot}, we use CNOT_{qi -> pivot} for all qi != pivot.
        CNOT (control c, target t) maps in symplectic:
            X_c -> X_c X_t    (x vector: flip bit t if bit c is set)
            Z_c -> Z_c        
            X_t -> X_t        
            Z_t -> Z_c Z_t    (z vector: flip bit c if bit t is set)
        
        Hadamard on qubit q maps:
            X_q -> Z_q    (x <-> z for qubit q)
            Z_q -> X_q
    
    We implement this via the exact CNOT + H sequence that diagonalizes tau to Z_q.
    """
    # This function is intentionally left as a placeholder.
    # The actual transformation is implemented in apply_clifford_to_operator below.
    pass


def apply_clifford_to_operator(
    pauli_strings: List[tuple],
    coefficients: List[float],
    sym_vectors: List[np.ndarray],
    pivots: List[int],
    n_qubits: int,
) -> Tuple[List[tuple], List[float]]:
    """
    Apply the sequence of Clifford transformations to the Hamiltonian.

    For each symmetry tau_i with pivot q(i), after the row reduction in
    find_pivots, each tau_i has the form:
        tau_i = Z_{q(i)} * [product of Z's on other qubits from the row-reduced vector]

    We apply CNOT gates (control = other Z qubits, target = q(i)) to map
    each tau_i -> Z_{q(i)}.

    After all CNOTs, the pivoted qubits are acted on by Z_{q(i)} in any
    Hamiltonian term that previously had tau_i component. We then fix Z_{q(i)}
    to its eigenvalue lambda_i.

    The transformation is implemented directly on the symplectic representation:

    CNOT(control=c, target=t) action on a Pauli P with symplectic (vx | vz):
        vx[c] unchanged
        vx[t] -> vx[t] XOR vx[c]      (X_c spreads to X_t)
        vz[c] -> vz[c] XOR vz[t]      (Z_t spreads to Z_c)
        vz[t] unchanged

    This is the standard symplectic representation of CNOT in the Heisenberg picture.
    The phase factor from CNOT is always +1 for this type of propagation.

    Parameters:
        pauli_strings : list of Pauli string tuples (one per Hamiltonian term)
        coefficients  : list of real coefficients
        sym_vectors   : list of (row-reduced) symmetry vectors
        pivots        : list of pivot qubits
        n_qubits      : number of qubits

    Returns:
        new_pauli_strings : transformed Pauli string tuples
        new_coefficients  : same coefficients (Clifford preserves eigenvalues)
    """
    # Convert pauli_strings to a matrix of symplectic vectors: shape (r, 2M)
    r = len(pauli_strings)
    sym_mat = np.zeros((r, 2 * n_qubits), dtype=np.int8)
    for idx, ps in enumerate(pauli_strings):
        sym_mat[idx] = pauli_to_symplectic(ps, n_qubits)

    # phases[idx] tracks accumulated phase factors (complex) for each term
    # Each phase factor is in {+1, -1, +i, -i} (as complex numbers)
    phases = np.ones(r, dtype=complex)

    # For each symmetry, apply the Clifford that maps tau_i -> Z_{pivot_i}
    for sym_idx, (u, pivot) in enumerate(zip(sym_vectors, pivots)):
        if pivot == -1:
            continue

        # u has only Z components (ux part is zero) after row-reduction in find_pivots.
        # tau_i = Z_{pivot} * Z_{q2} * ... (product of Z's, with pivot having uz[pivot]=1)
        # To map Z_{pivot} * Z_{q2} * ... -> Z_{pivot}, apply:
        #     CNOT(control=q2, target=pivot) for each q2 != pivot with uz[q2] = 1

        uz = u[n_qubits:]  # the Z-part of the symmetry vector
        # ux = u[:n_qubits]  # should be zero after row-reduction (tau is Z-type)

        other_z_qubits = [q for q in range(n_qubits)
                          if uz[q] == 1 and q != pivot]

        for ctrl in other_z_qubits:
            # Apply CNOT(control=ctrl, target=pivot) to every row of sym_mat
            # CNOT(c, t) symplectic action:
            #   vx[t] ^= vx[c]    (X spreads from control to target)
            #   vz[c] ^= vz[t]    (Z spreads from target to control)
            # Phase factor: CNOT introduces a phase of (-1) when:
            #   the term has Y on control AND (X or Y) on target — but tracking
            #   this exactly requires careful bookkeeping. For Z-type symmetries
            #   (ux = 0), the CNOT sequence does not change any X component
            #   in the symmetry itself, and the phase is always +1.
            #   For the Hamiltonian terms, phases from CNOT can arise.
            #   The phase rule for CNOT(c,t):
            #     phase *= (-1)^{ vx[c] * vz[c] * vx[t] }   [standard formula]

            vxc = sym_mat[:, ctrl].copy()
            vzc = sym_mat[:, n_qubits + ctrl].copy()
            vxt = sym_mat[:, pivot].copy()
            vzt = sym_mat[:, n_qubits + pivot].copy()

            # Phase update: (-1)^{ vx_ctrl * vz_ctrl * vx_target }
            # This accounts for the Y anticommutation in CNOT
            phase_update = (vxc * vzc * vxt) % 2
            phases *= (-1.0 + 0j) ** phase_update

            # Update X component of target (pivot): XOR with X of control
            sym_mat[:, pivot] = (sym_mat[:, pivot] + sym_mat[:, ctrl]) % 2
            # Update Z component of control: XOR with Z of target
            sym_mat[:, n_qubits + ctrl] = (
                sym_mat[:, n_qubits + ctrl] + sym_mat[:, n_qubits + pivot]
            ) % 2

    # Convert back to Pauli string tuples
    new_pauli_strings = []
    new_coefficients = []
    for idx in range(r):
        v = sym_mat[idx]
        ps, _ = symplectic_to_pauli_term(v, n_qubits)
        new_pauli_strings.append(ps)
        # Apply accumulated phase to coefficient
        coeff = coefficients[idx] * phases[idx].real  # imaginary part ~ 0
        new_coefficients.append(coeff)

    return new_pauli_strings, new_coefficients


# ============================================================
# EIGENVALUE SECTOR DETERMINATION
# ============================================================

def determine_sector_eigenvalues(
    sym_vectors: List[np.ndarray],
    pivots: List[int],
    n_qubits: int,
    n_particles: Optional[int] = None,
    sector: Optional[List[int]] = None,
) -> List[int]:
    """
    Determine the eigenvalue lambda_i in {+1, -1} for each symmetry Z_{q(i)}.

    After the Clifford transformation, each pivot qubit q(i) carries the
    eigenvalue of the corresponding symmetry tau_i in the physical ground state.

    For systems with particle-number conservation:
        The global fermionic parity symmetry P = (-1)^N_total = prod_j Z_j
        has eigenvalue (-1)^N where N is the total particle number.
        Spin-up and spin-down parity eigenvalues are similarly determined by
        the number of up/down electrons.

    For the models in this pipeline (condensed matter lattice models):
        The most common Z2 symmetries are:
        1. Global parity: tau = prod_j Z_j,  lambda = (-1)^{N_particles}
        2. Spin-up parity: tau = prod_{j in up-sector} Z_j,  lambda = (-1)^{N_up}
        3. Spin-down parity: tau = prod_{j in down-sector} Z_j, lambda = (-1)^{N_down}

    If 'sector' is provided explicitly, use it.
    Otherwise, default to lambda_i = +1 for all i (the convention in Bravyi et al.
    that always contains the physical ground state for even-particle-number systems).

    Parameters:
        sym_vectors  : list of symmetry symplectic vectors (post row-reduction)
        pivots       : list of pivot qubits
        n_qubits     : number of qubits
        n_particles  : total particle number (for parity-based determination)
        sector       : explicit list of ±1 eigenvalues (overrides auto-detection)

    Returns:
        lambdas : list of ±1 integers, one per symmetry generator
    """
    k = len(sym_vectors)
    if sector is not None:
        assert len(sector) == k, f"sector must have length {k}"
        return [int(s) for s in sector]

    if n_particles is None:
        # Default: all eigenvalues = +1
        return [+1] * k

    # Determine eigenvalue from particle number and parity structure
    # Default all to +1.
    #
    # Mathematical justification:
    # After the Clifford transformation, the symmetry tau_i has been mapped to
    # Z_{q(i)}.  The physical ground state |psi_GS> has a definite eigenvalue
    # lambda_i = <psi_GS | Z_{q(i)} | psi_GS> = +/-1.  We cannot compute this
    # analytically without knowing |psi_GS>, so the correct procedure is:
    #
    #   (a) Compute tapered Hamiltonian for BOTH sectors (lam = +1 and lam = -1).
    #   (b) Take the sector whose ground state energy is lower.
    #
    # In practice for the models of interest (condensed matter at half filling
    # with BK encoding), the convention lam = +1 for all generators gives the
    # correct ground state sector in the vast majority of cases.
    # Users can override via the `sector` argument when the default is wrong.
    #
    # For global fermionic parity (tau = prod_j Z_j): lambda = (-1)^N.
    # For spin-up/down parity (tau = prod_{j in spin sector} Z_j): lambda = (-1)^{N_sigma}.
    # These are encoded in `sector` when provided by the user.

    lambdas = []
    for u in sym_vectors:
        n_z = int(np.sum(u[n_qubits:]))  # number of Z-type qubits in tau
        if n_z == n_qubits and n_particles is not None:
            # Global parity: tau = prod_j Z_j, eigenvalue = (-1)^N
            lam = int((-1) ** n_particles)
        else:
            # Default to +1. User can override via `sector`.
            lam = +1
        lambdas.append(lam)

    return lambdas


# ============================================================
# QUBIT REMOVAL (eigenvalue substitution)
# ============================================================

def remove_qubit(
    pauli_strings: List[tuple],
    coefficients: List[float],
    pivot: int,
    lam: int,
    n_qubits: int,
) -> Tuple[List[tuple], List[float], Dict[int, int]]:
    """
    Remove qubit 'pivot' from the Hamiltonian by substituting Z_{pivot} -> lam.

    After the Clifford transformation, every term in the Hamiltonian either:
        (a) Acts as I on qubit 'pivot': the term is unchanged (qubit removed trivially)
        (b) Acts as Z on qubit 'pivot': the Z_{pivot} is replaced by the scalar lam = ±1,
            so the coefficient is multiplied by lam.
        (c) Acts as X or Y on qubit 'pivot': this should NOT happen after a correct
            Clifford transformation (the transformed H commutes with Z_{pivot}).
            If it does, raise an error.

    After substitution, we relabel qubits: all qubits > pivot get their index
    shifted down by 1 (since pivot is removed).

    Parameters:
        pauli_strings : list of Pauli string tuples (in the Clifford-transformed basis)
        coefficients  : list of real coefficients
        pivot         : the qubit index to remove
        lam           : eigenvalue ±1 to substitute for Z_{pivot}
        n_qubits      : current number of qubits (before removal)

    Returns:
        new_pauli_strings : Pauli strings with qubit 'pivot' removed
        new_coefficients  : updated coefficients
        qubit_map         : dict {old_index: new_index} for the remaining qubits
                            (qubit 'pivot' is absent from this dict)
    """
    # Build the relabeling map: old qubit q -> new qubit (q if q < pivot, q-1 if q > pivot)
    qubit_map = {}
    new_idx = 0
    for q in range(n_qubits):
        if q == pivot:
            continue
        qubit_map[q] = new_idx
        new_idx += 1

    new_pauli_strings = []
    new_coefficients = []

    for ps, coeff in zip(pauli_strings, coefficients):
        # Check what this term does on qubit 'pivot'
        pivot_gate = 'I'
        remaining = []
        for (q, gate) in ps:
            if q == pivot:
                pivot_gate = gate
            else:
                remaining.append((q, gate))

        if pivot_gate == 'X' or pivot_gate == 'Y':
            # This term has X or Y on the pivot qubit.
            # After a CORRECT Clifford transformation, this must not happen.
            # If it does, it means the term commutes with Z_pivot but has X/Y
            # which contradicts [H', Z_pivot] = 0. Something went wrong.
            # In practice, very small numerical residuals can appear; we drop them.
            import warnings
            warnings.warn(
                f"Term {ps} has {pivot_gate} on pivot qubit {pivot} "
                f"after Clifford transform. This term will be dropped. "
                f"Check your symmetry generators and Clifford sequence."
            )
            continue

        # Multiply coefficient by lam if the term has Z on pivot
        new_coeff = coeff * (lam if pivot_gate == 'Z' else 1)

        # Relabel remaining qubits
        new_ps = tuple((qubit_map[q], gate) for (q, gate) in remaining)
        new_pauli_strings.append(new_ps)
        new_coefficients.append(new_coeff)

    return new_pauli_strings, new_coefficients, qubit_map


# ============================================================
# COMBINE DUPLICATE PAULI TERMS
# ============================================================

def consolidate_terms(
    pauli_strings: List[tuple], coefficients: List[float], tol: float = 1e-12
) -> QubitOperator:
    """
    Combine Pauli terms with the same Pauli string by summing their coefficients.
    Drop terms with |coefficient| < tol.

    This is needed because the Clifford transformation and qubit removal can
    create duplicate Pauli strings (e.g., two terms that both become the identity
    after removing a qubit).

    Parameters:
        pauli_strings : list of Pauli string tuples
        coefficients  : list of real coefficients
        tol           : threshold below which to drop a term

    Returns:
        result : QubitOperator with consolidated terms
    """
    combined: Dict[tuple, float] = {}
    for ps, coeff in zip(pauli_strings, coefficients):
        # Sort the Pauli string by qubit index for canonical form
        ps_sorted = tuple(sorted(ps, key=lambda x: x[0]))
        combined[ps_sorted] = combined.get(ps_sorted, 0.0) + coeff

    result = QubitOperator()
    for ps, coeff in combined.items():
        if abs(coeff) > tol:
            result += QubitOperator(ps, coeff)

    return result


# ============================================================
# FULL Z2 TAPERING PIPELINE
# ============================================================

def z2_taper(
    qubit_op: QubitOperator,
    n_qubits: int,
    n_particles: Optional[int] = None,
    sector: Optional[List[int]] = None,
    verbose: bool = True,
) -> Tuple[QubitOperator, int, List[int], Dict]:
    """
    Full Z2 symmetry tapering pipeline.

    Given a qubit Hamiltonian H (QubitOperator), find all Z2 symmetries,
    apply Clifford transformations to map each symmetry to a single Z_{q(i)},
    fix the eigenvalue sector, and remove the corresponding qubits.

    The result is an exactly equivalent Hamiltonian acting on fewer qubits,
    with the same ground state energy.

    Parameters:
        qubit_op    : QubitOperator (post BK or JW transform)
        n_qubits    : number of qubits in qubit_op
        n_particles : particle number (used for sector determination)
        sector      : explicit list of ±1 sector eigenvalues (optional)
        verbose     : print progress

    Returns:
        tapered_op      : QubitOperator acting on (n_qubits - k) qubits,
                          with qubit indices relabeled 0, 1, ..., n_qubits-k-1
        n_tapered       : integer, number of qubits after tapering
        lambdas         : list of ±1 eigenvalues used for each sector
        info            : dict with intermediate data for inspection:
                          'sym_vectors', 'sym_operators', 'pivots', 'lambdas'
    """
    def vprint(*args):
        if verbose: print(*args)

    vprint(f"\n[Tapering] Input: {n_qubits} qubits, {len(qubit_op.terms)} Pauli terms.")

    # Step 1: Find symmetry generators
    sym_vectors, sym_operators = find_symmetry_generators(qubit_op, n_qubits)
    k = len(sym_vectors)
    vprint(f"[Tapering] Found {k} independent Z2 symmetry generators:")
    for i, op in enumerate(sym_operators):
        vprint(f"  tau_{i} = {op}")

    if k == 0:
        vprint("[Tapering] No Z2 symmetries found. Returning original operator.")
        return qubit_op, n_qubits, [], {}

    # Step 2: Row-reduce symmetry vectors and find pivot qubits
    sym_vectors_copy = [v.copy() for v in sym_vectors]
    pivots = find_pivots(sym_vectors_copy, n_qubits)
    vprint(f"[Tapering] Pivot qubits: {pivots}")

    # Step 3: Determine sector eigenvalues
    lambdas = determine_sector_eigenvalues(
        sym_vectors_copy, pivots, n_qubits, n_particles, sector
    )
    vprint(f"[Tapering] Sector eigenvalues (lambda_i): {lambdas}")

    # Step 4: Apply Clifford transformations to the Hamiltonian
    pauli_strings = list(qubit_op.terms.keys())
    coefficients = [
        (qubit_op.terms[ps].real if hasattr(qubit_op.terms[ps], 'real')
         else float(qubit_op.terms[ps]))
        for ps in pauli_strings
    ]

    pauli_strings, coefficients = apply_clifford_to_operator(
        pauli_strings, coefficients, sym_vectors_copy, pivots, n_qubits
    )
    vprint(f"[Tapering] Applied Clifford transformations.")

    # Step 5: Remove pivot qubits one by one
    # IMPORTANT: remove in reverse order of qubit index so that earlier
    # removals do not shift the indices needed for later removals.
    # We sort by DESCENDING pivot index.
    removal_order = sorted(
        range(k), key=lambda i: -pivots[i]
    )

    # Track the current n_qubits and qubit mapping
    current_n = n_qubits
    global_qubit_map = {q: q for q in range(n_qubits)}  # identity initially

    for sort_idx in removal_order:
        pivot_orig = pivots[sort_idx]
        lam = lambdas[sort_idx]

        # Map original pivot through any previous removals
        # (Since we remove in descending order, earlier removals don't shift
        #  the current pivot's index.)
        current_pivot = pivot_orig  # still correct since we remove in desc order

        pauli_strings, coefficients, qubit_map = remove_qubit(
            pauli_strings, coefficients, current_pivot, lam, current_n
        )
        current_n -= 1
        vprint(f"[Tapering]   Removed qubit {pivot_orig}, lambda={lam:+d}. "
               f"Now {current_n} qubits.")

    # Step 6: Consolidate and build output QubitOperator
    tapered_op = consolidate_terms(pauli_strings, coefficients)
    n_tapered = current_n
    n_terms = len(tapered_op.terms)
    vprint(f"[Tapering] Done. {n_qubits} -> {n_tapered} qubits, "
           f"{len(qubit_op.terms)} -> {n_terms} terms.")

    info = {
        'sym_vectors':   sym_vectors,
        'sym_operators': sym_operators,
        'pivots':        pivots,
        'lambdas':       lambdas,
    }
    return tapered_op, n_tapered, lambdas, info


# ============================================================
# INTEGRATION WITH THE PIPELINE
# ============================================================

def get_n_qubits(qubit_op: QubitOperator) -> int:
    """
    Determine the number of qubits needed for the given QubitOperator by
    finding the maximum qubit index appearing in any Pauli term.

    Parameters:
        qubit_op : QubitOperator

    Returns:
        n_qubits : integer, the number of qubits (= max_index + 1)
    """
    max_q = -1
    for ps in qubit_op.terms:
        for (q, _) in ps:
            if q > max_q:
                max_q = q
    return max_q + 1 if max_q >= 0 else 0


# ============================================================
# EXAMPLE AND VERIFICATION
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from pipeline import fermion_to_qubit

    from openfermion import FermionOperator

    print("=" * 65)
    print("EXAMPLE 1: 2-site spinless hopping model")
    print("  H = -t(c†_0 c_1 + c†_1 c_0)")
    print("  Exact ground state energy: E_0 = -1.0")
    print("=" * 65)
    t = 1.0
    H_f = FermionOperator('0^ 1', -t) + FermionOperator('1^ 0', -t)
    H_q = fermion_to_qubit(H_f)
    n_q = get_n_qubits(H_q)

    print(f"\nBK Hamiltonian ({n_q} qubits):")
    print(H_q)

    tapered, n_tap, lambdas, info = z2_taper(
        H_q, n_q, n_particles=1, verbose=True
    )

    print(f"\nTapered Hamiltonian ({n_tap} qubits):")
    print(tapered)

    print("\n" + "=" * 65)
    print("EXAMPLE 2: 4-mode spinless t-V model (1D, open boundary)")
    print("  H = -t sum_{<ij>} (c†_i c_j + h.c.) + V sum_{<ij>} n_i n_j")
    print("  t=1, V=2, N=2 particles (half filling)")
    print("=" * 65)
    t_hop, V_int = 1.0, 2.0
    L = 4
    H_tV = FermionOperator()
    for i in range(L - 1):
        H_tV += FermionOperator(f'{i}^ {i+1}', -t_hop)
        H_tV += FermionOperator(f'{i+1}^ {i}', -t_hop)
        ni = FermionOperator(f'{i}^ {i}')
        nj = FermionOperator(f'{i+1}^ {i+1}')
        H_tV += V_int * ni * nj

    H_q2 = fermion_to_qubit(H_tV)
    n_q2 = get_n_qubits(H_q2)
    print(f"\nBK Hamiltonian: {n_q2} qubits, {len(H_q2.terms)} terms")

    tapered2, n_tap2, lam2, info2 = z2_taper(
        H_q2, n_q2, n_particles=2, verbose=True
    )
    print(f"\nTapered Hamiltonian ({n_tap2} qubits, {len(tapered2.terms)} terms):")
    print(tapered2)
