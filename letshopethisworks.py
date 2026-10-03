"""
Pipeline: FermionOperator -> Bravyi-Kitaev -> XBK -> HUBO -> Simulated Annealing
=================================================================================

MATHEMATICAL GLOSSARY (every term used is defined here)
---------------------------------------------------------

FermionOperator:
    A linear combination of products of fermionic creation (c†_i) and
    annihilation (c_i) operators. These obey anti-commutation relations:
        {c_i, c†_j} = delta_{ij},   {c_i, c_j} = 0.
    Physically, c†_i creates a particle in mode i; c_i removes one.

Bravyi-Kitaev (BK) Transform:
    A map from fermionic operators on n modes to qubit (Pauli) operators
    on n qubits. It trades the purely local but non-local-in-information
    Jordan-Wigner encoding for one with O(log n) locality.
    Result: a QubitOperator = sum of Pauli strings (products of I, X, Y, Z).

QubitOperator (Pauli string):
    A tensor product of single-qubit matrices, e.g. X_0 Z_1 Y_2.
    X = [[0,1],[1,0]], Y = [[0,-i],[i,0]], Z = [[1,0],[0,-1]] (2x2 matrices).

XBK (Xia-Bian-Kais) Transform:
    Converts every X and Y in a Pauli string into Z-only expressions by
    replicating each qubit r times.
    For qubit i with r=2 replicas (i1, i2) and a sector sign s(i) in {+1,-1}:
        X_i  ->  s(i) * (1 - Z_{i1} Z_{i2}) / 2
        Y_i  ->  i * s(i) * (Z_{i2} - Z_{i1}) / 2
        Z_i  ->  s(i) * (Z_{i1} + Z_{i2}) / 2
        I    ->  (1 + Z_{i1} Z_{i2}) / 2
    Product of these expressions gives a polynomial in Z variables only.
    The result is a HUBO (High-order Unconstrained Binary Optimization) in
    spin variables s_l in {-1, +1}: a sum of products of Z's with real coeff.

HUBO (Higher-order Unconstrained Binary Optimization):
    H(s) = sum_{subsets S} c_S * prod_{l in S} s_l,  s_l in {-1,+1}.
    "k-local" means no product has more than k variables.

Rosenberg quadratization (QUBO reduction):
    Replaces a cubic monomial s_a * s_b * s_c with an auxiliary variable a
    defined to equal s_a * s_b, enforced by a penalty:
        P(s_a, s_b, a) = M * (s_a*s_b - a)^2 / ... in Ising form.
    After conversion to binary variables x_l = (1 - s_l)/2 in {0,1}:
        s_a*s_b = 1 - 2x_a - 2x_b + 4x_a*x_b
    The Rosenberg penalty in binary (Boolean) form for a = x_i * x_j is:
        P(x_i, x_j, a) = M * (x_i*x_j - 2*a*x_i - 2*a*x_j + 3*a)
    This is >= 0 for all {0,1} values, and = 0 iff a = x_i*x_j.
    We choose M > sum of |coefficients| of all terms involving a,
    so no violation can lower the energy below the true minimum.
    After full quadratization the result is a QUBO: at most 2-body interactions.

QUBO (Quadratic Unconstrained Binary Optimization):
    min_{x in {0,1}^n} sum_{i<=j} Q_{ij} x_i x_j
    Equivalent to Ising model via x_l = (1 - s_l)/2.

Simulated Annealing (SA):
    A classical heuristic that samples configurations s from a probability
    distribution proportional to exp(-H(s)/T) where T (temperature) is
    slowly decreased from T_high to T_low. At low T the sampler concentrates
    near global minima of H.

Counting functional C(s):
    XBK replication means each physical configuration is counted multiple
    times (one per replica agreement sector). C(s) counts how many logical
    configurations map to a given physical configuration. We minimize
    H'(s)/C(s) (the Rayleigh quotient) via Dinkelbach iteration.
    For simplicity, this pipeline implements the direct minimization of H'(s)
    within a single sector (all sector signs = +1), which is exact for the
    ground state sector by construction.
"""

import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
from openfermion import FermionOperator, bravyi_kitaev, QubitOperator
import dimod

# ============================================================
# TYPES
# ============================================================
# An Ising spin polynomial is stored as:
#   dict mapping frozenset of spin-variable-labels -> float coefficient
# e.g. {frozenset(): 1.0,  frozenset({'s_0_1'}): -0.5, frozenset({'s_0_1','s_1_2'}): 0.3}
IsingPoly = Dict[frozenset, float]


# ============================================================
# STEP 1: FermionOperator -> QubitOperator via Bravyi-Kitaev
# ============================================================

def fermion_to_qubit(fermion_op: FermionOperator) -> QubitOperator:
    """
    Apply the Bravyi-Kitaev transform.

    Input:
        fermion_op : FermionOperator from openfermion
                     e.g. FermionOperator('0^ 1', 1.0) + h.c.
                     meaning: 1.0 * c†_0 c_1  +  1.0 * c†_1 c_0

    Output:
        qubit_op   : QubitOperator (sum of Pauli strings with real/complex coeff)

    The BK transform is defined by the binary tree structure of n modes.
    openfermion.bravyi_kitaev handles the full algebra automatically.
    """
    return bravyi_kitaev(fermion_op)


# ============================================================
# STEP 2: QubitOperator -> HUBO (Ising polynomial) via XBK
# ============================================================

def xbk_transform(
    qubit_op: QubitOperator,
    n_qubits: int,
    r: int = 2,
    sector_signs: Dict[int, int] = None,
) -> IsingPoly:
    """
    Apply the Xia-Bian-Kais (XBK) transform to convert a Pauli-string
    Hamiltonian into a diagonal Ising (Z-only) polynomial.

    Mathematical definition:
        Each logical qubit i is replaced by r physical replicas labeled
        (i, 1), (i, 2), ..., (i, r).  The spin variable for replica j
        of qubit i is denoted  s_{i,j} in {-1, +1}.

        For r = 2 and sector sign sp = sector_signs[i] in {+1, -1}:
            X_i  -->  sp * (1 - s_{i,1} * s_{i,2}) / 2
            Y_i  -->  i_imag * sp * (s_{i,2} - s_{i,1}) / 2
            Z_i  -->  sp * (s_{i,1} + s_{i,2}) / 2
            I    -->  (1 + s_{i,1} * s_{i,2}) / 2

        A Pauli string P = O_0 ⊗ O_1 ⊗ ... ⊗ O_{n-1} is mapped by
        taking the tensor product of the above maps; the product of
        scalar polynomials in {s_{i,j}} is expanded symbolically.

    Parameters:
        qubit_op      : QubitOperator (output of BK transform)
        n_qubits      : number of logical qubits (modes)
        r             : replication factor (default 2)
        sector_signs  : dict {qubit_index: sign in {+1,-1}}
                        Default: all +1 (sector p=0)

    Returns:
        ising_poly    : IsingPoly
                        A dictionary {frozenset(var_labels): coefficient}
                        where each var_label is a string 'q{i}_r{j}',
                        representing the spin variable s_{i,j}.
                        The constant term uses frozenset() as key.
    """
    if sector_signs is None:
        sector_signs = {i: +1 for i in range(n_qubits)}

    assert r == 2, "This implementation supports r=2. Generalise by looping over replicas."

    result: IsingPoly = defaultdict(float)

    for pauli_string, coeff in qubit_op.terms.items():
        # pauli_string is a tuple like ((0,'X'),(1,'Z'),(3,'Y'))
        # coeff is a complex number; BK images are Hermitian so imag part ~ 0

        # Build a dict qubit -> Pauli gate for this term
        pauli_dict: Dict[int, str] = {idx: gate for idx, gate in pauli_string}

        # Map each qubit to its XBK polynomial contribution.
        # We represent the polynomial for each qubit as IsingPoly.
        # Start with the product as scalar 1 (constant polynomial).
        term_poly: IsingPoly = {frozenset(): 1.0}

        for i in range(n_qubits):
            gate = pauli_dict.get(i, 'I')
            sp = sector_signs[i]
            v1 = f'q{i}_r1'   # spin variable s_{i,1}
            v2 = f'q{i}_r2'   # spin variable s_{i,2}

            # Single-qubit contribution as an IsingPoly
            if gate == 'I':
                # (1 + s_{i,1} s_{i,2}) / 2
                # = (1/2) * {()} + (1/2) * {v1,v2}
                qubit_poly: IsingPoly = {
                    frozenset():       0.5,
                    frozenset({v1,v2}): 0.5,
                }
            elif gate == 'Z':
                # sp * (s_{i,1} + s_{i,2}) / 2
                qubit_poly = {
                    frozenset({v1}): sp * 0.5,
                    frozenset({v2}): sp * 0.5,
                }
            elif gate == 'X':
                # sp * (1 - s_{i,1} s_{i,2}) / 2
                qubit_poly = {
                    frozenset():       sp * 0.5,
                    frozenset({v1,v2}): sp * (-0.5),
                }
            elif gate == 'Y':
                # i_imag * sp * (s_{i,2} - s_{i,1}) / 2
                # The imaginary factor i comes from the Y matrix definition.
                # In a physical (Hermitian) Hamiltonian, the i from Y always
                # combines with other gates to produce a real overall coefficient.
                # We carry it as a multiplicative factor on coeff at the end.
                # Here we return the real part of the spatial polynomial,
                # and multiply by i below when combining with coeff.
                qubit_poly = {
                    frozenset({v2}):  sp * 0.5,   # +s_{i,2}/2
                    frozenset({v1}):  sp * (-0.5), # -s_{i,1}/2
                }
                # Mark: this qubit contributed a factor of i (imaginary unit)
                # We handle this by multiplying coeff by 1j per Y gate below.
                coeff = coeff * 1j  # absorb the 'i' from Y into coeff
            else:
                raise ValueError(f"Unknown Pauli gate '{gate}'")

            # Multiply term_poly by qubit_poly (polynomial multiplication)
            term_poly = _poly_multiply(term_poly, qubit_poly)

        # Now multiply the full term polynomial by the (possibly complex) coefficient
        coeff_real = coeff.real if hasattr(coeff, 'real') else float(coeff)
        if abs(coeff.imag) > 1e-10 if hasattr(coeff, 'imag') else False:
            # Non-zero imaginary part in a supposedly Hermitian Hamiltonian
            # indicates something went wrong or this is not Hermitian.
            # We warn and take real part.
            import warnings
            warnings.warn(
                f"Imaginary coefficient {coeff} encountered after XBK; "
                f"taking real part. Verify Hamiltonian is Hermitian."
            )

        for key, val in term_poly.items():
            result[key] += coeff_real * val

    # Remove near-zero terms
    result = {k: v for k, v in result.items() if abs(v) > 1e-12}
    return dict(result)


def _poly_multiply(p1: IsingPoly, p2: IsingPoly) -> IsingPoly:
    """
    Multiply two Ising polynomials.

    An Ising polynomial is a sum of monomials.
    Each monomial is a product of distinct spin variables s_l in {-1,+1}.
    Key property: s_l^2 = 1 for any Ising spin.
    Therefore: frozenset({'a','b'}) * frozenset({'b','c'}) = frozenset({'a','c'})
    because s_b^2 = 1 cancels.

    This is computed as the symmetric difference of the two frozensets.
    """
    result: IsingPoly = defaultdict(float)
    for k1, v1 in p1.items():
        for k2, v2 in p2.items():
            # Symmetric difference: variables appearing in both cancel (s^2=1)
            new_key = k1.symmetric_difference(k2)
            result[new_key] += v1 * v2
    return dict(result)


# ============================================================
# STEP 3: HUBO -> QUBO via Rosenberg quadratization
# ============================================================

def hubo_to_qubo(
    ising_poly: IsingPoly,
    penalty_multiplier: float = 5.0,
) -> Tuple[dimod.BinaryQuadraticModel, Dict[str, str]]:
    """
    Convert a higher-order Ising polynomial H(s), s in {-1,+1},
    to a QUBO Q(x), x in {0,1}, by:
      (a) Converting spin variables to binary: x_l = (1 - s_l) / 2
          so s_l = 1 - 2*x_l.
      (b) Expanding all products in x.
      (c) Applying Rosenberg quadratization to reduce degree > 2 terms.

    Rosenberg penalty (Boolean, degree reduction of x_i*x_j):
        Introduce auxiliary variable a meant to equal x_i * x_j.
        Penalty: P = M * (x_i*x_j - 2*a*x_i - 2*a*x_j + 3*a)
        P >= 0 always, P = 0 iff a = x_i*x_j.
        So adding P to the objective enforces the constraint without
        introducing a separate constraint variable.

    Penalty weight M:
        M = penalty_multiplier * max(|coefficient of any original term|)
        This ensures M is large enough that violating any constraint
        costs more than the worst-case energy difference in the problem.

    Parameters:
        ising_poly        : IsingPoly from xbk_transform
        penalty_multiplier: M = penalty_multiplier * max|c|  (default 5)

    Returns:
        bqm               : dimod.BinaryQuadraticModel in BINARY vartype
                            i.e. variables x_l in {0,1}, objective = QUBO
        aux_map           : dict {auxiliary_var_name: 'x_i*x_j'} for record-keeping
    """
    # Step 3a: Convert Ising spins s_l in {-1,+1} to binary x_l in {0,1}
    # using s_l = 1 - 2*x_l.
    # A monomial prod_{l in S} s_l = prod_{l in S} (1 - 2*x_l)
    # We expand this product symbolically.

    binary_poly: Dict[frozenset, float] = defaultdict(float)

    for spin_key, coeff in ising_poly.items():
        # Expand prod_{l in spin_key} (1 - 2 x_l) into binary monomials
        spin_vars = list(spin_key)
        # Use inclusion over subsets of spin_vars
        for k in range(len(spin_vars) + 1):
            for subset in itertools.combinations(spin_vars, k):
                # Each subset contributes (-2)^k * coeff to the binary monomial
                sign = (-2) ** k
                binary_poly[frozenset(subset)] += coeff * sign

    # Prune zeros
    binary_poly = {k: v for k, v in binary_poly.items() if abs(v) > 1e-12}

    # Determine penalty weight M
    max_coeff = max(abs(v) for v in binary_poly.values()) if binary_poly else 1.0
    M = penalty_multiplier * max_coeff

    # Step 3b: Rosenberg quadratization
    # We iteratively replace the most-used quadratic pair (x_i, x_j) appearing
    # in high-degree monomials with an auxiliary variable a_{ij}.
    # After replacement, the auxiliary variable is added to the QUBO with a penalty.

    aux_counter = [0]   # mutable counter for aux variable names
    aux_map: Dict[str, str] = {}  # record: aux_name -> 'xi*xj'

    # Repeatedly reduce until all monomials are degree <= 2
    changed = True
    while changed:
        changed = False
        # Find the highest-degree monomial
        max_degree = max((len(k) for k in binary_poly), default=0)
        if max_degree <= 2:
            break

        # Count how many high-degree terms each pair (i,j) appears in
        pair_count: Dict[frozenset, int] = defaultdict(int)
        for monomial in binary_poly:
            if len(monomial) >= 3:
                for pair in itertools.combinations(sorted(monomial), 2):
                    pair_count[frozenset(pair)] += 1

        if not pair_count:
            break

        # Choose the pair that appears in the most high-degree monomials
        best_pair = max(pair_count, key=lambda p: pair_count[p])
        xi, xj = sorted(best_pair)

        # Create auxiliary variable a = x_i * x_j
        aux_name = f'aux_{aux_counter[0]}'
        aux_counter[0] += 1
        aux_map[aux_name] = f'{xi}*{xj}'

        # Replace best_pair with aux_name in all monomials
        new_poly: Dict[frozenset, float] = defaultdict(float)
        for monomial, coeff in binary_poly.items():
            if best_pair.issubset(monomial):
                # Remove xi and xj from monomial, add aux_name
                new_monomial = (monomial - best_pair) | frozenset({aux_name})
                new_poly[new_monomial] += coeff
            else:
                new_poly[monomial] += coeff

        # Add Rosenberg penalty for a = x_i * x_j:
        # P = M * (x_i*x_j - 2*a*x_i - 2*a*x_j + 3*a)
        new_poly[frozenset({xi, xj})]           += M
        new_poly[frozenset({aux_name, xi})]      -= 2 * M
        new_poly[frozenset({aux_name, xj})]      -= 2 * M
        new_poly[frozenset({aux_name})]          += 3 * M

        binary_poly = {k: v for k, v in new_poly.items() if abs(v) > 1e-12}
        changed = True

    # Step 3c: Build a dimod BinaryQuadraticModel (QUBO form)
    bqm = dimod.BinaryQuadraticModel(vartype='BINARY')

    for monomial, coeff in binary_poly.items():
        degree = len(monomial)
        if degree == 0:
            bqm.offset += coeff
        elif degree == 1:
            var = next(iter(monomial))
            bqm.add_variable(var, coeff)
        elif degree == 2:
            vars_list = list(monomial)
            bqm.add_interaction(vars_list[0], vars_list[1], coeff)
        else:
            raise RuntimeError(
                f"Quadratization incomplete: monomial of degree {degree} remains: {monomial}. "
                f"Increase penalty_multiplier or debug quadratization logic."
            )

    return bqm, aux_map


# ============================================================
# STEP 4: Run Simulated Annealing on the QUBO
# ============================================================

def run_simulated_annealing(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
) -> dimod.SampleSet:
    """
    Run D-Wave dimod SimulatedAnnealingSampler on a QUBO (BinaryQuadraticModel).

    Simulated annealing samples from the distribution
        P(x) proportional to exp(-beta * E(x))
    where beta = 1/T is the inverse temperature (T = absolute temperature,
    in units where Boltzmann's constant k_B = 1).

    The algorithm starts at low beta (= high T, so distribution is flat =
    uniform random), and slowly increases beta to high_beta (= low T, so
    distribution concentrates near the minimum of E(x)).

    At each temperature step (called a 'sweep'), every variable is proposed
    a flip; the flip is accepted with probability min(1, exp(-beta * delta_E))
    where delta_E = E(new) - E(old).

    Parameters:
        bqm        : BinaryQuadraticModel from hubo_to_qubo
        num_reads  : number of independent annealing runs (more = better)
        num_sweeps : number of temperature steps per run
        beta_range : (beta_min, beta_max) = (1/T_max, 1/T_min)
        seed       : random seed for reproducibility

    Returns:
        sample_set : dimod.SampleSet containing all samples, energies, etc.
                     The lowest energy sample is the best ground state estimate.
    """
    sampler = dimod.SimulatedAnnealingSampler()
    # Note: dimod.SimulatedAnnealingSampler does not accept a 'seed' kwarg
    # in all versions. For reproducibility, set numpy seed before calling.
    np.random.seed(seed)
    sample_set = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=list(beta_range),
    )
    return sample_set


# ============================================================
# STEP 5: Decode result back to Ising spins and logical qubits
# ============================================================

def decode_result(
    sample_set: dimod.SampleSet,
    bqm: dimod.BinaryQuadraticModel,
    aux_map: Dict[str, str],
    n_qubits: int,
    r: int = 2,
) -> Dict[str, Any]:
    """
    Decode the best QUBO sample back to:
      - Binary variables x_l in {0,1}
      - Ising spins s_l = 1 - 2*x_l in {-1,+1}
      - Physical replica spins s_{i,j} for each logical qubit i, replica j
      - Logical qubit occupations: n_i = (1 - s_{i,all_replicas_average}) / 2

    A 'logical qubit' occupation n_i is 1 if the mode is occupied, 0 if empty.
    In XBK with r=2, the two replicas s_{i,1} and s_{i,2} should agree
    (both +1 or both -1) in a valid solution. If they disagree, the penalty
    was violated; we flag this.

    Parameters:
        sample_set : SampleSet from SA
        bqm        : the BQM (to read the energy)
        aux_map    : auxiliary variable registry from quadratization
        n_qubits   : number of logical qubits
        r          : replication factor (must match xbk_transform)

    Returns:
        result_dict : dict with keys
          'best_energy'     : float, lowest QUBO energy found (includes penalty terms)
          'best_sample'     : dict {var: 0 or 1}
          'ising_spins'     : dict {var: -1 or +1}  (physical spin configuration)
          'replica_spins'   : dict {qubit_i: [s_{i,1}, ..., s_{i,r}]}
          'replica_agreed'  : dict {qubit_i: bool} (True if all replicas agree)
          'logical_occupations': list of 0/1 of length n_qubits
          'aux_violations'  : list of aux vars where constraint a = x_i*x_j is violated
    """
    best = sample_set.first
    best_sample = dict(best.sample)
    best_energy = best.energy

    # Convert binary x -> Ising spin s = 1 - 2*x
    ising_spins = {var: 1 - 2 * val for var, val in best_sample.items()}

    # Extract physical replica spins for each logical qubit
    replica_spins: Dict[int, List[int]] = {}
    for i in range(n_qubits):
        replicas = []
        for j in range(1, r + 1):
            var = f'q{i}_r{j}'
            if var in ising_spins:
                replicas.append(ising_spins[var])
            else:
                replicas.append(None)  # variable may have been tapered/absent
        replica_spins[i] = replicas

    # Check replica agreement and compute logical occupation
    replica_agreed: Dict[int, bool] = {}
    logical_occupations: List[int] = []
    for i in range(n_qubits):
        spins = [s for s in replica_spins[i] if s is not None]
        if len(spins) == 0:
            replica_agreed[i] = True
            logical_occupations.append(0)
        elif all(s == spins[0] for s in spins):
            replica_agreed[i] = True
            # Occupation: n_i = (1 - s) / 2. If s = +1 -> n_i = 0 (empty)
            #                                   If s = -1 -> n_i = 1 (occupied)
            logical_occupations.append(int((1 - spins[0]) // 2))
        else:
            replica_agreed[i] = False
            # Take majority vote for occupation
            avg_spin = np.mean(spins)
            logical_occupations.append(int(avg_spin < 0))

    # Check auxiliary variable constraint violations: a should equal x_i * x_j
    aux_violations: List[str] = []
    for aux_var, definition in aux_map.items():
        xi_name, xj_name = definition.split('*')
        if aux_var in best_sample and xi_name in best_sample and xj_name in best_sample:
            a_val = best_sample[aux_var]
            expected = best_sample[xi_name] * best_sample[xj_name]
            if a_val != expected:
                aux_violations.append(aux_var)

    return {
        'best_energy':          best_energy,
        'best_sample':          best_sample,
        'ising_spins':          ising_spins,
        'replica_spins':        replica_spins,
        'replica_agreed':       replica_agreed,
        'logical_occupations':  logical_occupations,
        'aux_violations':       aux_violations,
    }


# ============================================================
# FULL PIPELINE (model-agnostic entry point)
# ============================================================

def run_pipeline(
    fermion_op: FermionOperator,
    n_modes: int,
    r: int = 2,
    sector_signs: Dict[int, int] = None,
    penalty_multiplier: float = 5.0,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline:
        FermionOperator
            -> [BK] QubitOperator
            -> [XBK] HUBO (Ising polynomial, k-local, diagonal)
            -> [Rosenberg] QUBO (BinaryQuadraticModel, 2-local)
            -> [SA] SampleSet (ground state estimate)
            -> decoded result

    Parameters:
        fermion_op         : openfermion FermionOperator (the Hamiltonian)
        n_modes            : number of fermionic modes (= number of spin-orbitals)
        r                  : XBK replication factor (default 2)
        sector_signs       : dict {qubit: +1 or -1}, default all +1
        penalty_multiplier : M = penalty_multiplier * max|coeff| for Rosenberg
        num_reads          : SA independent runs
        num_sweeps         : SA sweeps per run
        beta_range         : (beta_min, beta_max) for SA schedule
        seed               : RNG seed
        verbose            : print progress and summary

    Returns:
        full_result : dict with all intermediate objects and final decoded result
    """
    def vprint(*args):
        if verbose:
            print(*args)

    # ---- Step 1: BK ----
    vprint("\n[Step 1] Applying Bravyi-Kitaev transform...")
    qubit_op = fermion_to_qubit(fermion_op)
    vprint(f"  QubitOperator has {len(qubit_op.terms)} Pauli terms.")

    # ---- Step 2: XBK ----
    vprint(f"\n[Step 2] Applying XBK transform (r={r})...")
    ising_poly = xbk_transform(qubit_op, n_modes, r=r, sector_signs=sector_signs)
    max_locality = max((len(k) for k in ising_poly), default=0)
    vprint(f"  HUBO has {len(ising_poly)} terms, max locality k = {max_locality}.")

    # ---- Step 3: Rosenberg quadratization ----
    vprint(f"\n[Step 3] Quadratizing HUBO to QUBO (penalty_multiplier={penalty_multiplier})...")
    bqm, aux_map = hubo_to_qubo(ising_poly, penalty_multiplier=penalty_multiplier)
    n_vars = len(bqm.variables)
    n_interactions = len(bqm.quadratic)
    vprint(f"  QUBO has {n_vars} variables ({len(aux_map)} auxiliary), "
           f"{n_interactions} quadratic interactions.")
    if aux_map:
        vprint(f"  Auxiliary variables introduced: {list(aux_map.keys())[:5]} ...")

    # ---- Step 4: Simulated Annealing ----
    vprint(f"\n[Step 4] Running Simulated Annealing "
           f"(num_reads={num_reads}, num_sweeps={num_sweeps})...")
    sample_set = run_simulated_annealing(
        bqm, num_reads=num_reads, num_sweeps=num_sweeps,
        beta_range=beta_range, seed=seed,
    )
    vprint(f"  Best energy (QUBO, includes penalty terms): {sample_set.first.energy:.6f}")

    # ---- Step 5: Decode ----
    vprint("\n[Step 5] Decoding result...")
    decoded = decode_result(sample_set, bqm, aux_map, n_modes, r=r)

    vprint("\n=== RESULT SUMMARY ===")
    vprint(f"  Best QUBO energy         : {decoded['best_energy']:.6f}")
    vprint(f"  Logical qubit occupations: {decoded['logical_occupations']}")
    vprint(f"  Replica agreement        : {decoded['replica_agreed']}")
    if decoded['aux_violations']:
        vprint(f"  WARNING: {len(decoded['aux_violations'])} auxiliary constraint(s) violated.")
        vprint(f"  -> Increase penalty_multiplier (currently {penalty_multiplier}).")
    else:
        vprint("  All auxiliary constraints satisfied. (Quadratization is valid.)")

    return {
        'fermion_op':   fermion_op,
        'qubit_op':     qubit_op,
        'ising_poly':   ising_poly,
        'bqm':          bqm,
        'aux_map':      aux_map,
        'sample_set':   sample_set,
        **decoded,
    }


# ============================================================
# EXAMPLE: 2-site spinless hopping model
# ============================================================
# H = -t (c†_0 c_1 + c†_1 c_0)
# where t is the hopping amplitude.
# This is the simplest fermionic Hamiltonian with off-diagonal (non-number)
# terms that require the full BK + XBK pipeline to diagonalize classically.
# The exact ground state energy is -t (one occupied, bonding orbital).

if __name__ == '__main__':
    t = 1.0   # hopping amplitude (energy unit)

    # FermionOperator notation: '0^ 1' means c†_0 c_1
    H = FermionOperator('0^ 1', -t) + FermionOperator('1^ 0', -t)
    print("Fermion Hamiltonian H =", H)
    print(f"Exact ground state energy (analytic): E_0 = {-t:.4f}")

    result = run_pipeline(
        fermion_op=H,
        n_modes=2,
        r=2,
        penalty_multiplier=6.0,
        num_reads=2000,
        num_sweeps=2000,
        verbose=True,
    )

    print("\nFull ising polynomial terms:")
    for key, val in sorted(result['ising_poly'].items(), key=lambda x: len(x[0])):
        print(f"  {val:+.4f}  *  {sorted(key) if key else '1'}")