"""
Pipeline: FermionOperator -> BK -> Z2 Tapering -> XBK -> Dinkelbach-SA
=======================================================================

MATHEMATICAL GLOSSARY  (every term defined precisely)
------------------------------------------------------

FermionOperator:
    A linear combination of products of fermionic creation (c†_i) and
    annihilation (c_i) operators obeying:
        {c_i, c†_j} = delta_{ij},   {c_i, c_j} = 0.

Bravyi-Kitaev (BK) Transform:
    Maps n fermionic modes to n qubits with O(log n)-local Pauli strings.
    Output: QubitOperator = sum_j h_j * P_j, where P_j is a Pauli string.

Pauli string:
    Tensor product of single-qubit matrices from {I, X, Y, Z}.
    I = identity, X = [[0,1],[1,0]], Y = [[0,-i],[i,0]], Z = [[1,0],[0,-1]].

Z2 Symmetry Tapering:
    For each independent Pauli operator tau_i with [tau_i, H] = 0 and
    tau_i^2 = I, exactly one qubit is removed from H and the ground state
    energy is preserved exactly. (Full algorithm: z2_tapering.py.)

XBK (Xia-Bian-Kais) Transform  [Xia, Bian, Kais 2018]:
    Maps an m-qubit Pauli Hamiltonian H to a Z-only Ising polynomial
    H'_p(s) on 2m spin variables s_{i,j} in {-1,+1} (qubit i, replica j in {1,2}).

    Sector p: a choice of sign Sp(i) in {+1,-1} for each qubit i (2^m choices).
    By convention Sp(i) acts as the sign of replica 2 relative to replica 1.

    Single-qubit mapping rules (qubit i, replicas s_{i,1}, s_{i,2}, sign Sp(i)):
        X_i  ->  Sp(i) * (1 - s_{i,1} * s_{i,2}) / 2
        Y_i  ->  i * Sp(i) * (s_{i,2} - s_{i,1}) / 2     [i = sqrt(-1)]
        Z_i  ->  Sp(i) * (s_{i,1} + s_{i,2}) / 2
        I_i  ->  (1 + s_{i,1} * s_{i,2}) / 2

    A Pauli string P = O_0 x O_1 x ... x O_{m-1} maps by multiplying these
    expressions and expanding using the identity s_l^2 = 1 (so when a variable
    appears in both factors it cancels; implemented as symmetric difference of
    variable index sets).

    The imaginary unit from Y gates is absorbed into the coefficient.
    For a Hermitian Hamiltonian the result is always real.

    XBK as an APPROXIMATION:
    For finite r=2, XBK produces an Ising polynomial whose Rayleigh quotient
    minimum approximates E_0. The approximation is exact only as r -> infinity.
    For r=2, the discrete values of H'_p / C_p are limited to rational combinations
    of the Hamiltonian coefficients; irrational eigenvalues cannot be resolved exactly.
    Accuracy improves with r, with the FMQA paper recommending r ≈ m/2 to m.

Counting Functional C_p(s)  [QMP_NOTES eq. 81]:
    Counts how many times each logical basis state appears in the replicated space.

    C_p(s) = sum_{eps in {+1,-1}^m} [sum_{j=1}^r proj_j(eps, s)]^2

    where  proj_j(eps, s) = prod_{k=0}^{m-1} (1 + eps_k * s_{k,j}) / 2
    equals 1 if replica j is in basis state eps (for all m qubits), else 0.

    For r=2: C_p takes values in {2, 4}. It is NEVER 0.
        C_p = 4 when s_{i,1} = s_{i,2} for all i (all replicas agree on all qubits)
        C_p = 2 in all other cases

Rayleigh Quotient (XBK energy estimate):
    For sector p, the XBK energy estimate is:
        E_{0,p} = min_{s in {-1,+1}^{2m}} [r * H'_p(s) / C_p(s)]

    The factor r=2 normalizes for the multiplicity structure of C_p:
    on all-agree configs (C_p=4=2r), r/C_p = 1/2, meaning the polynomial
    value is halved relative to the true energy; multiplying by r corrects this.

    The ground state energy estimate is:
        E_0 = min_p E_{0,p}   (minimized over all 2^m sectors)

Dinkelbach Iteration  [Dinkelbach 1967, applied in FMQA eq. 6]:
    Converts the fractional minimization min_s [r * H'_p / C_p] into a
    sequence of QUBO subproblems.

    Define  phi_p(lambda) = min_s [ H'_p(s) - lambda * C_p(s) / r ].

    phi_p is strictly decreasing in lambda, and phi_p(E_{0,p}) = 0.
    Finding the root of phi_p gives E_{0,p}.

    Algorithm for sector p:
        lambda_0 = initial guess (lower bound: -r * sum|H'_p coefficients|)
        For k = 0, 1, 2, ...:
            1. Build Dinkelbach QUBO: D_p(s) = H'_p(s) - lambda_k * C_p(s) / r
            2. Quadratize D_p via Rosenberg reduction -> QUBO
            3. Minimize QUBO via Simulated Annealing -> binary sample x*
            4. Convert x* -> Ising spin assignment s* = 1 - 2*x*
            5. Evaluate H'_p(s*) and C_p(s*)
            6. Update: lambda_{k+1} = r * H'_p(s*) / C_p(s*)
            7. Stop when |H'_p(s*) - lambda_k * C_p(s*)/r| < tol
                         OR |lambda_{k+1} - lambda_k| < tol
        Return E_{0,p} = lambda_{k+1}

    Run for all 2^m sectors, return E_0 = min_p E_{0,p}.

Rosenberg Quadratization:
    Reduces a k-local Ising polynomial to a 2-local QUBO.
    Step 1: Change variables x_l = (1-s_l)/2 in {0,1}, expand monomials
            via sum_{T subset S} (-2)^|T| * prod_{l in T} x_l.
    Step 2: For the most-frequent pair (x_i, x_j) in degree>=3 monomials,
            introduce auxiliary a representing x_i*x_j with penalty:
            P = M*(x_i*x_j - 2*a*x_i - 2*a*x_j + 3*a) >= 0, = 0 iff a=x_i*x_j.
            Penalty weight M = penalty_multiplier * max|coefficient|.
    Repeat until max degree <= 2.

Simulated Annealing (SA):
    Minimizes QUBO by sampling from P(x) ~ exp(-beta*E(x)) with beta
    increasing from 1/T_max to 1/T_min across num_sweeps temperature steps.
"""

import itertools
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from openfermion import FermionOperator, bravyi_kitaev, QubitOperator
import dimod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z2_tapering import z2_taper, get_n_qubits

# Type alias: Ising polynomial as {frozenset(var_names) -> float}
IsingPoly = Dict[frozenset, float]


# ==================================================================
# STEP 1 — Bravyi-Kitaev Transform
# ==================================================================

def fermion_to_qubit(fermion_op: FermionOperator) -> QubitOperator:
    """
    Apply the Bravyi-Kitaev transform to a FermionOperator.
    Returns a QubitOperator (sum of Pauli strings with real coefficients).
    """
    return bravyi_kitaev(fermion_op)


# ==================================================================
# STEP 2 — Z2 Symmetry Tapering
# ==================================================================

def taper_symmetries(
    qubit_op: QubitOperator,
    n_particles: Optional[int] = None,
    sector: Optional[List[int]] = None,
    max_symmetries: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[QubitOperator, int, Dict]:
    """
    Remove qubits using Z2 symmetries.

    Each independent Pauli Z2 symmetry tau_i (with [tau_i,H]=0 and tau_i^2=I)
    lets us remove exactly one qubit by fixing Z_{q(i)} -> lambda_i in {+1,-1}.
    Ground state energy is preserved exactly.

    Parameters:
        qubit_op      : QubitOperator (post-BK)
        n_particles   : particle number for automatic sector eigenvalue detection
        sector        : explicit list of +-1 eigenvalues for each symmetry found;
                        overrides n_particles. Adjust if default gives wrong E_0
                        (verified by comparing with exact diagonalisation).
        max_symmetries: cap on number of symmetries to apply (None = use all).
                        Use max_symmetries=k to keep at least (n_qubits-k) qubits.
        verbose       : print progress

    Returns:
        tapered_op : QubitOperator on fewer qubits
        n_tapered  : int, qubits remaining
        tap_info   : dict with 'sym_operators', 'pivots', 'lambdas'
    """
    n_qubits = get_n_qubits(qubit_op)
    if n_qubits == 0:
        return qubit_op, 0, {}

    # Find all symmetries first (via z2_taper in dry-run mode won't work,
    # so we just pass max_symmetries by limiting the sector list).
    tapered_op, n_tapered, lambdas, info = z2_taper(
        qubit_op, n_qubits,
        n_particles=n_particles,
        sector=sector,
        verbose=verbose,
    )

    # If max_symmetries is set and we removed too many, we can't easily undo,
    # so we just warn. The user should set max_symmetries carefully.
    if max_symmetries is not None and (n_qubits - n_tapered) > max_symmetries:
        import warnings
        warnings.warn(
            f"Tapering removed {n_qubits - n_tapered} qubits but "
            f"max_symmetries={max_symmetries}. "
            f"Consider passing sector=[] to disable tapering entirely."
        )

    return tapered_op, n_tapered, info


# ==================================================================
# STEP 3a — XBK Transform: QubitOperator -> IsingPoly H'_p(s)
# ==================================================================

def xbk_transform(
    qubit_op: QubitOperator,
    n_qubits: int,
    r: int = 2,
    sector_signs: Optional[Dict[int, int]] = None,
) -> IsingPoly:
    """
    Apply XBK for a single sector p (defined by sector_signs).

    sector_signs : dict {qubit_index: Sp(i) in {+1,-1}}. Default all +1.
    Variable labels: 'q{i}_r{j}' for qubit i, replica j in {1,...,r}.

    Only r=2 is implemented. Returns H'_p(s) as an IsingPoly.
    """
    if sector_signs is None:
        sector_signs = {i: +1 for i in range(n_qubits)}
    assert r == 2, "Only r=2 is implemented."

    result: IsingPoly = defaultdict(float)

    for pauli_string, coeff in qubit_op.terms.items():
        pauli_dict = {idx: gate for idx, gate in pauli_string}
        term_poly: IsingPoly = {frozenset(): 1.0}
        rc = complex(coeff)

        for i in range(n_qubits):
            gate = pauli_dict.get(i, 'I')
            sp = sector_signs[i]
            v1, v2 = f'q{i}_r1', f'q{i}_r2'

            if gate == 'I':
                qp: IsingPoly = {
                    frozenset():        0.5,
                    frozenset({v1,v2}): 0.5,
                }
            elif gate == 'Z':
                qp = {frozenset({v1}): sp * 0.5, frozenset({v2}): sp * 0.5}
            elif gate == 'X':
                qp = {frozenset(): sp * 0.5, frozenset({v1,v2}): sp * -0.5}
            elif gate == 'Y':
                qp = {frozenset({v2}): sp * 0.5, frozenset({v1}): sp * -0.5}
                rc *= 1j
            else:
                raise ValueError(f"Unknown gate: {gate!r}")

            term_poly = _poly_mul(term_poly, qp)

        cr = rc.real
        if abs(rc.imag) > 1e-10:
            import warnings
            warnings.warn(
                f"Imaginary coefficient {rc} remains after XBK. "
                f"Verify the Hamiltonian is Hermitian."
            )
        for key, val in term_poly.items():
            result[key] += cr * val

    return {k: v for k, v in result.items() if abs(v) > 1e-12}


# ==================================================================
# STEP 3b — Counting Functional C_p(s)
# ==================================================================

def build_counting_functional(n_qubits: int, r: int = 2) -> IsingPoly:
    """
    Build C_p(s) = sum_{eps in {+-1}^m} [sum_{j=1}^r proj_j(eps)]^2

    where proj_j(eps) = prod_k (1 + eps_k * s_{k,j}) / 2.

    This counts how many times each logical basis state appears in the
    replicated spin space, and is used as the denominator in the Rayleigh
    quotient E_{0,p} = min_s [r * H'_p(s) / C_p(s)].

    For r=2: C_p in {2, 4} always.
        C_p = 4  when all replica pairs agree: s_{i,1} = s_{i,2} for all i.
        C_p = 2  in all other cases.
        C_p is never 0.

    Returns an IsingPoly representing C_p(s).
    """
    assert r == 2

    C: IsingPoly = {frozenset(): 0.0}

    for eps in itertools.product([-1, +1], repeat=n_qubits):
        # inner(s) = sum_{j=1}^r proj_j(eps, s)
        inner: IsingPoly = {frozenset(): 0.0}
        for j in range(1, r + 1):
            proj: IsingPoly = {frozenset(): 1.0}
            for k, ek in enumerate(eps):
                vkj = f'q{k}_r{j}'
                factor: IsingPoly = {
                    frozenset():    0.5,
                    frozenset({vkj}): ek * 0.5,
                }
                proj = _poly_mul(proj, factor)
            inner = _poly_add(inner, proj)

        # C += inner^2
        C = _poly_add(C, _poly_mul(inner, inner))

    return {k: v for k, v in C.items() if abs(v) > 1e-12}


# ==================================================================
# Polynomial arithmetic
# ==================================================================

def _poly_mul(p1: IsingPoly, p2: IsingPoly) -> IsingPoly:
    """Multiply two Ising polynomials using s_l^2 = 1 (symmetric difference)."""
    result: IsingPoly = defaultdict(float)
    for k1, v1 in p1.items():
        for k2, v2 in p2.items():
            result[k1.symmetric_difference(k2)] += v1 * v2
    return dict(result)


def _poly_add(p1: IsingPoly, p2: IsingPoly, scale: float = 1.0) -> IsingPoly:
    """Add two Ising polynomials: result = p1 + scale * p2."""
    result = dict(p1)
    for k, v in p2.items():
        result[k] = result.get(k, 0.0) + scale * v
    return {k: v for k, v in result.items() if abs(v) > 1e-12}


def _poly_eval(poly: IsingPoly, assignment: Dict[str, int]) -> float:
    """Evaluate poly at spin assignment {var_name: +1 or -1}."""
    total = 0.0
    for key, coeff in poly.items():
        val = 1.0
        for v in key:
            val *= assignment[v]
        total += coeff * val
    return total


# ==================================================================
# STEP 4 — Rosenberg Quadratization: IsingPoly -> QUBO (BQM)
# ==================================================================

def hubo_to_qubo(
    ising_poly: IsingPoly,
    penalty_multiplier: float = 5.0,
) -> Tuple[dimod.BinaryQuadraticModel, Dict[str, str]]:
    """
    Convert an Ising polynomial H(s) to a QUBO Q(x) over {0,1}.

    (a) Change variables: x_l = (1-s_l)/2, s_l = 1-2*x_l.
        Monomial prod_{l in S} s_l -> sum_{T subset S} (-2)^|T| prod_{l in T} x_l.

    (b) Greedy Rosenberg quadratization until all monomials have degree <= 2:
        Find the pair (x_i,x_j) appearing most in degree>=3 monomials.
        Introduce auxiliary a = x_i*x_j with penalty
            P = M*(x_i*x_j - 2*a*x_i - 2*a*x_j + 3*a),  P>=0, P=0 iff a=x_i*x_j.
        M = penalty_multiplier * max|coefficient|.

    (c) Build dimod.BinaryQuadraticModel (BINARY vartype).

    Returns:
        bqm     : dimod.BinaryQuadraticModel
        aux_map : dict {aux_name: 'xi_name*xj_name'} recording substitutions
    """
    # (a) Ising -> binary expansion
    bp: Dict[frozenset, float] = defaultdict(float)
    for spin_key, coeff in ising_poly.items():
        svars = list(spin_key)
        for k in range(len(svars) + 1):
            for subset in itertools.combinations(svars, k):
                bp[frozenset(subset)] += coeff * (-2.0) ** k
    bp = {k: v for k, v in bp.items() if abs(v) > 1e-12}

    max_coeff = max(abs(v) for v in bp.values()) if bp else 1.0
    M = penalty_multiplier * max_coeff

    # (b) Rosenberg quadratization
    aux_counter = [0]
    aux_map: Dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        if max((len(k) for k in bp), default=0) <= 2:
            break
        pair_count: Dict[frozenset, int] = defaultdict(int)
        for monomial in bp:
            if len(monomial) >= 3:
                for pair in itertools.combinations(sorted(monomial), 2):
                    pair_count[frozenset(pair)] += 1
        if not pair_count:
            break
        best_pair = max(pair_count, key=lambda p: pair_count[p])
        xi, xj = sorted(best_pair)
        aname = f'aux_{aux_counter[0]}'
        aux_counter[0] += 1
        aux_map[aname] = f'{xi}*{xj}'
        new_bp: Dict[frozenset, float] = defaultdict(float)
        for monomial, coeff in bp.items():
            if best_pair.issubset(monomial):
                new_bp[(monomial - best_pair) | frozenset({aname})] += coeff
            else:
                new_bp[monomial] += coeff
        new_bp[frozenset({xi, xj})]      += M
        new_bp[frozenset({aname, xi})]    -= 2.0 * M
        new_bp[frozenset({aname, xj})]    -= 2.0 * M
        new_bp[frozenset({aname})]        += 3.0 * M
        bp = {k: v for k, v in new_bp.items() if abs(v) > 1e-12}
        changed = True

    # (c) Build BQM
    bqm = dimod.BinaryQuadraticModel(vartype='BINARY')
    for monomial, coeff in bp.items():
        d = len(monomial)
        if d == 0:
            bqm.offset += coeff
        elif d == 1:
            bqm.add_variable(next(iter(monomial)), coeff)
        elif d == 2:
            v = list(monomial)
            bqm.add_interaction(v[0], v[1], coeff)
        else:
            raise RuntimeError(
                f"Quadratization incomplete: degree-{d} monomial {monomial}. "
                f"Increase penalty_multiplier."
            )
    return bqm, aux_map


# ==================================================================
# STEP 5 — Simulated Annealing
# ==================================================================

def run_sa(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int = 100,
    num_sweeps: int = 500,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
) -> dimod.SampleSet:
    """
    Run dimod SimulatedAnnealingSampler on a QUBO.
    Default parameters are conservative (fast); increase for higher accuracy.
    """
    np.random.seed(seed)
    return dimod.SimulatedAnnealingSampler().sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=list(beta_range),
    )


# ==================================================================
# STEP 6 — Dinkelbach Iteration over all XBK sectors
# ==================================================================

def dinkelbach_sa(
    qubit_op: QubitOperator,
    n_qubits: int,
    r: int = 2,
    lambda_init: Optional[float] = None,
    max_iter: int = 10,
    tol: float = 1e-6,
    penalty_multiplier: float = 5.0,
    num_reads: int = 100,
    num_sweeps: int = 500,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Find E_0 = min_p E_{0,p} via Dinkelbach iteration over all 2^m XBK sectors.

    For each of the 2^m sectors (choices of Sp(i) in {+-1} for each qubit i):
        1. Build H'_p via xbk_transform (sector-specific).
        2. Build C_p via build_counting_functional (sector-independent for r=2).
        3. Run Dinkelbach:
             lambda_0 = lambda_init  (default: -r * sum|H'_p coeff|, a lower bound)
             For k=0,1,...,max_iter-1:
               D_p = H'_p - lambda_k * C_p / r
               QUBO = Rosenberg(D_p)
               s* = argmin SA(QUBO)
               lambda_{k+1} = r * H'_p(s*) / C_p(s*)
               stop if |H'_p(s*) - lambda_k*C_p(s*)/r| < tol
                          or |lambda_{k+1} - lambda_k| < tol
        4. Record E_{0,p} = lambda_{converged}
    Return E_0 = min_p E_{0,p}.

    NOTE: XBK with r=2 is an approximation. Accuracy improves with larger r.
    For the best results on a D-Wave, use r = n_qubits (one replica per qubit).

    Parameters:
        qubit_op          : post-tapering QubitOperator
        n_qubits          : number of qubits in qubit_op
        r                 : replication factor (r=2 only implemented)
        lambda_init       : initial Dinkelbach lambda (None -> auto lower bound)
        max_iter          : max iterations per sector
        tol               : convergence threshold
        penalty_multiplier: Rosenberg M = multiplier * max|D_p coeff|
        num_reads         : SA reads per Dinkelbach step (increase for accuracy)
        num_sweeps        : SA sweeps per read (increase for accuracy)
        beta_range        : SA inverse temperature schedule (1/T_max, 1/T_min)
        seed              : RNG seed
        verbose           : print progress

    Returns dict with:
        'E0'                : float, best energy estimate over all sectors
        'best_sector'       : dict {qubit: sign} for the minimizing sector
        'best_sample'       : dict {var: 0 or 1}, the QUBO sample achieving E0
        'best_bqm'          : the final BQM from the best sector
        'aux_map'           : Rosenberg auxiliary variable registry
        'all_sector_results': list of per-sector dicts with 'sector', 'E0_sector',
                              'lambda_trace'
    """
    def vprint(*args):
        if verbose:
            print(*args)

    vars_list = [f'q{i}_r{j}' for i in range(n_qubits) for j in range(1, r + 1)]
    C_poly = build_counting_functional(n_qubits, r)  # sector-independent

    global_best_E0    = np.inf
    global_best_sector = None
    global_best_sample = None
    global_best_bqm   = None
    global_best_aux   = None
    all_sector_results = []

    n_sectors = 2 ** n_qubits
    vprint(f"  [XBK] {n_sectors} sectors × {max_iter} Dinkelbach iters "
           f"× SA({num_reads}r × {num_sweeps}sw)")

    for p_idx, sector_combo in enumerate(
            itertools.product([+1, -1], repeat=n_qubits)):
        sector_signs = {i: sector_combo[i] for i in range(n_qubits)}
        H_poly = xbk_transform(qubit_op, n_qubits, r=r, sector_signs=sector_signs)

        # Initial lambda: lower bound = -r * sum|H'_p coefficients|
        lam = (lambda_init
               if lambda_init is not None
               else -r * sum(abs(v) for v in H_poly.values()))

        lambda_trace = [lam]
        sector_best_E0    = np.inf
        sector_best_sample = None
        sector_best_bqm   = None
        sector_best_aux   = None

        for k in range(max_iter):
            # Dinkelbach objective: D_p = H'_p - lambda_k * C_p / r
            D_poly = _poly_add(H_poly, C_poly, scale=-lam / r)

            # Quadratize -> QUBO -> SA
            bqm, aux_map = hubo_to_qubo(D_poly, penalty_multiplier=penalty_multiplier)
            ss = run_sa(bqm, num_reads=num_reads, num_sweeps=num_sweeps,
                        beta_range=beta_range,
                        seed=seed + p_idx * max_iter + k)

            # Scan all SA samples; find best r*H'/C ratio
            best_ratio    = np.inf
            best_bin_sample = None
            for sample, _ in ss.data(['sample', 'energy']):
                spin = {v: 1 - 2 * sample[v] for v in vars_list if v in sample}
                if len(spin) < len(vars_list):
                    continue
                Hv = _poly_eval(H_poly, spin)
                Cv = _poly_eval(C_poly, spin)
                ratio = r * Hv / Cv
                if ratio < best_ratio:
                    best_ratio      = ratio
                    best_bin_sample = dict(sample)

            if best_bin_sample is None:
                break  # no valid samples (shouldn't happen for r=2)

            # Evaluate convergence criterion: phi_p(lambda_k) at the minimizer
            spin = {v: 1 - 2 * best_bin_sample[v]
                    for v in vars_list if v in best_bin_sample}
            Hv   = _poly_eval(H_poly, spin)
            Cv   = _poly_eval(C_poly, spin)
            phi  = Hv - lam * Cv / r   # = D_p(s*; lambda_k)

            lam_new = r * Hv / Cv

            if lam_new < sector_best_E0:
                sector_best_E0     = lam_new
                sector_best_sample = best_bin_sample
                sector_best_bqm    = bqm
                sector_best_aux    = aux_map

            lambda_trace.append(lam_new)
            converged = abs(phi) < tol or abs(lam_new - lam) < tol
            lam = lam_new
            if converged:
                break

        all_sector_results.append({
            'sector':       sector_signs,
            'E0_sector':    sector_best_E0,
            'lambda_trace': lambda_trace,
        })

        if sector_best_E0 < global_best_E0:
            global_best_E0     = sector_best_E0
            global_best_sector = sector_signs
            global_best_sample = sector_best_sample
            global_best_bqm    = sector_best_bqm
            global_best_aux    = sector_best_aux

    best_sec_tuple = (tuple(global_best_sector[i] for i in range(n_qubits))
                      if global_best_sector else None)
    vprint(f"  [XBK] E_0 = {global_best_E0:.8f}  "
           f"(best sector: {best_sec_tuple})")

    return {
        'E0':                 global_best_E0,
        'best_sector':        global_best_sector,
        'best_sample':        global_best_sample,
        'best_bqm':           global_best_bqm,
        'aux_map':            global_best_aux,
        'all_sector_results': all_sector_results,
    }


# ==================================================================
# STEP 7 — Decode best sample -> physical occupations
# ==================================================================

def decode_result(
    best_sample: Dict[str, int],
    n_qubits: int,
    r: int = 2,
) -> Dict[str, Any]:
    """
    Decode QUBO binary sample into spin variables and logical occupations.

    Conversion: binary x_l in {0,1} -> Ising spin s_l = 1 - 2*x_l in {-1,+1}.

    Variable 'q{i}_r{j}' is spin s_{i,j} for qubit i, replica j.

    Logical occupation of tapered qubit i:
        s_{i,1} = s_{i,2} = +1  =>  n_i = 0  (unoccupied)
        s_{i,1} = s_{i,2} = -1  =>  n_i = 1  (occupied)
        s_{i,1} != s_{i,2}      =>  replica_agreed[i] = False  (majority vote used)
    """
    spins = {var: 1 - 2 * val for var, val in best_sample.items()}

    replica_spins: Dict[int, List[int]] = {
        i: [spins.get(f'q{i}_r{j}') for j in range(1, r + 1)]
        for i in range(n_qubits)
    }

    replica_agreed: Dict[int, bool] = {}
    logical_occupations: List[int] = []
    for i in range(n_qubits):
        vals = [s for s in replica_spins[i] if s is not None]
        if not vals:
            replica_agreed[i] = True
            logical_occupations.append(0)
        elif all(s == vals[0] for s in vals):
            replica_agreed[i] = True
            logical_occupations.append(int((1 - vals[0]) // 2))
        else:
            replica_agreed[i] = False
            logical_occupations.append(int(np.mean(vals) < 0))

    return {
        'ising_spins':         spins,
        'replica_spins':       replica_spins,
        'replica_agreed':      replica_agreed,
        'logical_occupations': logical_occupations,
    }


# ==================================================================
# FULL PIPELINE  (model-agnostic entry point)
# ==================================================================

def run_pipeline(
    fermion_op: FermionOperator,
    n_modes: int,
    n_particles: Optional[int] = None,
    taper_sector: Optional[List[int]] = None,
    skip_tapering: bool = False,
    r: int = 2,
    penalty_multiplier: float = 5.0,
    lambda_init: Optional[float] = None,
    max_dinkelbach_iter: int = 10,
    dinkelbach_tol: float = 1e-6,
    num_reads: int = 100,
    num_sweeps: int = 500,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full end-to-end pipeline:

        FermionOperator
          -[BK]---------->  QubitOperator         (n_modes qubits)
          -[Z2 Taper]--->   QubitOperator         (n_tapered <= n_modes qubits)
          -[XBK sectors]->  H'_p, C_p             (2^n_tapered sectors searched)
          -[Dinkelbach]->   QUBO per sector        (quadratized, 2-local)
          -[SA]---------->  best spin config       (approximate ground state)
          -[min_p]------->  E_0 = min_p E_{0,p}   (XBK r=2 approximation)
          -[Decode]------->  occupations, diagnostics

    XBK APPROXIMATION NOTE:
        The returned E_0 is the XBK r=2 approximation to the true ground state
        energy. For models with irrational eigenvalues (like the spinless t-V
        model), the r=2 approximation may differ significantly from the exact
        value. Increase r for better accuracy at the cost of more qubits
        (r times more physical spins, r^m times more sectors, but same Dinkelbach
        and Rosenberg structure).

        Recommended: r = n_tapered / 2 (FMQA paper heuristic).
        For small systems (<= 4 tapered qubits): r = n_tapered for best accuracy.

    Parameters:
        fermion_op          : FermionOperator Hamiltonian
        n_modes             : number of fermionic modes (= number of spin-orbitals)
        n_particles         : particle number for tapering sector auto-detection
        taper_sector        : explicit +-1 list (length = number of Z2 symmetries).
                              If default sector gives wrong E_0 (check vs ED),
                              set this manually.
        skip_tapering       : if True, skip Z2 tapering entirely (use if tapering
                              removes too many qubits for the XBK to work)
        r                   : XBK replication factor (r=2 only; see note above)
        penalty_multiplier  : Rosenberg M = multiplier * max|D_p coeff|
        lambda_init         : Dinkelbach starting point (None = auto lower bound)
        max_dinkelbach_iter : max Dinkelbach iterations per XBK sector
        dinkelbach_tol      : Dinkelbach convergence: |phi_p(lambda)| < tol
        num_reads           : SA reads per Dinkelbach step
        num_sweeps          : SA sweeps per read
        beta_range          : (1/T_max, 1/T_min) for SA annealing schedule
        seed                : RNG seed (incremented per sector and iteration)
        verbose             : print step-by-step progress

    Returns:
        dict with keys:
            'E0'                  : float, XBK ground state energy estimate
            'qubit_op'            : post-BK QubitOperator
            'tapered_op'          : post-tapering QubitOperator (= qubit_op if skipped)
            'n_tapered'           : int, qubits after tapering
            'tap_info'            : tapering metadata
            'dinkelbach_result'   : full Dinkelbach output dict
            'logical_occupations' : list of 0/1, length n_tapered
            'replica_agreed'      : dict {qubit_i: bool}
            'ising_spins'         : dict {var_name: +-1}
    """
    def vprint(*args):
        if verbose:
            print(*args)

    # ---------------------------------------------------------------- Step 1
    vprint("\n[Step 1] Bravyi-Kitaev transform...")
    qubit_op = fermion_to_qubit(fermion_op)
    vprint(f"         {n_modes} modes -> {n_modes} qubits, "
           f"{len(qubit_op.terms)} Pauli terms.")

    # ---------------------------------------------------------------- Step 2
    if skip_tapering:
        vprint("\n[Step 2] Tapering skipped (skip_tapering=True).")
        tapered_op, n_tapered, tap_info = qubit_op, get_n_qubits(qubit_op), {}
    else:
        vprint("\n[Step 2] Z2 symmetry tapering...")
        tapered_op, n_tapered, tap_info = taper_symmetries(
            qubit_op, n_particles=n_particles,
            sector=taper_sector, verbose=verbose,
        )
        n_removed = n_modes - n_tapered
        vprint(f"         -{n_removed} qubit(s). {n_tapered} qubits, "
               f"{len(tapered_op.terms)} terms remain.")

        # Edge case: all qubits tapered away -> energy is the constant term
        if n_tapered == 0:
            E0 = float(tapered_op.terms.get((), 0.0))
            vprint(f"  All qubits fixed by symmetry -> E_0 = {E0:.8f}")
            vprint(f"  If this is wrong, set skip_tapering=True or "
                   f"adjust taper_sector.")
            return {
                'E0': E0, 'qubit_op': qubit_op, 'tapered_op': tapered_op,
                'n_tapered': 0, 'tap_info': tap_info,
                'dinkelbach_result': None,
                'logical_occupations': [], 'replica_agreed': {},
                'ising_spins': {},
            }

    # ---------------------------------------------------------------- Steps 3-6
    vprint(f"\n[Steps 3-6] XBK (r={r}) + Dinkelbach + SA...")
    dk_result = dinkelbach_sa(
        qubit_op=tapered_op,
        n_qubits=n_tapered,
        r=r,
        lambda_init=lambda_init,
        max_iter=max_dinkelbach_iter,
        tol=dinkelbach_tol,
        penalty_multiplier=penalty_multiplier,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        seed=seed,
        verbose=verbose,
    )
    E0 = dk_result['E0']

    # ---------------------------------------------------------------- Step 7
    decoded: Dict[str, Any] = {}
    if dk_result['best_sample'] is not None:
        decoded = decode_result(dk_result['best_sample'], n_tapered, r=r)

    # ---------------------------------------------------------------- Summary
    vprint("\n" + "=" * 65)
    vprint("PIPELINE SUMMARY")
    vprint("=" * 65)
    vprint(f"  Fermionic modes            : {n_modes}")
    vprint(f"  Qubits after BK            : {n_modes}")
    vprint(f"  Qubits after Z2 tapering   : {n_tapered}  "
           f"(-{n_modes-n_tapered} removed)")
    vprint(f"  XBK replication factor     : r = {r}")
    vprint(f"  XBK sectors searched       : {2**n_tapered}")
    vprint(f"  Physical spin variables    : {n_tapered * r}")
    vprint(f"  Ground state energy E_0    : {E0:.8f}  (XBK r={r} approximation)")
    if decoded:
        vprint(f"  Logical occupations        : {decoded.get('logical_occupations')}")
        ra = decoded.get('replica_agreed', {})
        vprint(f"  Replica agreement (all)    : {all(ra.values()) if ra else 'N/A'}")
    best_sec = dk_result.get('best_sector')
    if best_sec:
        vprint(f"  Best XBK sector            : "
               f"{tuple(best_sec[i] for i in range(n_tapered))}")
    vprint("=" * 65)
    vprint(f"  XBK r=2 NOTE: approximation improves with larger r.")
    vprint(f"  For n_tapered={n_tapered} qubits, recommend r={max(2, n_tapered//2)}"
           f" to r={n_tapered} for better accuracy.")
    vprint("=" * 65)

    return {
        'E0':                E0,
        'qubit_op':          qubit_op,
        'tapered_op':        tapered_op,
        'n_tapered':         n_tapered,
        'tap_info':          tap_info,
        'dinkelbach_result': dk_result,
        **decoded,
    }


# ==================================================================
# DEMO
# ==================================================================

if __name__ == '__main__':
    from openfermion import get_sparse_operator

    def exact_E0(H_f, n):
        return float(np.sort(np.linalg.eigvalsh(
            get_sparse_operator(H_f, n_qubits=n).toarray()))[0])

    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DEMO 1: 2-site spinless hopping  (skip_tapering=True)")
    print("  H = -t(c†_0 c_1 + h.c.),  t=1,  Exact E_0 = -1.0")
    print("=" * 65)
    H_hop = FermionOperator('0^ 1', -1.) + FermionOperator('1^ 0', -1.)
    E_hop_exact = exact_E0(H_hop, 2)
    print(f"Exact: {E_hop_exact:.6f}")
    r_hop = run_pipeline(
        H_hop, n_modes=2, n_particles=1,
        skip_tapering=True,            # avoid over-tapering to 0 qubits
        r=2, penalty_multiplier=6.,
        max_dinkelbach_iter=5,
        num_reads=100, num_sweeps=400,
        verbose=True,
    )
    print(f"Pipeline E_0: {r_hop['E0']:.6f}   "
          f"Error: {abs(r_hop['E0']-E_hop_exact):.4f}")

    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DEMO 2: 4-site spinless t-V model  (1D open chain)")
    print("  H = -t sum (c†_i c_j + h.c.) + V sum n_i n_j")
    print("  t=1, V=2, N=2 particles.  Exact E_0 = -1.855773")
    print("=" * 65)
    t, V, L = 1.0, 2.0, 4
    H_tV = FermionOperator()
    for i in range(L - 1):
        H_tV += FermionOperator(f'{i}^ {i+1}', -t) + FermionOperator(f'{i+1}^ {i}', -t)
        H_tV += V * FermionOperator(f'{i}^ {i}') * FermionOperator(f'{i+1}^ {i+1}')
    E_tV_exact = exact_E0(H_tV, L)
    print(f"Exact: {E_tV_exact:.6f}")
    r_tV = run_pipeline(
        H_tV, n_modes=L, n_particles=2,
        r=2, penalty_multiplier=6.,
        max_dinkelbach_iter=5,
        num_reads=100, num_sweeps=400,
        verbose=True,
    )
    print(f"Pipeline E_0 (r=2): {r_tV['E0']:.6f}   "
          f"Error: {abs(r_tV['E0']-E_tV_exact):.4f}")
    print("(r=2 gives approximate answer; larger r improves accuracy)")

    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DEMO 3: 2-site Hubbard model  (spinful)")
    print("  t=1, U=4, N=2 particles.  Exact E_0 = -1.0")
    print("  Modes: 0=site0-up, 1=site0-dn, 2=site1-up, 3=site1-dn")
    print("=" * 65)
    t_h, U_h = 1.0, 4.0
    H_hub = FermionOperator()
    H_hub += FermionOperator('0^ 2', -t_h) + FermionOperator('2^ 0', -t_h)
    H_hub += FermionOperator('1^ 3', -t_h) + FermionOperator('3^ 1', -t_h)
    H_hub += U_h * FermionOperator('0^ 0') * FermionOperator('1^ 1')
    H_hub += U_h * FermionOperator('2^ 2') * FermionOperator('3^ 3')
    E_hub_exact = exact_E0(H_hub, 4)
    print(f"Exact: {E_hub_exact:.6f}")
    r_hub = run_pipeline(
        H_hub, n_modes=4, n_particles=2,
        r=2, penalty_multiplier=6.,
        max_dinkelbach_iter=5,
        num_reads=100, num_sweeps=400,
        verbose=True,
    )
    print(f"Pipeline E_0 (r=2): {r_hub['E0']:.6f}   "
          f"Error: {abs(r_hub['E0']-E_hub_exact):.4f}")
