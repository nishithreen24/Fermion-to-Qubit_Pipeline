"""
Pipeline: FermionOperator -> BK -> Z2 Tapering -> XBK -> Dinkelbach-SA
=======================================================================

MATHEMATICAL GLOSSARY
----------------------

FermionOperator:
    A linear combination of products of fermionic creation (c†_i) and
    annihilation (c_i) operators obeying anti-commutation relations:
        {c_i, c†_j} = delta_{ij},   {c_i, c_j} = 0.

Bravyi-Kitaev (BK) Transform:
    Maps n fermionic modes to n qubits with O(log n)-local Pauli strings.
    Output: QubitOperator = sum_j h_j * sigma_j.

Pauli string:
    Tensor product of single-qubit matrices from {I, X, Y, Z}.

Z2 Symmetry Tapering:
    Removes one qubit per independent Z2 symmetry of H while preserving
    the ground state energy exactly. (Full description in z2_tapering.py.)

XBK (Xia-Bian-Kais) Transform  [Xia, Bian, Kais, J. Phys. Chem. B 2018]:
    Embeds an m-qubit Pauli Hamiltonian H into a 2m-spin Ising polynomial
    H'_p(s) by replicating each qubit into r=2 copies.

    Sector: a table of signs Sp(i,2) in {+1,-1} for each qubit i, with
    Sp(i,1) = +1 by convention. There are 2^m sectors for r=2.
    The full spectrum is recovered by minimizing over ALL sectors.

    Mapping rules for qubit i, with replicas s_{i,1} and s_{i,2} and sign Sp:
        X_i  ->  Sp * (1 - s_{i,1} * s_{i,2}) / 2
        Y_i  ->  i * Sp * (s_{i,2} - s_{i,1}) / 2
        Z_i  ->  Sp * (s_{i,1} + s_{i,2}) / 2
        I    ->  (1 + s_{i,1} * s_{i,2}) / 2

    A full Pauli string maps by multiplying these per-qubit expressions,
    using s_l^2 = 1 (symmetric difference rule for variable sets).

    Result: H'_p(s) is an Ising polynomial (sum of products of Z-variables).

Counting Functional C_p(s)  [QMP_NOTES eq. 81]:
    C_p(s) = sum_{eps in {+1,-1}^m} [sum_{j=1}^r proj(eps, j)]^2

    where proj(eps, j) = prod_{k=0}^{m-1} (1 + eps_k * s_{k,j}) / 2
    is 1 if replica j is in state eps, and 0 otherwise.

    For r=2 with Sp=+1:
        C_p takes values in {2, 4}.
        C_p = 4 when s_{i,1} = s_{i,2} for ALL i (both replicas agree).
        C_p = 2 when exactly one pair disagrees.
        C_p = 0 NEVER for r=2 (the formula never gives 0).

    This is different from the simpler product formula (1+s_{i,1}s_{i,2})/2
    which gives {0,1} — that formula is WRONG; C is never 0 for r=2.

Rayleigh Quotient  [FMQA eq. 6]:
    The ground state energy in sector p is:
        E_{0,p} = min_{s in {-1,+1}^{r*m}} [r * H'_p(s) / C_p(s)]
    The factor r normalizes for the multiplicity structure of C_p.
    The global ground state energy is:
        E_0 = min_p E_{0,p}

    IMPORTANT: This is an APPROXIMATION for finite r. It becomes exact as
    r -> infinity because C_p becomes a finer measure of quantum amplitudes.
    For r=2: the approximation quality depends on the model.
    Accuracy can be improved by increasing r at the cost of more qubits.

Dinkelbach Iteration  [Dinkelbach 1967]:
    Converts the fractional minimization E_{0,p} = min_s [r*H'_p/C_p] into
    a sequence of linear (unconstrained) subproblems. For sector p:

    phi_p(lambda) = min_s [H'_p(s) - lambda * C_p(s) / r]

    This is a strictly decreasing, convex function of lambda, and
    phi_p(E_{0,p}) = 0. Fixed-point iteration:
        0. Start: lambda_0 (initial guess, e.g. a lower bound).
        1. Minimize D_p(s; lambda_k) = H'_p(s) - lambda_k * C_p(s) / r
           over all s in {-1,+1}^{r*m}, via SA on the resulting QUBO.
        2. Evaluate at minimizer s*:
               lambda_{k+1} = r * H'_p(s*) / C_p(s*)
        3. Stop when |phi_p(lambda_k)| = |H'_p(s*) - lambda_k * C_p(s*)/r| < tol.

    We run this for all 2^m sectors and return min_p lambda_p*.

Rosenberg Quadratization:
    Reduces the k-local Ising polynomial D_p(s; lambda_k) to a 2-local QUBO
    by introducing auxiliary binary variables. The Rosenberg penalty for
    auxiliary a = x_i * x_j (with x_l = (1-s_l)/2 in {0,1}) is:
        P = M * (x_i*x_j - 2*a*x_i - 2*a*x_j + 3*a) >= 0, = 0 iff a=x_i*x_j.
    M = penalty_multiplier * max|coefficient in D_p|.

Simulated Annealing (SA):
    Samples from P(x) proportional to exp(-E(x)/T) with T decreasing.
    Used to minimize the QUBO at each Dinkelbach step.
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

# Type alias: Ising polynomial = dict {frozenset(var_names): coefficient}
IsingPoly = Dict[frozenset, float]


# ==================================================================
# STEP 1 — Bravyi-Kitaev Transform
# ==================================================================

def fermion_to_qubit(fermion_op: FermionOperator) -> QubitOperator:
    """Apply the Bravyi-Kitaev transform. Returns a QubitOperator."""
    return bravyi_kitaev(fermion_op)


# ==================================================================
# STEP 2 — Z2 Symmetry Tapering
# ==================================================================

def taper_symmetries(
    qubit_op: QubitOperator,
    n_particles: Optional[int] = None,
    sector: Optional[List[int]] = None,
    verbose: bool = True,
) -> Tuple[QubitOperator, int, Dict]:
    """
    Remove qubits using Z2 symmetries.

    Each independent Pauli Z2 symmetry tau_i (with [tau_i, H]=0 and tau_i^2=I)
    lets us remove exactly one qubit by fixing its eigenvalue in {+1,-1}.

    Returns:
        tapered_op : QubitOperator on fewer qubits
        n_tapered  : int (qubits remaining)
        tap_info   : dict with symmetry generators and eigenvalues used
    """
    n_qubits = get_n_qubits(qubit_op)
    if n_qubits == 0:
        return qubit_op, 0, {}
    tapered_op, n_tapered, lambdas, info = z2_taper(
        qubit_op, n_qubits,
        n_particles=n_particles,
        sector=sector,
        verbose=verbose,
    )
    return tapered_op, n_tapered, info


# ==================================================================
# STEP 3 — XBK Transform
# ==================================================================

def xbk_transform(
    qubit_op: QubitOperator,
    n_qubits: int,
    r: int = 2,
    sector_signs: Optional[Dict[int, int]] = None,
) -> IsingPoly:
    """
    Apply XBK with the given sector (sign table Sp(i,2) for each qubit i).

    Variable labels: 'q{i}_r{j}' for qubit i, replica j in {1,...,r}.
    r=2 only (the standard implementation; larger r requires more replicas).

    Returns H'_p(s) as an IsingPoly.
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
            warnings.warn(f"Non-zero imaginary coefficient {rc}. Verify Hamiltonian is Hermitian.")

        for key, val in term_poly.items():
            result[key] += cr * val

    return {k: v for k, v in result.items() if abs(v) > 1e-12}


def build_counting_functional(n_qubits: int, r: int = 2) -> IsingPoly:
    """
    Build the XBK counting functional C_p(s) as an Ising polynomial.

    CORRECT FORMULA (QMP_NOTES eq. 81):
        C_p(s) = sum_{eps in {+1,-1}^m} [sum_{j=1}^r proj(eps, j)]^2

    where:
        proj(eps, j) = prod_{k=0}^{m-1} (1 + eps_k * s_{k,j}) / 2
                     = 1 if replica j is in basis state eps for ALL qubits
                     = 0 otherwise

    The inner sum [sum_j proj(eps,j)] counts how many replicas are in state eps.
    Squaring and summing over all eps gives the total multiplicity C_p(s).

    For r=2:
        C_p = 4  when s_{i,1} = s_{i,2} for all i  (both replicas in same state)
        C_p = 2  when replicas differ on exactly the right qubits
        (C is NEVER 0 for r=2)

    The Rayleigh quotient E_{0,p} = min_s [r * H'_p(s) / C_p(s)] includes the
    factor r to normalize: on all-agree configs (C=4=r*2), r/C = 2/4 = 1/2, so
    r*H'/C = H'/(2) and at the minimum this recovers E_0 approximately.

    Parameters:
        n_qubits : number of logical (post-tapering) qubits m
        r        : replication factor (r=2 only)

    Returns:
        C_poly : IsingPoly for C_p(s)
    """
    assert r == 2

    C: IsingPoly = {frozenset(): 0.0}

    for eps in itertools.product([-1, +1], repeat=n_qubits):
        # Build inner(s; eps) = sum_{j=1}^r proj(eps, s_*j)
        # as an Ising polynomial in s_{k,j}.
        inner: IsingPoly = {frozenset(): 0.0}
        for j in range(1, r + 1):
            # proj(eps, j) = prod_k (1 + eps_k * s_{k,j}) / 2
            proj: IsingPoly = {frozenset(): 1.0}
            for k, ek in enumerate(eps):
                vkj = f'q{k}_r{j}'
                factor: IsingPoly = {frozenset(): 0.5, frozenset({vkj}): ek * 0.5}
                proj = _poly_mul(proj, factor)
            inner = _poly_add(inner, proj)

        # C += inner^2
        inner_sq = _poly_mul(inner, inner)
        C = _poly_add(C, inner_sq)

    return {k: v for k, v in C.items() if abs(v) > 1e-12}


# ==================================================================
# Polynomial arithmetic helpers
# ==================================================================

def _poly_mul(p1: IsingPoly, p2: IsingPoly) -> IsingPoly:
    """Multiply two Ising polynomials. Uses s_l^2 = 1 (symmetric difference)."""
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
    """Evaluate an Ising polynomial at a spin assignment {var: +/-1}."""
    total = 0.0
    for key, coeff in poly.items():
        monomial = 1.0
        for v in key:
            monomial *= assignment[v]
        total += coeff * monomial
    return total


# ==================================================================
# STEP 4 — Rosenberg Quadratization: IsingPoly -> QUBO
# ==================================================================

def hubo_to_qubo(
    ising_poly: IsingPoly,
    penalty_multiplier: float = 5.0,
) -> Tuple[dimod.BinaryQuadraticModel, Dict[str, str]]:
    """
    Convert an Ising polynomial H(s), s in {-1,+1}, to a QUBO Q(x), x in {0,1}.

    (a) Variable change: x_l = (1 - s_l)/2.
        Each monomial prod_{l in S} s_l = prod_{l in S} (1 - 2*x_l)
        expands as sum_{T subset S} (-2)^|T| * prod_{l in T} x_l.

    (b) Greedy Rosenberg quadratization: for the most-used pair (x_i, x_j)
        in degree >= 3 monomials, introduce auxiliary a = x_i * x_j with:
            P = M * (x_i*x_j - 2*a*x_i - 2*a*x_j + 3*a) >= 0, = 0 iff a=x_i*x_j.
        Repeat until all monomials have degree <= 2.

    M = penalty_multiplier * max|coefficient|.

    Returns:
        bqm     : dimod.BinaryQuadraticModel (BINARY vartype)
        aux_map : dict {aux_name: 'xi_name*xj_name'}
    """
    binary_poly: Dict[frozenset, float] = defaultdict(float)
    for spin_key, coeff in ising_poly.items():
        svars = list(spin_key)
        for k in range(len(svars) + 1):
            for subset in itertools.combinations(svars, k):
                binary_poly[frozenset(subset)] += coeff * (-2.0) ** k
    binary_poly = {k: v for k, v in binary_poly.items() if abs(v) > 1e-12}

    max_coeff = max(abs(v) for v in binary_poly.values()) if binary_poly else 1.0
    M = penalty_multiplier * max_coeff

    aux_counter = [0]
    aux_map: Dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        if max((len(k) for k in binary_poly), default=0) <= 2:
            break
        pair_count: Dict[frozenset, int] = defaultdict(int)
        for monomial in binary_poly:
            if len(monomial) >= 3:
                for pair in itertools.combinations(sorted(monomial), 2):
                    pair_count[frozenset(pair)] += 1
        if not pair_count:
            break
        best_pair = max(pair_count, key=lambda p: pair_count[p])
        xi, xj = sorted(best_pair)
        aux_name = f'aux_{aux_counter[0]}'
        aux_counter[0] += 1
        aux_map[aux_name] = f'{xi}*{xj}'

        new_poly: Dict[frozenset, float] = defaultdict(float)
        for monomial, coeff in binary_poly.items():
            if best_pair.issubset(monomial):
                new_poly[(monomial - best_pair) | frozenset({aux_name})] += coeff
            else:
                new_poly[monomial] += coeff
        new_poly[frozenset({xi, xj})]       += M
        new_poly[frozenset({aux_name, xi})]  -= 2.0 * M
        new_poly[frozenset({aux_name, xj})]  -= 2.0 * M
        new_poly[frozenset({aux_name})]      += 3.0 * M
        binary_poly = {k: v for k, v in new_poly.items() if abs(v) > 1e-12}
        changed = True

    bqm = dimod.BinaryQuadraticModel(vartype='BINARY')
    for monomial, coeff in binary_poly.items():
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
    num_reads: int = 1000,
    num_sweeps: int = 2000,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
) -> dimod.SampleSet:
    """
    Run dimod SimulatedAnnealingSampler on a QUBO.
    Samples from P(x) proportional to exp(-beta * E(x)) with beta increasing.
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
    max_iter: int = 20,
    tol: float = 1e-6,
    penalty_multiplier: float = 5.0,
    num_reads: int = 1000,
    num_sweeps: int = 2000,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Find E_0 = min_p E_{0,p} via Dinkelbach iteration over all 2^m XBK sectors.

    For each sector p (defined by sector_signs Sp(i,2) in {+1,-1} for each qubit i):
        Build H'_p(s) via xbk_transform.
        Build C_p(s) via build_counting_functional (shared across sectors).
        Run Dinkelbach: find lambda_p* = E_{0,p} by iterating:
            D_p(s; lambda_k) = H'_p(s) - lambda_k * C_p(s) / r
            s* = argmin_s D_p(s; lambda_k)   [via SA on QUBO of D_p]
            lambda_{k+1} = r * H'_p(s*) / C_p(s*)
        until |phi_p(lambda_k)| = |D_p(s*; lambda_k)| < tol.

    E_0 = min_p lambda_p*.

    Parameters:
        qubit_op          : post-tapering QubitOperator
        n_qubits          : number of tapered qubits m
        r                 : replication factor (r=2 only)
        lambda_init       : initial lambda per sector (default: -sum|H'_p coeffs|)
        max_iter          : max Dinkelbach iterations per sector
        tol               : convergence threshold on |phi_p(lambda)|
        penalty_multiplier: Rosenberg penalty weight scale
        num_reads         : SA reads per Dinkelbach step
        num_sweeps        : SA sweeps per read
        beta_range        : (1/T_max, 1/T_min) for SA
        seed              : RNG seed (incremented per sector and iteration)
        verbose           : print progress

    Returns:
        dict with 'E0', 'best_sector', 'best_sample', 'best_bqm', 'aux_map',
                  'lambda_trace', 'all_sector_results'.
    """
    def vprint(*args):
        if verbose:
            print(*args)

    vars_list = [f'q{i}_r{j}' for i in range(n_qubits) for j in range(1, r+1)]

    # Build counting functional (sector-independent for r=2 with all Sp=+1 for C)
    # Note: C_p is actually sector-independent for r=2 as defined in the paper;
    # it depends only on the AGREEMENT structure, not the sector signs.
    C_poly = build_counting_functional(n_qubits, r)

    global_best_E0 = np.inf
    global_best_sector = None
    global_best_sample = None
    global_best_bqm = None
    global_best_aux = None
    all_sector_results = []

    n_sectors = 2 ** n_qubits
    vprint(f"\n[Dinkelbach] Searching {n_sectors} XBK sectors x {max_iter} iterations.")

    for p_idx, sector_combo in enumerate(itertools.product([+1, -1], repeat=n_qubits)):
        sector_signs = {i: sector_combo[i] for i in range(n_qubits)}

        # Build H'_p
        H_poly = xbk_transform(qubit_op, n_qubits, r=r, sector_signs=sector_signs)

        max_H = max(abs(v) for v in H_poly.values()) if H_poly else 1.0

        # Initial lambda: lower bound = -r * sum|H'_p coeffs|
        lam = lambda_init if lambda_init is not None else -r * sum(abs(v) for v in H_poly.values())
        lambda_trace = [lam]
        sector_best_E0 = np.inf
        sector_best_sample = None
        sector_best_bqm = None
        sector_best_aux = None

        for k in range(max_iter):
            # Dinkelbach objective: D_p(s; lambda) = H'_p(s) - lambda * C_p(s) / r
            D_poly = _poly_add(H_poly, C_poly, scale=-lam / r)

            # Quadratize D_p
            bqm, aux_map = hubo_to_qubo(D_poly, penalty_multiplier=penalty_multiplier)

            # Run SA
            ss = run_sa(
                bqm,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                beta_range=beta_range,
                seed=seed + p_idx * max_iter + k,
            )

            # Evaluate best sample in terms of H'/C
            best_ratio = np.inf
            best_bin_sample = None
            for sample, energy in ss.data(['sample', 'energy']):
                spin_xbk = {v: 1 - 2*sample[v] for v in vars_list if v in sample}
                if len(spin_xbk) < len(vars_list):
                    continue
                H_val = _poly_eval(H_poly, spin_xbk)
                C_val = _poly_eval(C_poly, spin_xbk)
                if C_val < 1e-9:
                    continue
                ratio = r * H_val / C_val
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_bin_sample = dict(sample)

            if best_bin_sample is None:
                # All samples had C=0 — shouldn't happen for r=2 correct C
                vprint(f"  Sector {sector_combo} iter {k}: No valid samples.")
                break

            # Compute phi_p(lambda_k) = D_p(s*; lambda_k) at the best sample
            spin_xbk = {v: 1 - 2*best_bin_sample[v] for v in vars_list if v in best_bin_sample}
            H_val = _poly_eval(H_poly, spin_xbk)
            C_val = _poly_eval(C_poly, spin_xbk)
            phi_val = H_val - lam * C_val / r  # = D_p evaluated at minimizer

            lam_new = r * H_val / C_val

            if lam_new < sector_best_E0:
                sector_best_E0 = lam_new
                sector_best_sample = best_bin_sample
                sector_best_bqm = bqm
                sector_best_aux = aux_map

            lambda_trace.append(lam_new)

            # Convergence: phi_p(lambda*) = 0
            converged = abs(phi_val) < tol or abs(lam_new - lam) < tol
            lam = lam_new

            if converged:
                break

        all_sector_results.append({
            'sector': sector_signs,
            'E0_sector': sector_best_E0,
            'lambda_trace': lambda_trace,
        })

        if sector_best_E0 < global_best_E0:
            global_best_E0 = sector_best_E0
            global_best_sector = sector_signs
            global_best_sample = sector_best_sample
            global_best_bqm = sector_best_bqm
            global_best_aux = sector_best_aux

    vprint(f"\n[Dinkelbach] Best E_0 = {global_best_E0:.8f}  "
           f"(sector {tuple(global_best_sector[i] for i in range(n_qubits))})")

    return {
        'E0':                 global_best_E0,
        'best_sector':        global_best_sector,
        'best_sample':        global_best_sample,
        'best_bqm':           global_best_bqm,
        'aux_map':            global_best_aux,
        'all_sector_results': all_sector_results,
    }


# ==================================================================
# STEP 7 — Decode best sample
# ==================================================================

def decode_result(
    best_sample: Dict[str, int],
    n_qubits: int,
    r: int = 2,
) -> Dict[str, Any]:
    """
    Decode the best QUBO sample into physical spin variables and occupations.

    For replica j of qubit i: spin s_{i,j} = 1 - 2*x_{i,j}.
    Logical occupation of qubit i: n_i = (1 - s_{i,1}) / 2 if replicas agree.
    If s_{i,1} != s_{i,2}: replica disagreement (use majority vote).
    """
    ising_spins = {var: 1 - 2 * val for var, val in best_sample.items()}

    replica_spins: Dict[int, List[int]] = {}
    for i in range(n_qubits):
        replica_spins[i] = [
            ising_spins.get(f'q{i}_r{j}') for j in range(1, r + 1)
        ]

    replica_agreed: Dict[int, bool] = {}
    logical_occupations: List[int] = []
    for i in range(n_qubits):
        spins = [s for s in replica_spins[i] if s is not None]
        if not spins:
            replica_agreed[i] = True
            logical_occupations.append(0)
        elif all(s == spins[0] for s in spins):
            replica_agreed[i] = True
            logical_occupations.append(int((1 - spins[0]) // 2))
        else:
            replica_agreed[i] = False
            logical_occupations.append(int(np.mean(spins) < 0))

    return {
        'ising_spins':         ising_spins,
        'replica_spins':       replica_spins,
        'replica_agreed':      replica_agreed,
        'logical_occupations': logical_occupations,
    }


# ==================================================================
# FULL PIPELINE
# ==================================================================

def run_pipeline(
    fermion_op: FermionOperator,
    n_modes: int,
    n_particles: Optional[int] = None,
    taper_sector: Optional[List[int]] = None,
    r: int = 2,
    penalty_multiplier: float = 5.0,
    lambda_init: Optional[float] = None,
    max_dinkelbach_iter: int = 10,
    dinkelbach_tol: float = 1e-6,
    num_reads: int = 1000,
    num_sweeps: int = 2000,
    beta_range: Tuple[float, float] = (0.1, 10.0),
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline:
        FermionOperator
          -[BK]-------->  QubitOperator           (n_modes qubits)
          -[Z2 Taper]-->  QubitOperator           (n_tapered qubits)
          -[Dinkelbach]>  Search over 2^n_tapered XBK sectors:
              H'_p + C_p -> QUBO -> SA -> E_{0,p}
          -[min_p]----->  E_0 = min_p E_{0,p}
          -[Decode]----->  occupations, diagnostics

    The XBK approximation quality improves with r; r=2 is the standard starting
    point. For better accuracy, increase r (at the cost of r times more qubits).

    Parameters:
        fermion_op          : FermionOperator Hamiltonian
        n_modes             : number of fermionic modes
        n_particles         : particle number (for tapering sector detection)
        taper_sector        : explicit +-1 eigenvalues for each Z2 symmetry
        r                   : XBK replication factor (default 2)
        penalty_multiplier  : Rosenberg penalty M = multiplier * max|coeff|
        lambda_init         : initial lambda for Dinkelbach (default: -r*sum|coeffs|)
        max_dinkelbach_iter : max iterations per sector
        dinkelbach_tol      : convergence threshold |phi(lambda)|
        num_reads           : SA reads per Dinkelbach step
        num_sweeps          : SA sweeps per read
        beta_range          : (1/T_max, 1/T_min) for SA
        seed                : RNG seed
        verbose             : print progress

    Returns:
        dict with 'E0', 'qubit_op', 'tapered_op', 'n_tapered', 'tap_info',
                  'dinkelbach_result', 'logical_occupations', 'replica_agreed',
                  and all other intermediate objects.
    """
    def vprint(*args):
        if verbose:
            print(*args)

    # Step 1: BK
    vprint("\n[Step 1] Bravyi-Kitaev transform...")
    qubit_op = fermion_to_qubit(fermion_op)
    vprint(f"         {n_modes} modes -> {n_modes} qubits, "
           f"{len(qubit_op.terms)} Pauli terms.")

    # Step 2: Z2 tapering
    vprint("\n[Step 2] Z2 symmetry tapering...")
    tapered_op, n_tapered, tap_info = taper_symmetries(
        qubit_op, n_particles=n_particles, sector=taper_sector, verbose=verbose,
    )
    n_removed = n_modes - n_tapered
    vprint(f"         -{n_removed} qubit(s). {n_tapered} qubits, "
           f"{len(tapered_op.terms)} terms remain.")

    if n_tapered == 0:
        E0 = float(tapered_op.terms.get((), 0.0))
        vprint(f"  All qubits fixed by symmetry. E_0 = {E0:.8f}")
        return {
            'E0': E0, 'qubit_op': qubit_op, 'tapered_op': tapered_op,
            'n_tapered': 0, 'tap_info': tap_info, 'dinkelbach_result': None,
            'logical_occupations': [], 'replica_agreed': {},
        }

    # Steps 3-6: XBK + Dinkelbach + SA (sector search included)
    vprint(f"\n[Steps 3-6] XBK sector search + Dinkelbach + SA...")
    vprint(f"           r={r}, {2**n_tapered} sectors, {max_dinkelbach_iter} iters/sector.")
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

    # Step 7: Decode
    decoded = {}
    if dk_result['best_sample'] is not None:
        decoded = decode_result(dk_result['best_sample'], n_tapered, r=r)

    # Summary
    vprint("\n" + "=" * 60)
    vprint("PIPELINE SUMMARY")
    vprint("=" * 60)
    vprint(f"  Fermionic modes           : {n_modes}")
    vprint(f"  Qubits after BK           : {n_modes}")
    vprint(f"  Qubits after Z2 tapering  : {n_tapered}  (-{n_removed})")
    vprint(f"  XBK replication factor    : r = {r}")
    vprint(f"  XBK physical qubits       : {n_tapered * r}")
    vprint(f"  XBK sectors searched      : {2**n_tapered}")
    vprint(f"  Ground state energy E_0   : {E0:.8f}  (XBK r={r} approximation)")
    vprint(f"  Best XBK sector           : "
           f"{tuple(dk_result['best_sector'][i] for i in range(n_tapered)) if dk_result['best_sector'] else 'N/A'}")
    if decoded:
        vprint(f"  Logical occupations       : {decoded.get('logical_occupations', [])}")
        vprint(f"  Replica agreement         : {all(decoded.get('replica_agreed', {}).values())}")
    vprint("=" * 60)
    vprint(f"  NOTE: XBK r=2 is an approximation. Accuracy improves with larger r.")
    vprint(f"  For the FMQA paper benchmarks, r = n_tapered/2 is recommended.")
    vprint("=" * 60)

    return {
        'E0':                  E0,
        'qubit_op':            qubit_op,
        'tapered_op':          tapered_op,
        'n_tapered':           n_tapered,
        'tap_info':            tap_info,
        'dinkelbach_result':   dk_result,
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

    print("=" * 60)
    print("DEMO 1: 2-site spinless hopping  H = -t(c†0 c1 + h.c.)")
    print("        t=1, N=1 particle.  Exact E_0 = -1.0")
    print("=" * 60)
    H_hop = FermionOperator('0^ 1', -1.0) + FermionOperator('1^ 0', -1.0)
    E_hop = exact_E0(H_hop, 2)
    print(f"Exact: {E_hop:.8f}")
    result = run_pipeline(
        H_hop, n_modes=2, n_particles=1, r=2,
        penalty_multiplier=6.0, max_dinkelbach_iter=8,
        num_reads=500, num_sweeps=1000, verbose=True,
    )
    print(f"Pipeline E_0: {result['E0']:.8f}   Exact: {E_hop:.8f}")

    print("\n" + "=" * 60)
    print("DEMO 2: 4-site spinless t-V chain")
    print("        t=1, V=2, N=2 particles.  Exact E_0 = -1.855773")
    print("=" * 60)
    t, V, L = 1.0, 2.0, 4
    H_tV = FermionOperator()
    for i in range(L - 1):
        H_tV += FermionOperator(f'{i}^ {i+1}', -t) + FermionOperator(f'{i+1}^ {i}', -t)
        H_tV += V * FermionOperator(f'{i}^ {i}') * FermionOperator(f'{i+1}^ {i+1}')
    E_tV = exact_E0(H_tV, L)
    print(f"Exact: {E_tV:.8f}")
    result2 = run_pipeline(
        H_tV, n_modes=L, n_particles=2, r=2,
        penalty_multiplier=6.0, max_dinkelbach_iter=8,
        num_reads=500, num_sweeps=1000, verbose=True,
    )
    print(f"Pipeline E_0: {result2['E0']:.8f}   Exact: {E_tV:.8f}")
    print(f"Error: {abs(result2['E0'] - E_tV):.4f}")
    print("(r=2 XBK is an approximation; error decreases as r increases)")