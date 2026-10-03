"""
xbk_pipeline.py
===============
Self-contained ground-state-energy pipeline for an *arbitrary* fermionic model:

    FermionOperator  ->  Bravyi-Kitaev  ->  Z2 symmetry tapering
                     ->  XBK transform   ->  Dinkelbach fractional-programming
                     ->  Simulated Annealing  ->  E0

Unlike pipeline_base.py / XBK_method.py / helper_functions.py, this module has
NO dependency on mpi4py, dimod or symengine.  It needs only:

    openfermion, numpy, scipy, and the local fieldmath.py (pure python GF(2)).

Public API
----------
ground_state_energy(H_fermion, ...)   -> float          # the whole pipeline
exact_ground_energy(H_fermion, n)     -> float          # ED reference
bk_transform(H_fermion)               -> QubitOperator
z2_taper(qubit_H)                     -> [QubitOperator] # symmetry sectors
run_xbk(qubit_H, ...)                 -> float           # XBK on one operator

Theory notes
------------
XBK (Xia-Bian-Kais) represents an n-qubit Pauli Hamiltonian H on r "copies" of
the system.  With per-copy signs S'(j) = -1 if j < p else +1, the energy is the
minimum of a *ratio* of two Ising polynomials in the r*n binary copy-spins:

        E0(p) = min_s  H'_p(s) / C_p(s)

    H'_p(s) = sum_{j,k} S'(j) S'(k) <copy_j| H |copy_k>      (Z-only polynomial)
    C_p(s)  = sum_b ( sum_j S'(j) P_{j,b}(s) )^2             (counting operator)

The ratio minimisation is solved by Dinkelbach's algorithm: iterate
    s* = argmin_s [ H'_p(s) - lam * C_p(s) ],   lam <- H'(s*)/C(s*)
where each inner argmin is a QUBO solved by simulated annealing after the
k-local polynomial is quadratised (Rosenberg) to 2-local.
"""

import os
import math
import argparse
from collections import defaultdict
from itertools import product as iproduct
from multiprocessing import Pool

# Intra-task SA parallelism: the independent SA restarts of a single Dinkelbach
# sub-problem are spread across this many worker processes.
#
# IMPORTANT: spawning a multiprocessing.Pool inside an MPI rank corrupts
# OpenMPI's InfiniBand/UCX RDMA registrations and hangs at the first fork.  So
# when running under MPI (detected via the launcher's env vars) we force serial
# SA regardless of -c, and rely on task-level MPI parallelism instead (one rank
# per core).  Locally (no MPI) it defaults to 1 and can be raised with
# XBK_SA_WORKERS for testing.
_UNDER_MPI = any(v in os.environ for v in
                 ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK",
                  "OMPI_COMM_WORLD_RANK"))
if _UNDER_MPI:
    SA_WORKERS = 1
else:
    SA_WORKERS = int(os.environ.get("XBK_SA_WORKERS",
                                    os.environ.get("SLURM_CPUS_PER_TASK", "1")))

import numpy as np

from openfermion import FermionOperator, QubitOperator, bravyi_kitaev, commutator
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import taper_off_qubits
from openfermion.utils import count_qubits

import fieldmath as fm


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 - model registry (any FermionOperator is accepted directly too)
# ═════════════════════════════════════════════════════════════════════════════

def tv1d(L=3, t=1.0, V=1.0, periodic=False):
    """1D spinless t-V chain, particle-hole symmetric convention.

        H = -t sum_<ij> (c+_i c_j + h.c.) + V sum_<ij> (n_i-1/2)(n_j-1/2)
    """
    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic and L >= 3:
        bonds.append((L - 1, 0))
    H = FermionOperator()
    for (i, j) in bonds:
        H += FermionOperator(((i, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (i, 0)), -t)
        H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)
        H += FermionOperator(((i, 1), (i, 0)), -V / 2)
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator((), V / 4)
    return H


def hubbard1d(L=2, t=1.0, U=2.0, mu=None, periodic=False):
    """1D Fermi-Hubbard chain. Spin-orbital index = 2*site + spin (0=up,1=dn).

        H = -t sum_<ij>,s (c+_is c_js + h.c.) + U sum_i n_i,up n_i,dn
            - mu sum_i,s n_i,s
    """
    if mu is None:
        mu = U / 2.0  # half filling
    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic and L >= 3:
        bonds.append((L - 1, 0))

    def so(site, spin):
        return 2 * site + spin

    H = FermionOperator()
    for (i, j) in bonds:
        for s in (0, 1):
            H += FermionOperator(((so(i, s), 1), (so(j, s), 0)), -t)
            H += FermionOperator(((so(j, s), 1), (so(i, s), 0)), -t)
    for i in range(L):
        H += FermionOperator(((so(i, 0), 1), (so(i, 0), 0),
                              (so(i, 1), 1), (so(i, 1), 0)), U)
        for s in (0, 1):
            H += FermionOperator(((so(i, s), 1), (so(i, s), 0)), -mu)
    return H


def siam(L=2, U=4.0, t=1.0, V=1.0):
    """Single Impurity Anderson Model (star geometry), matching model_siam.py.

        H = U (n_imp,up - 1/2)(n_imp,dn - 1/2)
          + sum_{b,s} eps_b n_{b,s}
          + sum_{b,s} V_b (a+_{imp,s} a_{b,s} + h.c.)

    L = N_bath bath sites.  Mode ordering: mode = 2*site + spin, site 0 = impurity,
    sites 1..L = bath.  Discretised flat band: eps_b = -2t cos(b pi/(L+1)),
    V_b = V sqrt(2/(L+1)) sqrt(1 - (eps_b/2t)^2).
    """
    b_idx = np.arange(1, L + 1)
    eps_bath = -2.0 * t * np.cos(b_idx * np.pi / (L + 1))
    V_bath = V * np.sqrt(2.0 / (L + 1)) * np.sqrt(
        np.maximum(0.0, 1.0 - (eps_bath / (2.0 * t)) ** 2))

    def mode(site, spin):
        return 2 * site + spin

    H = FermionOperator()
    up, dn = mode(0, 0), mode(0, 1)
    H += FermionOperator(((up, 1), (dn, 1), (dn, 0), (up, 0)), U)
    for spin in range(2):
        H += FermionOperator(((mode(0, spin), 1), (mode(0, spin), 0)), -U / 2)
    H += FermionOperator((), U / 4)
    for b_site, eps_b in zip(range(1, L + 1), eps_bath):
        for spin in range(2):
            p = mode(b_site, spin)
            H += FermionOperator(((p, 1), (p, 0)), float(eps_b))
    for b_site, V_b in zip(range(1, L + 1), V_bath):
        for spin in range(2):
            imp_m, bath_m = mode(0, spin), mode(b_site, spin)
            H += FermionOperator(((imp_m, 1), (bath_m, 0)), float(V_b))
            H += FermionOperator(((bath_m, 1), (imp_m, 0)), float(V_b))
    return H


def tv2d(L=2, t=1.0, V=1.0):
    """2D spinless t-V on an LxL periodic square lattice (ratio V/t).

        H = -t sum_<ij>(c+_i c_j + h.c.) + V sum_<ij>(n_i-1/2)(n_j-1/2)

    site = iy*L + ix; bonds to the right and up neighbours (mod L).
    """
    def site(ix, iy):
        return iy * L + ix

    H = FermionOperator()
    for iy in range(L):
        for ix in range(L):
            i = site(ix, iy)
            for jx, jy in (((ix + 1) % L, iy), (ix, (iy + 1) % L)):
                j = site(jx, jy)
                H += FermionOperator(((i, 1), (j, 0)), -t)
                H += FermionOperator(((j, 1), (i, 0)), -t)
                H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)
                H += FermionOperator(((i, 1), (i, 0)), -V / 2)
                H += FermionOperator(((j, 1), (j, 0)), -V / 2)
                H += FermionOperator((), V / 4)
    return H


def hubbard2d(L=2, t=1.0, U=2.0):
    """2D Fermi-Hubbard on an LxL periodic square lattice (ratio U/t).

        H = -t sum_<ij>,s (c+_is c_js + h.c.) + U sum_i n_i,up n_i,dn

    mode = 2*site + spin; site = iy*L + ix.
    """
    def site(ix, iy):
        return iy * L + ix

    def mode(s, spin):
        return 2 * s + spin

    H = FermionOperator()
    for iy in range(L):
        for ix in range(L):
            i = site(ix, iy)
            for jx, jy in (((ix + 1) % L, iy), (ix, (iy + 1) % L)):
                j = site(jx, jy)
                for spin in range(2):
                    H += FermionOperator(((mode(i, spin), 1), (mode(j, spin), 0)), -t)
                    H += FermionOperator(((mode(j, spin), 1), (mode(i, spin), 0)), -t)
            up, dn = mode(i, 0), mode(i, 1)
            H += FermionOperator(((up, 1), (dn, 1), (dn, 0), (up, 0)), U)
    return H


def hubbard2d_rect(Lx=2, Ly=3, t=1.0, U=2.0, periodic=False):
    """2D Fermi-Hubbard on an Lx x Ly rectangular lattice (ratio U/t).

        H = -t sum_<ij>,s (c+_is c_js + h.c.) + U sum_i n_i,up n_i,dn

    site = iy*Lx + ix; mode = 2*site + spin.  Same convention as hubbard2d
    (plain U n_up n_dn, no -U/2 shift).  Open BC by default; periodic wraps a
    direction only when that dimension >= 3 (a length-2 ring would double-count
    the single bond).  hubbard2d_rect(L, L, periodic=True) == hubbard2d(L).
    """
    def site(ix, iy):
        return iy * Lx + ix

    def mode(s, spin):
        return 2 * s + spin

    bonds = []
    for iy in range(Ly):
        for ix in range(Lx):
            i = site(ix, iy)
            if ix + 1 < Lx:
                bonds.append((i, site(ix + 1, iy)))
            elif periodic and Lx >= 3:
                bonds.append((i, site(0, iy)))
            if iy + 1 < Ly:
                bonds.append((i, site(ix, iy + 1)))
            elif periodic and Ly >= 3:
                bonds.append((i, site(ix, 0)))

    H = FermionOperator()
    for (i, j) in bonds:
        for spin in range(2):
            H += FermionOperator(((mode(i, spin), 1), (mode(j, spin), 0)), -t)
            H += FermionOperator(((mode(j, spin), 1), (mode(i, spin), 0)), -t)
    for s in range(Lx * Ly):
        up, dn = mode(s, 0), mode(s, 1)
        H += FermionOperator(((up, 1), (dn, 1), (dn, 0), (up, 0)), U)
    return H


def kitaev(L=4, t=1.0, Delta=1.0, mu=0.0):
    """Kitaev chain (open boundary conditions, ratio Delta/t, mu=0 sweet spot).

        H = -mu sum_j (n_j - 1/2)
            + sum_{j<L-1} [ -t (c+_{j+1} c_j + h.c.) + Delta (c+_j c+_{j+1} + h.c.) ]
    """
    H = FermionOperator()
    for j in range(L):
        H += FermionOperator(((j, 1), (j, 0)), -mu)
        H += FermionOperator((), mu / 2.0)
    for j in range(L - 1):
        jp1 = j + 1
        H += FermionOperator(((jp1, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (jp1, 0)), -t)
        H += FermionOperator(((j, 1), (jp1, 1)), -Delta)
        H += FermionOperator(((jp1, 0), (j, 0)), -Delta)
    return H


def pam(L=2, t=1.0, V=1.0, U=2.0):
    """Periodic Anderson Model (1D ring of L sites, 4 modes/site, ratio U/t).

    Particle-hole symmetric: eps_f = -U/2.  Modes per site j:
        4j+0,4j+1 = conduction c up/dn;  4j+2,4j+3 = localised f up/dn.
    """
    eps_f = -U / 2.0

    def c(j, spin):
        return 4 * j + spin

    def f(j, spin):
        return 4 * j + 2 + spin

    H = FermionOperator()
    for j in range(L):
        jp = (j + 1) % L
        for spin in range(2):
            H += FermionOperator(((c(j, spin), 1), (c(jp, spin), 0)), -t)
            H += FermionOperator(((c(jp, spin), 1), (c(j, spin), 0)), -t)
        for spin in range(2):
            H += FermionOperator(((f(j, spin), 1), (f(j, spin), 0)), eps_f)
        f_up, f_dn = f(j, 0), f(j, 1)
        H += FermionOperator(((f_up, 1), (f_dn, 1), (f_dn, 0), (f_up, 0)), U)
        for spin in range(2):
            H += FermionOperator(((c(j, spin), 1), (f(j, spin), 0)), V)
            H += FermionOperator(((f(j, spin), 1), (c(j, spin), 0)), V)
    return H


def j1j2_1d(L=4, J1=1.0, J2=0.5):
    """1D J1-J2 frustrated Heisenberg chain (periodic, ratio J2/J1).

    Spin model -> returns a QubitOperator (BK is skipped downstream).
        H = J1 sum_j S_j.S_{j+1} + J2 sum_j S_j.S_{j+2},
        S_i.S_j = (1/4)(X_iX_j + Y_iY_j + Z_iZ_j).
    """
    H = QubitOperator()
    for j in range(L):
        jp1, jp2 = (j + 1) % L, (j + 2) % L
        for p in ('X', 'Y', 'Z'):
            H += QubitOperator(f'{p}{j} {p}{jp1}', J1 / 4.0)
            H += QubitOperator(f'{p}{j} {p}{jp2}', J2 / 4.0)
    return H


def j1j2_2d(L=2, J1=1.0, J2=0.5):
    """2D J1-J2 Heisenberg on an LxL periodic square lattice (ratio J2/J1).

    Spin model -> returns a QubitOperator.  J1 on nearest neighbours
    (right/up), J2 on next-nearest neighbours (the two diagonals).
    """
    def site(ix, iy):
        return iy * L + ix

    H = QubitOperator()
    for iy in range(L):
        for ix in range(L):
            i = site(ix, iy)
            for jx, jy in (((ix + 1) % L, iy), (ix, (iy + 1) % L)):
                j = site(jx, jy)
                for p in ('X', 'Y', 'Z'):
                    H += QubitOperator(f'{p}{i} {p}{j}', J1 / 4.0)
            for jx, jy in (((ix + 1) % L, (iy + 1) % L),
                           ((ix - 1) % L, (iy + 1) % L)):
                j = site(jx, jy)
                for p in ('X', 'Y', 'Z'):
                    H += QubitOperator(f'{p}{i} {p}{j}', J2 / 4.0)
    return H


MODELS = {
    "tv1d": tv1d,
    "tv2d": tv2d,
    "hubbard1d": hubbard1d,
    "hubbard2d": hubbard2d,
    "hubbard2d_2x3": lambda L=2, t=1.0, U=2.0: hubbard2d_rect(2, 3, t, U),  # fixed 2x3 OBC; L ignored
    "siam": siam,
    "pam": pam,
    "kitaev": kitaev,
    "j1j2_1d": j1j2_1d,
    "j1j2_2d": j1j2_2d,
}

# Models whose builder returns a QubitOperator (spin models): BK is skipped.
SPIN_MODELS = {"j1j2_1d", "j1j2_2d"}


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 - Bravyi-Kitaev transformation
# ═════════════════════════════════════════════════════════════════════════════

def bk_transform(H_fermion):
    """FermionOperator -> BK-mapped QubitOperator (real, compressed)."""
    H_bk = bravyi_kitaev(H_fermion)
    H_bk.compress()
    return H_bk


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 - Z2 symmetry tapering (GF(2) parity-check / nullspace, fieldmath)
# ═════════════════════════════════════════════════════════════════════════════

def _symmetry_generators(qubit_H):
    """Return Z2 stabiliser generators (list of QubitOperator) of qubit_H.

    Builds the binary symplectic parity-check matrix E of the Pauli terms,
    reduces it over GF(2), and reads symmetry Paulis off the nullspace.
    """
    n = count_qubits(qubit_H)
    terms = list(qubit_H.terms.keys())
    num_terms = len(terms)
    if num_terms == 0 or n == 0:
        return []

    field = fm.PrimeField(2)
    E = fm.Matrix(num_terms, 2 * n, field)
    for i, term in enumerate(terms):
        spot = 0
        for j in range(n):
            try:
                if term[spot][0] == j:
                    char = term[spot][1]
                    spot += 1
                else:
                    char = 'I'
            except IndexError:
                char = 'I'
            x, z = {'I': (0, 0), 'X': (0, 1), 'Y': (1, 1), 'Z': (1, 0)}[char]
            E.set(i, j, x)
            E.set(i, j + n, z)

    E.reduced_row_echelon_form()
    E_red = np.array([[E.get(r, c) for c in range(2 * n)]
                      for r in range(num_terms)], dtype=int)
    del E
    while len(E_red) and (E_red[-1] == 0).all():
        E_red = E_red[:-1]

    # nullspace of the (reduced) parity matrix
    Et = E_red.transpose()
    pivots, first_entries = [], []
    for col in range(len(Et)):
        try:
            fe = list(Et[col]).index(1)
        except ValueError:
            continue
        if all(Et[c2][fe] == 0 for c2 in range(col)):
            pivots.append(col)
            first_entries.append(fe)
    nonpivots = sorted(set(range(len(Et))) - set(pivots))

    nullspace = []
    for col in nonpivots:
        nv = [0] * (2 * n)
        nv[col] = 1
        for i in range(2 * n):
            if i in pivots:
                fe = first_entries[pivots.index(i)]
                if Et[col][fe] == 1:
                    nv[i] = 1
        nullspace.append(nv)

    generators = []
    for nv in nullspace:
        s = ''
        for j in range(n):
            x, z = nv[j], nv[j + n]
            char = {(0, 0): 'I', (1, 0): 'X', (1, 1): 'Y', (0, 1): 'Z'}[(x, z)]
            if char != 'I':
                s += f'{char}{j} '
        generators.append(QubitOperator(s.strip()))
    return generators


def _pauli_symplectic(qop, n):
    """Single-Pauli QubitOperator -> length-2n GF(2) symplectic vector (x|z)."""
    term = next(iter(qop.terms))           # the one Pauli string
    v = [0] * (2 * n)
    for (idx, ch) in term:
        if ch in ('X', 'Y'):
            v[idx] = 1
        if ch in ('Z', 'Y'):
            v[idx + n] = 1
    return v


def _commutes(a, b):
    c = commutator(a, b)
    c.compress()
    return len(c.terms) == 0


def _valid_stabilizers(qubit_H, candidates):
    """Select a maximal subset of `candidates` that (i) commute with H,
    (ii) mutually commute, and (iii) are GF(2)-independent.

    This guarantees a genuine abelian stabiliser group, so taper_off_qubits
    can never manufacture a sector whose spectrum lies below E0.  Validity is
    checked convention-free via OpenFermion's commutator, so it does not rely
    on the X/Z bookkeeping in the nullspace construction above.
    """
    n = count_qubits(qubit_H)
    chosen, basis = [], []                  # basis: row-reduced symplectic vecs
    for g in candidates:
        if not g.terms or () in g.terms:
            continue
        if not _commutes(qubit_H, g):
            continue
        if not all(_commutes(g, h) for h in chosen):
            continue
        # GF(2) independence vs already-chosen generators
        v = _pauli_symplectic(g, n)
        r = v[:]
        for pivot, brow in basis:
            if r[pivot]:
                r = [(ri ^ bi) for ri, bi in zip(r, brow)]
        if not any(r):
            continue                        # dependent -> skip
        pivot = next(i for i, ri in enumerate(r) if ri)
        basis.append((pivot, r))
        chosen.append(g)
    return chosen


def z2_taper(qubit_H):
    """Split qubit_H into all Z2 symmetry sectors.

    Returns a list of tapered QubitOperators (one per choice of stabiliser
    eigenvalues +/-1).  If no valid symmetries are found, returns [qubit_H].
    """
    candidates = _symmetry_generators(qubit_H)
    generators = _valid_stabilizers(qubit_H, candidates)
    if not generators:
        return [qubit_H]

    sectors = []
    for signs in iproduct([1, -1], repeat=len(generators)):
        signed = [signs[i] * generators[i] for i in range(len(generators))]
        sec = taper_off_qubits(qubit_H, signed)
        sec.compress()
        sectors.append(sec)
    return sectors


# ═════════════════════════════════════════════════════════════════════════════
# Ising-polynomial arithmetic   { frozenset(spin indices) -> float }
# (Z_a Z_a = I  =>  product of monomials == symmetric difference of index sets)
# ═════════════════════════════════════════════════════════════════════════════

def _poly_mul(p1, p2):
    out = defaultdict(complex)
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            out[s1.symmetric_difference(s2)] += c1 * c2
    return dict(out)


def _poly_clean(poly, tol=1e-12):
    return {m: v.real for m, v in poly.items() if abs(v.real) > tol}


def _qubitop_to_terms(qubit_op):
    """QubitOperator -> [(real_coeff, {qubit: pauli_char})], indices remapped
    to a contiguous 0..n-1 range so the XBK copy layout has no gaps."""
    active = sorted({idx for term in qubit_op.terms
                     for (idx, _) in term})
    remap = {q: i for i, q in enumerate(active)}
    terms = []
    for term, coeff in qubit_op.terms.items():
        c = float(np.real(coeff))
        if abs(c) <= 1e-14:
            continue
        terms.append((c, {remap[idx]: op for (idx, op) in term}))
    return terms, len(active)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4a - XBK transform of H and the counting operator C (Z-only Ising polys)
# physical spin for logical qubit i on copy j (0-based):  phys = i + n*j
# ═════════════════════════════════════════════════════════════════════════════

def _xbk_single(pauli, i, j, k, n):
    """<copy_j| P_i |copy_k> as an Ising polynomial (P_i acts on logical i)."""
    pj, pk = i + n * j, i + n * k
    if j == k:                                  # same copy
        if pauli in ('X', 'Y'):
            return {}                           # vanishes
        if pauli == 'Z':
            return {frozenset([pj]): 1.0 + 0j}
        return {frozenset(): 1.0 + 0j}          # I
    # cross copy (j != k)
    if pauli == 'X':
        return {frozenset(): 0.5 + 0j, frozenset([pj, pk]): -0.5 + 0j}
    if pauli == 'Y':
        return {frozenset([pk]): 0.5j, frozenset([pj]): -0.5j}
    if pauli == 'Z':
        return {frozenset([pj]): 0.5 + 0j, frozenset([pk]): 0.5 + 0j}
    return {frozenset(): 0.5 + 0j, frozenset([pj, pk]): 0.5 + 0j}   # I


def build_xbk_ising(terms, n, r, p):
    """H'_p(s) as a Z-only Ising polynomial over r*n copy-spins."""
    Sp = [-1.0 if j < p else 1.0 for j in range(r)]
    H = defaultdict(complex)
    for j in range(r):
        for k in range(r):
            sign = Sp[j] * Sp[k]
            for coeff, pd in terms:
                term = {frozenset(): 1.0 + 0j}
                for i in range(n):
                    term = _poly_mul(term, _xbk_single(pd.get(i, 'I'), i, j, k, n))
                    if not term:
                        break
                for mono, val in term.items():
                    H[mono] += sign * coeff * val
    return _poly_clean(dict(H))


def build_counting(n, r, p):
    """C_p(s) = <psi|psi> for the XBK ansatz, as a Z-only Ising polynomial.

    Closed form (avoids the 2^n basis-state sum the naive definition implies):

        C = sum_{j,k} S'(j) S'(k) <copy_j|copy_k>,
        <copy_j|copy_k> = prod_i (1 + Z_{i,j} Z_{i,k}) / 2,

    using  sum_b prod_i (1+(-1)^{b_i} Z_{i,j})/2 (1+(-1)^{b_i} Z_{i,k})/2
         = prod_i (1 + Z_{i,j} Z_{i,k})/2.

    This is O(r^2 n 2^n) instead of the naive O(2^{3n}); for the tapered SIAM at
    N_bath=5 (n=10) it drops build time from hours to well under a second.  The
    j==k self-overlap is the identity (each copy is a unit-norm basis state).
    """
    Sp = [-1.0 if j < p else 1.0 for j in range(r)]
    C = defaultdict(complex)
    for j in range(r):
        for k in range(r):
            sign = Sp[j] * Sp[k]
            term = {frozenset(): 1.0 + 0j}
            if j != k:
                for i in range(n):
                    pj, pk = i + n * j, i + n * k
                    term = _poly_mul(term, {frozenset(): 0.5 + 0j,
                                            frozenset([pj, pk]): 0.5 + 0j})
            for m, v in term.items():
                C[m] += sign * v
    return _poly_clean(dict(C))


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4b - quadratisation (Rosenberg): k-local Ising -> 2-local
# x_i = (1-s_i)/2;  penalty M*(x_a x_b - 2 w x_a - 2 w x_b + 3 w) >= 0
# ═════════════════════════════════════════════════════════════════════════════

def _ising_to_binary(ising):
    out = defaultdict(float)
    for S, c in ising.items():
        sl = sorted(S)
        k = len(sl)
        for tb in range(1 << k):
            T = frozenset(sl[m] for m in range(k) if (tb >> m) & 1)
            out[T] += c * ((-2) ** len(T))
    return dict(out)


def _binary_to_ising(binary):
    out = defaultdict(float)
    for T, c in binary.items():
        tl = sorted(T)
        k = len(tl)
        for ub in range(1 << k):
            U = frozenset(tl[m] for m in range(k) if (ub >> m) & 1)
            out[U] += c * ((-1) ** len(U)) / (2 ** k)
    return dict(out)


def quadratize(binary, M, next_id):
    """Reduce a binary polynomial to degree <= 2. Returns (poly, next_id)."""
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
            w = next_id
            next_id += 1
            poly[frozenset([w]) | rest] += c
            poly[frozenset([a, b])] += M
            poly[frozenset([w, a])] += -2.0 * M
            poly[frozenset([w, b])] += -2.0 * M
            poly[frozenset([w])] += 3.0 * M
            changed = True
            break
    return dict(poly), next_id


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5 - Simulated Annealing over an Ising polynomial
# ═════════════════════════════════════════════════════════════════════════════

def _eval(ising, spins):
    E = 0.0
    for mono, c in ising.items():
        p = 1.0
        for i in mono:
            p *= spins[i]
        E += c * p
    return E


def _build_index(ising):
    """For each spin, the list of (other-spins, coeff) monomials touching it."""
    idx = defaultdict(list)
    for mono, c in ising.items():
        for i in mono:
            idx[i].append((tuple(j for j in mono if j != i), c))
    return idx


def _dE(idx, spins, flip):
    si = spins[flip]
    field = 0.0
    for others, c in idx[flip]:
        p = c
        for j in others:
            p *= spins[j]
        field += p
    return -2.0 * si * field


def _sa_chain(ising, n_spins, n_sweeps, T_start, T_end, rng):
    idx = _build_index(ising)
    spins = {i: rng.choice((-1, 1)) for i in range(n_spins)}
    E = _eval(ising, spins)
    best_s, best_E = dict(spins), E
    alpha = (T_end / T_start) ** (1.0 / max(n_sweeps, 1))
    T = T_start
    for _ in range(n_sweeps):
        for __ in range(n_spins):
            i = rng.randrange(n_spins)
            d = _dE(idx, spins, i)
            if d < 0 or rng.random() < math.exp(-min(d / T, 700.0)):
                spins[i] = -spins[i]
                E += d
                if E < best_E:
                    best_E, best_s = E, dict(spins)
        T *= alpha
    return best_s, best_E


def _sa_chain_seed(packed):
    """Top-level (picklable) worker: run one SA chain from an integer seed."""
    import random
    ising, n_spins, n_sweeps, T_start, T_end, seed = packed
    return _sa_chain(ising, n_spins, n_sweeps, T_start, T_end,
                     random.Random(seed))


def sa_run(ising, n_spins, n_restarts, n_sweeps, T_start, T_end, seed,
           workers=None):
    """Multi-restart SA. Returns (best_spins, best_energy).

    The n_restarts chains are independent, so they are run across `workers`
    processes (default SA_WORKERS = SLURM_CPUS_PER_TASK).  This is the
    intra-task parallelism that keeps a single heavy XBK solve (e.g. SIAM
    N_bath=5) from bottlenecking the MPI task pool.  workers<=1 runs serially.
    """
    if workers is None:
        workers = SA_WORKERS
    packed = [(ising, n_spins, n_sweeps, T_start, T_end, seed + 7919 * c)
              for c in range(n_restarts)]

    if workers > 1 and n_restarts > 1:
        with Pool(processes=min(workers, n_restarts)) as pool:
            results = pool.map(_sa_chain_seed, packed)
    else:
        results = [_sa_chain_seed(p) for p in packed]

    return min(results, key=lambda r: r[1])


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4c - Dinkelbach fractional programming:  E0 = min_s H'(s)/C(s)
# ═════════════════════════════════════════════════════════════════════════════

def _sa_budget(n_spins, n_restarts, n_sweeps):
    """Adaptive SA budget. If caller passes None, scale with the spin count so
    larger r (more physical spins) automatically anneals harder/longer.

    Tuned on the r=4 XBK targets (tv1d L3/L4, hubbard L2): ~150 sweeps + 2
    restarts per spin reaches the true optimum on every case across seeds, with
    headroom (100/2 already suffices on the hardest, PBC L3).
    """
    if n_sweeps is None:
        n_sweeps = max(800, 150 * n_spins)
    if n_restarts is None:
        n_restarts = max(6, 2 * n_spins)
    return n_restarts, n_sweeps


def dinkelbach(H_ising, C_ising, n_phys, penalty_scale, seed,
               n_restarts=None, n_sweeps=None, max_iter=30, tol=1e-6,
               quadratic=False):
    """Minimise the ratio H'(s)/C(s) over the n_phys physical XBK spins.

    H_ising / C_ising are the original (k-local) physical Ising polynomials.
    Each Dinkelbach iteration forms the combined objective (H - lam*C) and
    minimises it by simulated annealing, then re-evaluates the ratio on the
    physical polynomials using only the physical spins.  Returns best E0.

    Inner-SA modes
    --------------
    quadratic=False (default, recommended):
        anneal the k-local combined objective directly on the n_phys physical
        spins.  Our SA is k-local-native, so this avoids the Rosenberg
        quadratisation blow-up (e.g. 8 -> 44 spins at r=4) entirely.  Faster
        AND more reliable -- this is what lets r=4 hit its true optimum.

    quadratic=True:
        quadratise (H - lam*C) *fresh* each iteration to a 2-local QUBO before
        annealing (penalties keep the correct +M sign for either sign of lam).
        Use only when you need a genuine 2-local QUBO, e.g. to hand off to a
        hardware sampler such as D-Wave.

    The ratio is ALWAYS evaluated on the physical k-local polynomials with the
    physical spins only -- never on the penalty-bearing quadratised polynomial.
    """
    n_eff = n_phys if not quadratic else max(n_phys, 4 * n_phys)
    n_restarts, n_sweeps = _sa_budget(n_eff, n_restarts, n_sweeps)

    lam = 0.0
    best_E0 = float('inf')
    base = max((abs(c) for c in H_ising.values()), default=1.0) \
        + max((abs(c) for c in C_ising.values()), default=1.0)

    for it in range(max_iter):
        # physical combined polynomial H - lam*C
        combined = defaultdict(float, H_ising)
        for m, c in C_ising.items():
            combined[m] -= lam * c
        combined = {m: v for m, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            break

        peak = max(abs(v) for v in combined.values())

        if quadratic:
            # penalty must dominate every coefficient of the combined polynomial
            M = penalty_scale * max(base, peak)
            T0 = max(3.0 * M, 5.0)
            obj, n_obj = quadratize(_ising_to_binary(combined), M, n_phys)
            obj = _binary_to_ising(obj)
        else:
            obj, n_obj = combined, n_phys
            # warm enough to cross barriers ~ magnitude of the objective
            T0 = max(2.0 * max(base, peak), 5.0)

        # T_end warm (1e-3, not 1e-4): freezing too cold strands chains in
        # local minima -- a key part of reaching the r=4 optimum.
        spins, _ = sa_run(obj, n_obj, n_restarts, n_sweeps,
                          T0, 1e-3, seed + it * 1009)

        # ratio on the PHYSICAL polynomials, physical spins only
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


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 - XBK on one qubit operator (sweeps p, returns min E0)
# ═════════════════════════════════════════════════════════════════════════════

def run_xbk(qubit_H, r=2, penalty_scale=12.0, n_restarts=None, n_sweeps=None,
            seed=0, verbose=False, quadratic=False):
    """Full XBK + Dinkelbach + SA on a single QubitOperator. Returns E0.

    n_restarts / n_sweeps default to None -> adaptive budget that scales with
    the physical-spin count (see _sa_budget), so raising r anneals harder
    automatically.  Pass explicit ints to override.
    """
    terms, n = _qubitop_to_terms(qubit_H)
    if n == 0:                                  # pure constant
        return float(np.real(qubit_H.terms.get((), 0.0)))

    n_phys = n * r

    best_E0 = float('inf')
    for p in range(int(math.ceil(r / 2)) + 1):
        H_is = build_xbk_ising(terms, n, r, p)
        C_is = build_counting(n, r, p)
        if not C_is:
            continue

        E0 = dinkelbach(H_is, C_is, n_phys, penalty_scale, seed + p * 37,
                        n_restarts=n_restarts, n_sweeps=n_sweeps,
                        quadratic=quadratic)
        if verbose:
            print(f"      p={p}: E0={E0:.6f}")
        best_E0 = min(best_E0, E0)
    return best_E0


# ═════════════════════════════════════════════════════════════════════════════
# Full pipeline:  FermionOperator OR QubitOperator -> E0
# ═════════════════════════════════════════════════════════════════════════════

def ground_state_energy(H, r=2, taper=True, penalty_scale=12.0,
                        n_restarts=None, n_sweeps=None, seed=0, verbose=False,
                        quadratic=False, fermionic=None):
    """Run the complete pipeline on a model Hamiltonian and return E0.

    H may be a FermionOperator (fermionic model -> BK is applied) or a
    QubitOperator (spin model, e.g. J1-J2 Heisenberg -> BK is skipped).  The
    type is auto-detected; override with fermionic=True/False.

    [BK ->] Z2 taper -> (per sector) XBK/Dinkelbach/SA -> min over sectors.

    n_restarts / n_sweeps default to None -> adaptive SA budget scaled to the
    physical-spin count, so larger r anneals harder automatically and reaches
    its variational optimum.  Pass ints to pin a fixed budget.
    """
    if fermionic is None:
        fermionic = isinstance(H, FermionOperator)
    H_qubit = bk_transform(H) if fermionic else H
    if not isinstance(H_qubit, QubitOperator):
        raise TypeError("spin model Hamiltonian must be a QubitOperator")
    H_qubit.compress()

    sectors = z2_taper(H_qubit) if taper else [H_qubit]
    if verbose:
        tag = "BK qubits" if fermionic else "qubits"
        print(f"  {tag} = {count_qubits(H_qubit)}, "
              f"sectors after tapering = {len(sectors)}")

    best_E0 = float('inf')
    for si, sec in enumerate(sectors):
        nq = count_qubits(sec)
        if verbose:
            print(f"  sector {si}: {nq} qubits, {len(sec.terms)} terms")
        E0 = run_xbk(sec, r=r, penalty_scale=penalty_scale,
                     n_restarts=n_restarts, n_sweeps=n_sweeps,
                     seed=seed + si * 101, verbose=verbose, quadratic=quadratic)
        best_E0 = min(best_E0, E0)
    return best_E0


def exact_ground_energy(H, n_qubits=None):
    """Exact diagonalisation reference (dense; for small systems).

    Accepts a FermionOperator or QubitOperator -- get_sparse_operator handles
    both (Jordan-Wigner for fermions, which preserves the spectrum).
    """
    if n_qubits is None:
        n_qubits = count_qubits(H)
    M = get_sparse_operator(H, n_qubits=n_qubits).toarray()
    return float(np.linalg.eigvalsh(M)[0].real)


# ═════════════════════════════════════════════════════════════════════════════
# CLI / demo
# ═════════════════════════════════════════════════════════════════════════════

def _main():
    ap = argparse.ArgumentParser(description="XBK ground-state pipeline")
    ap.add_argument("--model", default="tv1d", choices=list(MODELS),
                    help="built-in model name")
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--V", type=float, default=1.0, help="t-V interaction")
    ap.add_argument("--U", type=float, default=2.0, help="Hubbard/SIAM/PAM U")
    ap.add_argument("--Delta", type=float, default=1.0, help="Kitaev pairing")
    ap.add_argument("--J2", type=float, default=0.5, help="J1-J2 next-NN coupling")
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--r", type=int, default=2, help="XBK copies")
    ap.add_argument("--no-taper", action="store_true")
    ap.add_argument("--restarts", type=int, default=None,
                    help="SA restarts (default: adaptive, scales with spins)")
    ap.add_argument("--sweeps", type=int, default=None,
                    help="SA sweeps (default: adaptive, scales with spins)")
    ap.add_argument("--quadratic", action="store_true",
                    help="anneal a 2-local QUBO (Rosenberg) instead of k-local")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    builders = {
        "tv1d":      lambda: tv1d(args.L, args.t, args.V, periodic=args.periodic),
        "tv2d":      lambda: tv2d(args.L, args.t, args.V),
        "hubbard1d": lambda: hubbard1d(args.L, args.t, args.U, periodic=args.periodic),
        "hubbard2d": lambda: hubbard2d(args.L, args.t, args.U),
        "siam":      lambda: siam(args.L, U=args.U, t=args.t),
        "pam":       lambda: pam(args.L, args.t, args.V, args.U),
        "kitaev":    lambda: kitaev(args.L, args.t, args.Delta),
        "j1j2_1d":   lambda: j1j2_1d(args.L, 1.0, args.J2),
        "j1j2_2d":   lambda: j1j2_2d(args.L, 1.0, args.J2),
    }
    H = builders[args.model]()

    n = count_qubits(H)
    print(f"Model={args.model}  L={args.L}  qubits={n}  r={args.r}  "
          f"taper={not args.no_taper}")

    E_exact = exact_ground_energy(H, n)
    E_xbk = ground_state_energy(H, r=args.r, taper=not args.no_taper,
                                n_restarts=args.restarts, n_sweeps=args.sweeps,
                                seed=args.seed, verbose=True,
                                quadratic=args.quadratic)
    print(f"\n  E0 (exact ED) = {E_exact:.6f}")
    print(f"  E0 (XBK+SA)   = {E_xbk:.6f}")
    print(f"  abs error     = {abs(E_xbk - E_exact):.2e}")


if __name__ == "__main__":
    _main()
