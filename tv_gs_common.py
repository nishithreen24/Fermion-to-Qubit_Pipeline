#!/usr/bin/env python3
"""
tv_gs_common.py -- shared utilities for the SA/SQA ground-state-energy solvers
for the 1D spinless t-V model (Levy SQA + Yip SA subspace).

Convention (identical to tv_levy.py::tv1d and tv1d_exact_E0.py, OBC, PH-sym):
    H = -t sum_<ij>(c+_i c_j + h.c.) + V sum_<ij>(n_i-1/2)(n_j-1/2)

Jordan-Wigner image = the XXZ spin-1/2 chain (spin up == occupied):
    H_spin = sum_<ij> [ -t/2 (X_iX_j + Y_iY_j) + V/4 Z_iZ_j ]
with, in the S^z (occupation) basis:
    * diagonal energy of a config  = V/4 sum_<ij> z_i z_j,   z = 1 - 2*occ  (= +/-1)
    * off-diagonal (hopping): a bond with opposite occupations connects the two
      swapped configs with matrix element -t.  All off-diagonal elements are <= 0
      => the chain is STOQUASTIC on a 1D lattice => sign-problem-free for PIMC/SSE.
      (Caveat: odd-L PBC can carry a Marshall sign; OBC used here is always safe.)

Ground-state sector = half filling N = L//2 (for odd L this is one of the two
degenerate (L +/- 1)/2 sectors -- verified against ED).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from itertools import combinations


# ---------------------------------------------------------------------------
# Fermionic Hamiltonian (for the ED reference only)
# ---------------------------------------------------------------------------
def tv1d_fermion(L, t=1.0, V=2.0):
    """OBC particle-hole-symmetric t-V FermionOperator (matches tv_levy.py)."""
    from openfermion import FermionOperator
    H = FermionOperator()
    for i in range(L - 1):
        j = i + 1
        H += FermionOperator(((i, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (i, 0)), -t)
        H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)
        H += FermionOperator(((i, 1), (i, 0)), -V / 2)
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator((), V / 4)
    return H


def bonds_obc(L):
    return [(i, i + 1) for i in range(L - 1)]


# ---------------------------------------------------------------------------
# ED reference (small L only) -- global Fock-space ground state
# ---------------------------------------------------------------------------
def ed_reference(L, t=1.0, V=2.0):
    """Exact E0 over the full 2^L Fock space via sparse Lanczos."""
    from openfermion import jordan_wigner, get_sparse_operator
    H = get_sparse_operator(jordan_wigner(tv1d_fermion(L, t, V)),
                            n_qubits=L).real.tocsr()
    if H.shape[0] <= 512:
        return float(np.linalg.eigvalsh(H.toarray()).min())
    return float(sla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])


# ---------------------------------------------------------------------------
# Occupation-string (S^z sector) machinery -- shared by SSE / PIMC / Yip
# ---------------------------------------------------------------------------
def half_filling_N(L):
    return L // 2


def enumerate_sector(L, N):
    """All length-L occupation configs (as int bitmasks) with exactly N ones."""
    cfgs = []
    for occ in combinations(range(L), N):
        b = 0
        for k in occ:
            b |= (1 << k)
        cfgs.append(b)
    return cfgs


def zvals(cfg, L):
    """z_i = 1 - 2*occ_i  in {+1,-1}, as an int8 array."""
    return np.array([1 - 2 * ((cfg >> i) & 1) for i in range(L)], dtype=np.int8)


def diag_energy(cfg, L, V, bonds):
    """Diagonal (classical) energy  V/4 sum_<ij> z_i z_j."""
    z = zvals(cfg, L)
    e = 0.0
    for (i, j) in bonds:
        e += z[i] * z[j]
    return 0.25 * V * e


def hop_neighbors(cfg, bonds):
    """Configs reachable by one nearest-neighbour hop (opposite occupations on a
    bond swapped).  Returns list of (new_cfg, matrix_element).  Element = -t is
    applied by the caller; here we just return the connected configs."""
    out = []
    for (i, j) in bonds:
        bi = (cfg >> i) & 1
        bj = (cfg >> j) & 1
        if bi != bj:
            out.append(cfg ^ (1 << i) ^ (1 << j))
    return out


def build_sparse_H(basis, L, t, V, bonds):
    """Sparse XXZ Hamiltonian restricted to an ordered list of config bitmasks.

    Diagonal = V/4 sum z z ; off-diagonal hopping = -t between basis members that
    differ by one adjacent swap (elements outside `basis` are dropped -- this is
    exactly the subspace truncation for the Yip selected-CI energy)."""
    index = {c: k for k, c in enumerate(basis)}
    n = len(basis)
    rows, cols, vals = [], [], []
    for k, c in enumerate(basis):
        rows.append(k); cols.append(k); vals.append(diag_energy(c, L, V, bonds))
        for nb in hop_neighbors(c, bonds):
            m = index.get(nb)
            if m is not None:
                rows.append(k); cols.append(m); vals.append(-t)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def lowest_eig(Hsp):
    """Lowest eigenvalue of a small sparse symmetric matrix."""
    n = Hsp.shape[0]
    if n == 1:
        return float(Hsp[0, 0])
    if n <= 512:
        return float(np.linalg.eigvalsh(Hsp.toarray()).min())
    return float(sla.eigsh(Hsp, k=1, which="SA", return_eigenvectors=False)[0])


# ---------------------------------------------------------------------------
# CSV writer (header style matches the existing tv_levy/tv_yip CSVs)
# ---------------------------------------------------------------------------
def write_csv(path, columns, rows, header_meta):
    """columns: list of names; rows: list of tuples; header_meta: str appended
    after the column line as a '# ...' comment (same look as tv_levy CSVs)."""
    with open(path, "w") as f:
        f.write(",".join(columns) + "  # " + header_meta + "\n")
        for r in rows:
            f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x)
                             for x in r) + "\n")


# ---------------------------------------------------------------------------
# self-test: verify the spin encoding reproduces ED in the GS sector
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t, V = 1.0, 2.0
    print(f"{'L':>3} {'N':>3} {'E0_ED(full)':>12} {'E0_sector(spin)':>16} {'match':>6}")
    for L in range(3, 9):
        bonds = bonds_obc(L)
        N = half_filling_N(L)
        basis = enumerate_sector(L, N)
        Hsp = build_sparse_H(basis, L, t, V, bonds)
        e_spin = lowest_eig(Hsp)
        e_ed = ed_reference(L, t, V)
        ok = abs(e_spin - e_ed) < 1e-9
        print(f"{L:>3} {N:>3} {e_ed:>12.6f} {e_spin:>16.6f} {str(ok):>6}")
