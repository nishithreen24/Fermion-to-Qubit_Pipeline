#!/usr/bin/env python3
"""
hubbard2d_ed.py -- exact (Lanczos) ground-state energy of the 2D Hubbard model

    H = -t sum_<ij>,s (c^+_is c_js + h.c.)  +  U sum_i n_iu n_id

on an Lx x Ly square lattice (OBC or PBC) in the fixed (N_up, N_dn) sector.

Method: the sector factorises as (up configs) x (down configs).  With d = the
number of single-spin configurations, the state is stored as a (d, d) matrix
Psi[a, b] and

    H Psi = T @ Psi  +  Psi @ T.T  +  U * Dc * Psi ,

where T is the (sparse, d x d) single-species hopping matrix with fermionic
signs and Dc[a, b] = #(doubly occupied sites) of (up config a, down config b),
computed as a bool-matrix product.  Because the Hamiltonian conserves each spin
species and the Jordan-Wigner ordering puts all up modes before all down modes,
the fermion signs factorise exactly: sign(up hop) depends only on the up
config, sign(down hop) only on the down config (the up-parity string enters
twice and cancels).

Ground state via a memory-lean 3-vector Lanczos (no reorthogonalisation --
harmless for the LOWEST Ritz value, which is what we report).

4x4 half filling: d = C(16,8) = 12870, sector dim = d^2 = 165,636,900.
One float64 vector = 1.32 GB; peak ~6-7 GB.  Runs on a 16 GB machine.

Validation ladder (all in --selftest):
  * 2x2, 2x3, 2x4: Lanczos vs DENSE numpy.eigh on the explicitly built sector H
  * 3x4: Lanczos vs scipy.sparse eigsh on the explicit kron-built sparse H
  * U=0: E0 equals the filled free-fermion sea (tight-binding eigenvalues)
"""
import argparse
import itertools
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal


# ---------------------------------------------------------------- lattice ----
def square_bonds(Lx, Ly, bc):
    """Unique nearest-neighbour bonds of an Lx x Ly square lattice.
    Site index = x + Lx*y.  PBC wrap bonds deduplicated (matters for L=2)."""
    bonds = set()
    for y in range(Ly):
        for x in range(Lx):
            i = x + Lx * y
            nbrs = []
            if x + 1 < Lx: nbrs.append((x + 1) + Lx * y)
            elif bc == "pbc" and Lx > 1: nbrs.append(0 + Lx * y)
            if y + 1 < Ly: nbrs.append(x + Lx * (y + 1))
            elif bc == "pbc" and Ly > 1: nbrs.append(x + 0)
            for j in nbrs:
                if i != j:
                    bonds.add((min(i, j), max(i, j)))
    return sorted(bonds)


def free_fermion_E0(Lx, Ly, t, bc, Nup, Ndn):
    """U=0 reference: fill the lowest single-particle levels per spin."""
    N = Lx * Ly
    T1 = np.zeros((N, N))
    for i, j in square_bonds(Lx, Ly, bc):
        T1[i, j] = T1[j, i] = -t
    eps = np.sort(np.linalg.eigvalsh(T1))
    return eps[:Nup].sum() + eps[:Ndn].sum()


# ------------------------------------------------- single-species machinery ----
def sector_configs(N, Np):
    """All N-site bitmasks with Np particles, plus mask -> index dict."""
    confs = []
    for occ in itertools.combinations(range(N), Np):
        m = 0
        for o in occ:
            m |= (1 << o)
        confs.append(m)
    confs.sort()
    return np.array(confs, dtype=np.int64), {m: k for k, m in enumerate(confs)}


def hop_matrix(N, t, bonds, confs, index):
    """Sparse single-species hopping matrix with Jordan-Wigner signs.
    T[a,b] = <a| -t sum_bonds (c^+_i c_j + h.c.) |b>."""
    rows, cols, vals = [], [], []
    for b_idx, m in enumerate(confs):
        for (i, j) in bonds:                      # i < j
            bi, bj = 1 << i, 1 << j
            occ_i, occ_j = bool(m & bi), bool(m & bj)
            if occ_i == occ_j:
                continue                          # both empty / both full
            between = m & (((1 << j) - 1) ^ ((1 << (i + 1)) - 1))
            sign = -1.0 if (bin(between).count("1") & 1) else 1.0
            m2 = m ^ bi ^ bj
            rows.append(index[m2]); cols.append(b_idx); vals.append(-t * sign)
    d = len(confs)
    T = sp.csr_matrix((vals, (rows, cols)), shape=(d, d))
    assert (abs(T - T.T)).max() < 1e-12
    return T


def double_occ_matrix(N, confs):
    """Dc[a,b] = number of doubly occupied sites -- via bool matmul (float32)."""
    d = len(confs)
    O = np.zeros((d, N), dtype=np.float32)
    for k, m in enumerate(confs):
        for s in range(N):
            if m & (1 << s):
                O[k, s] = 1.0
    return O @ O.T                                # (d, d) float32


# ----------------------------------------------------------------- Lanczos ----
def lanczos_gs(matvec, dim, shape, maxiter=400, tol=1e-9, seed=7, verbose=True):
    """3-vector Lanczos for the lowest eigenvalue.  State stored with `shape`
    (a (d,d) matrix here); `matvec` maps state -> state.  Returns (E0, niter)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(shape)
    v /= np.linalg.norm(v)
    v_prev = np.zeros_like(v)
    alphas, betas = [], []
    E_old = np.inf
    beta_prev = 0.0
    for k in range(1, maxiter + 1):
        w = matvec(v)
        if beta_prev != 0.0:
            w -= beta_prev * v_prev
        a = float(np.vdot(v, w))
        w -= a * v
        b = float(np.linalg.norm(w))
        alphas.append(a); betas.append(b)
        if k >= 2:
            E0 = eigh_tridiagonal(np.array(alphas), np.array(betas[:-1]),
                                  eigvals_only=True, select="i",
                                  select_range=(0, 0))[0]
            if verbose and (k % 10 == 0):
                print(f"#   Lanczos it {k:>4}  E0 = {E0:.10f}", flush=True)
            if abs(E0 - E_old) < tol:
                return float(E0), k
            E_old = E0
        if b < 1e-13:                              # invariant subspace exhausted
            break
        v_prev, v = v, w / b
        beta_prev = b
    return float(E_old), k


# ------------------------------------------------------------------ solver ----
def hubbard_E0(Lx, Ly, t, U, bc, Nup, Ndn, maxiter=400, tol=1e-9,
               seed=7, verbose=True):
    """Lanczos ground-state energy in the (Nup, Ndn) sector."""
    N = Lx * Ly
    bonds = square_bonds(Lx, Ly, bc)
    up_c, up_i = sector_configs(N, Nup)
    if Ndn == Nup:
        dn_c, dn_i = up_c, up_i
    else:
        dn_c, dn_i = sector_configs(N, Ndn)
    Tup = hop_matrix(N, t, bonds, up_c, up_i)
    Tdn = Tup if Ndn == Nup else hop_matrix(N, t, bonds, dn_c, dn_i)
    Ou = double_occ_matrix(N, up_c) if Ndn == Nup else None
    if Ndn == Nup:
        Dc = Ou
    else:
        du = np.zeros((len(up_c), N), dtype=np.float32)
        for k, m in enumerate(up_c):
            for s in range(N):
                if m & (1 << s): du[k, s] = 1.0
        dd = np.zeros((len(dn_c), N), dtype=np.float32)
        for k, m in enumerate(dn_c):
            for s in range(N):
                if m & (1 << s): dd[k, s] = 1.0
        Dc = du @ dd.T
    dU, dD = len(up_c), len(dn_c)
    if verbose:
        print(f"# Hubbard {Lx}x{Ly} {bc}  t={t} U={U}  N_up={Nup} N_dn={Ndn}"
              f"  sector dim = {dU}x{dD} = {dU*dD:,}", flush=True)

    TdnT = Tdn.T.tocsr()

    def matvec(psi):
        out = Tup @ psi
        out += psi @ TdnT
        out += (U * Dc) * psi if U != 0.0 else 0.0
        return out

    t0 = time.time()
    E0, nit = lanczos_gs(matvec, dU * dD, (dU, dD),
                         maxiter=maxiter, tol=tol, seed=seed, verbose=verbose)
    if verbose:
        print(f"# converged in {nit} iterations, {time.time()-t0:.1f}s", flush=True)
    return E0


# ------------------------------------------------------------- validation ----
def dense_sector_E0(Lx, Ly, t, U, bc, Nup, Ndn):
    """Independent check: explicitly build the full sector H and diagonalise."""
    N = Lx * Ly
    bonds = square_bonds(Lx, Ly, bc)
    up_c, up_i = sector_configs(N, Nup)
    dn_c, dn_i = sector_configs(N, Ndn)
    Tup = hop_matrix(N, t, bonds, up_c, up_i)
    Tdn = hop_matrix(N, t, bonds, dn_c, dn_i)
    du = np.array([[1.0 if m & (1 << s) else 0.0 for s in range(N)] for m in up_c])
    dd = np.array([[1.0 if m & (1 << s) else 0.0 for s in range(N)] for m in dn_c])
    Dc = du @ dd.T
    dU, dD = len(up_c), len(dn_c)
    H = (sp.kron(Tup, sp.identity(dD)) + sp.kron(sp.identity(dU), Tdn)
         + sp.diags((U * Dc).ravel()))
    if dU * dD <= 6000:
        return float(np.linalg.eigvalsh(H.toarray())[0])
    return float(spla.eigsh(H.tocsr(), k=1, which="SA",
                            return_eigenvectors=False)[0])


def selftest():
    ok = True
    cases = [
        (2, 2, 4.0, "pbc", 1, 1), (2, 2, 4.0, "obc", 1, 1),
        (2, 3, 4.0, "obc", 3, 3), (2, 4, 4.0, "pbc", 4, 4),
        (2, 4, 8.0, "obc", 4, 4), (3, 4, 4.0, "pbc", 6, 6),
    ]
    for (Lx, Ly, U, bc, nu, nd) in cases:
        E_l = hubbard_E0(Lx, Ly, 1.0, U, bc, nu, nd, tol=1e-10, verbose=False)
        E_d = dense_sector_E0(Lx, Ly, 1.0, U, bc, nu, nd)
        good = abs(E_l - E_d) < 1e-7
        ok &= good
        print(f"  {Lx}x{Ly} {bc} U={U} ({nu},{nd}):  Lanczos {E_l:.8f}"
              f"  ref {E_d:.8f}  {'OK' if good else 'FAIL'}")
    # U=0 free-fermion check (includes an open-shell degenerate case)
    for (Lx, Ly, bc, nu, nd) in [(2, 3, "obc", 3, 3), (3, 4, "pbc", 6, 6)]:
        E_l = hubbard_E0(Lx, Ly, 1.0, 0.0, bc, nu, nd, tol=1e-10, verbose=False)
        E_f = free_fermion_E0(Lx, Ly, 1.0, bc, nu, nd)
        good = abs(E_l - E_f) < 1e-7
        ok &= good
        print(f"  {Lx}x{Ly} {bc} U=0 ({nu},{nd}):   Lanczos {E_l:.8f}"
              f"  free-fermion {E_f:.8f}  {'OK' if good else 'FAIL'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, default=4)
    ap.add_argument("--Ly", type=int, default=4)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--U", type=float, default=4.0)
    ap.add_argument("--bc", choices=["obc", "pbc"], default="pbc")
    ap.add_argument("--Nup", type=int, default=None)
    ap.add_argument("--Ndn", type=int, default=None)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    N = args.Lx * args.Ly
    Nup = args.Nup if args.Nup is not None else N // 2
    Ndn = args.Ndn if args.Ndn is not None else N // 2

    E0 = hubbard_E0(args.Lx, args.Ly, args.t, args.U, args.bc, Nup, Ndn,
                    maxiter=args.maxiter, tol=args.tol)
    print(f"# E0 = {E0:.8f}    E0/site = {E0/N:.8f}")
    if args.U == 0.0:
        Ef = free_fermion_E0(args.Lx, args.Ly, args.t, args.bc, Nup, Ndn)
        print(f"# free-fermion check: {Ef:.8f}   |dE| = {abs(E0-Ef):.2e}")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write(f"# Hubbard2D ED  Lx={args.Lx} Ly={args.Ly} t={args.t} "
                     f"U={args.U} bc={args.bc} Nup={Nup} Ndn={Ndn} "
                     f"E0={E0:.8f} E0_per_site={E0/N:.8f}\n")
            fh.write("Lx,Ly,U,bc,E0,E0_per_site\n")
            fh.write(f"{args.Lx},{args.Ly},{args.U},{args.bc},"
                     f"{E0:.8f},{E0/N:.8f}\n")
        print(f"# wrote {args.csv}")


if __name__ == "__main__":
    main()
