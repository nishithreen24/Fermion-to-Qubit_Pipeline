#!/usr/bin/env python3
"""
tv_levy.py  --  Levy-style adiabatic (quantum-annealing) simulation of the
spinless 1D t-V chain, run classically to MEASURE the cost that sets the
required annealing time.

Method (Levy et al., arXiv:2207.14374, "tailored quantum annealers"):
  * encode the fermionic t-V Hamiltonian to qubits (Jordan-Wigner; for a 1D
    nearest-neighbour chain the JW strings stay short, matching the low-weight
    spirit of the tailored-annealer proposal),
  * adiabatically interpolate  H(s) = (1-s) H_drive + s H_problem,  s: 0->1,
    with the standard transverse-field driver  H_drive = -sum_i X_i,
  * the ground state is prepared if the sweep is slow compared to 1/gap_min^2,
    where gap_min = min_s [E1(s) - E0(s)] is the minimum instantaneous spectral
    gap along the path.

WHAT THIS SCRIPT REPORTS (the honest cost quantity):
  gap_min  and  T_required ~ 1/gap_min^2 .
Whether Levy's route "scales" past exact diagonalisation is decided ENTIRELY by
whether gap_min shrinks polynomially (gap ~ 1/L, T ~ L^2, feasible) or
exponentially (gap ~ 2^{-L}, T ~ 4^L, hopeless) with system size L.  This
script measures gap_min(L); the scaling driver (tv_scaling.slurm) sweeps L so
the L-dependence -- the actual answer -- can be fit.

Caveat stated plainly: the transverse-field driver does NOT conserve particle
number, whereas H_problem does.  We therefore also report the gap RESTRICTED to
the final ground state's particle-number sector (the physically relevant gap for
a number-conserving tailored driver, which is what Levy actually advocates).
Both are printed; the sector-restricted gap is the one to trust for the
tailored-annealer scaling claim.

Usage:
    python3 tv_levy.py --L 3 --t 1.0 --V 2.0 --npts 201 --csv tv_levy_L3.csv
    python3 tv_levy.py --L 3 --evolve      # also real-time evolve to check fidelity(T)
"""
import argparse
import numpy as np
from openfermion import FermionOperator, jordan_wigner, get_sparse_operator
from openfermion.utils import count_qubits
import scipy.sparse as sp
import scipy.sparse.linalg as sla


# ----------------------------------------------------------------------------
# Model: spinless 1D t-V chain (particle-hole symmetric convention), OBC.
#   H = -t sum_<ij>(c+_i c_j + h.c.) + V sum_<ij>(n_i-1/2)(n_j-1/2)
# ----------------------------------------------------------------------------
def tv1d(L, t=1.0, V=1.0, periodic=False):
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


def number_operator_matrix(L):
    """Total particle number N = sum_i n_i as a sparse diagonal operator."""
    N = FermionOperator()
    for i in range(L):
        N += FermionOperator(((i, 1), (i, 0)), 1.0)
    return get_sparse_operator(jordan_wigner(N), n_qubits=L)


def driver_hamiltonian(nq):
    """H_drive = -sum_i X_i (transverse field). Sparse."""
    X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    I2 = sp.identity(2, format="csr", dtype=complex)
    Hd = sp.csr_matrix((2 ** nq, 2 ** nq), dtype=complex)
    for i in range(nq):
        ops = [I2] * nq
        ops[i] = X
        term = ops[0]
        for o in ops[1:]:
            term = sp.kron(term, o, format="csr")
        Hd = Hd - term
    return Hd


def hopping_driver(L):
    """Number-conserving hopping driver  H_drive = -sum_<ij>(c+_i c_j + h.c.).

    This is Levy's actual proposal: a driver that stays inside the physical
    particle-number sector, unlike the transverse field.  It is off-diagonal
    (delocalising) so it still provides quantum fluctuations for the sweep.
    """
    Hd = FermionOperator()
    for j in range(L - 1):
        Hd += FermionOperator(((j, 1), (j + 1, 0)), -1.0)
        Hd += FermionOperator(((j + 1, 1), (j, 0)), -1.0)
    return get_sparse_operator(jordan_wigner(Hd), n_qubits=L).astype(complex)


def gap_above_manifold(evs, tol=1e-6):
    """Given sorted eigenvalues, return (E_manifold, gap to first state ABOVE
    the (possibly degenerate) ground manifold, degeneracy)."""
    E0 = evs[0]
    deg = int(np.sum(np.abs(evs - E0) < tol))
    if deg >= len(evs):
        return E0, np.nan, deg
    return E0, evs[deg] - E0, deg


def lowest_k(Hs, k):
    k = min(k, Hs.shape[0] - 1)
    if Hs.shape[0] <= 64:
        w = np.linalg.eigvalsh(Hs.toarray())
        return np.sort(w.real)[:k]
    w = sla.eigsh(Hs, k=k, which="SA", return_eigenvectors=False)
    return np.sort(w.real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--V", type=float, default=2.0)
    ap.add_argument("--npts", type=int, default=201, help="grid of s in [0,1]")
    ap.add_argument("--kstates", type=int, default=6)
    ap.add_argument("--driver", choices=["hop", "field"], default="hop",
                    help="hop = number-conserving hopping driver (Levy's, default);"
                         " field = transverse field (breaks N conservation)")
    ap.add_argument("--evolve", action="store_true",
                    help="also real-time evolve at several T and report fidelity")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    Hf = tv1d(args.L, args.t, args.V)
    # Encoding note: for a 1D nearest-neighbour chain, Jordan-Wigner IS the
    # compact encoding (1 qubit/mode, operator locality <= 2).  Derby-Klassen
    # compact encoding only improves on JW for 2D lattices where JW strings
    # grow as O(L); for 1D it gives no benefit.
    Hp = get_sparse_operator(jordan_wigner(Hf), n_qubits=args.L).astype(complex)
    nq = args.L

    if args.driver == "hop":
        Hd = hopping_driver(args.L)
    else:
        Hd = driver_hamiltonian(args.L)

    # exact reference
    wp, vp = np.linalg.eigh(Hp.toarray())
    E0 = wp[0].real
    psi0 = vp[:, 0]
    Nmat = number_operator_matrix(args.L).toarray().real
    Nexp = float(np.real(psi0.conj() @ Nmat @ psi0))
    Nsec = int(round(Nexp))
    Ndiag = np.round(np.diag(Nmat)).astype(int)
    sec_idx = np.where(Ndiag == Nsec)[0]

    print(f"# t-V Levy adiabatic:  L={args.L}  t={args.t}  V={args.V}"
          f"  driver={args.driver}  nq={nq}")
    print(f"# ED ground energy E0 = {E0:.6f}   <N>={Nexp:.3f}"
          f" -> sector N={Nsec} (dim {len(sec_idx)})")

    svals = np.linspace(0.0, 1.0, args.npts)
    rows = []
    gmin_full = np.inf; s_full = 0.0
    gmin_sec  = np.inf; s_sec  = 0.0

    for s in svals:
        Hs = (1.0 - s) * Hd + s * Hp
        ev = lowest_k(Hs, args.kstates)
        _, gap_full, deg_full = gap_above_manifold(ev)
        # sector-restricted gap (the physically relevant one for Levy)
        Hs_sec = Hs.toarray()[np.ix_(sec_idx, sec_idx)]
        wsec = np.sort(np.linalg.eigvalsh(Hs_sec).real)
        _, gap_sec, _ = gap_above_manifold(wsec)
        rows.append((s, ev[0], gap_full if np.isfinite(gap_full) else 0.0,
                     gap_sec if np.isfinite(gap_sec) else 0.0))
        if np.isfinite(gap_full) and gap_full < gmin_full:
            gmin_full, s_full = gap_full, s
        if np.isfinite(gap_sec) and gap_sec < gmin_sec:
            gmin_sec, s_sec = gap_sec, s

    Treq_full = 1.0 / gmin_full**2 if gmin_full > 0 else np.inf
    Treq_sec  = 1.0 / gmin_sec**2  if gmin_sec  > 0 else np.inf
    print(f"# gap_min (full)     = {gmin_full:.5f} at s={s_full:.3f}"
          f"  -> T_req ~ {Treq_full:.2f}")
    print(f"# gap_min (N-sector) = {gmin_sec:.5f} at s={s_sec:.3f}"
          f"  -> T_req ~ {Treq_sec:.2f}   <-- USE THIS")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("s,E0,gap_full,gap_sector"
                    f"  # L={args.L} t={args.t} V={args.V} driver={args.driver}"
                    f" E0_exact={E0:.6f}"
                    f" gapmin_full={gmin_full:.6f} gapmin_sec={gmin_sec:.6f}\n")
            for (s, e0, gf, gs) in rows:
                f.write(f"{s:.5f},{e0:.6f},{gf:.6f},{gs:.6f}\n")
        print(f"# wrote {args.csv}")

    if args.evolve:
        from scipy.linalg import expm
        w0, v0 = np.linalg.eigh(Hd.toarray())
        target = psi0 / np.linalg.norm(psi0)
        print("# real-time evolution fidelity(T):")
        print(f"#   {'T':>8} {'fidelity':>10}")
        for T in (1.0, 3.0, 10.0, 30.0, 100.0):
            nstep = max(50, int(20 * T))
            st = np.linspace(0, 1, nstep)
            state = v0[:, 0].astype(complex)
            dt = T / nstep
            for s in st:
                Hs = ((1.0 - s) * Hd + s * Hp).toarray()
                state = expm(-1j * Hs * dt) @ state
            fid = abs(np.vdot(target, state)) ** 2
            print(f"#   {T:>8.1f} {fid:>10.5f}")


if __name__ == "__main__":
    main()
