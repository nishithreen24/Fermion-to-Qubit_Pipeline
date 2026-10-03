"""
siam_local.py  (v2 — sector-projected, all bugs fixed)
=======================================================
Self-contained SIAM pipeline for N_bath = 1, 2, 3.
No MPI.  Uses dwave-neal (simulated annealing) and scipy exact diagonalisation.

PHYSICAL SETUP
--------------
The Single-Impurity Anderson Model (SIAM) Hamiltonian is:

    H = U n_{imp,up} n_{imp,dn}
        - (U/2)(n_{imp,up} + n_{imp,dn})
        + U/4
        + sum_{b=1}^{N} eps_b (n_{b,up} + n_{b,dn})
        + sum_{b=1}^{N} V_b sum_sigma
              (a+_{imp,sigma} a_{b,sigma} + h.c.)

Mode indexing:  mode = 2*site + spin   (spin 0=up, 1=dn)
  site 0 -> impurity (modes 0,1)
  site b -> bath b   (modes 2b, 2b+1),  b=1..N

Bath discretisation (star geometry):
  eps_b = -2t cos(b pi/(N+1))
  V_b   =  V  sqrt(2/(N+1)) sqrt(1 - (eps_b/2t)^2)

PHYSICAL SECTOR
---------------
H commutes with total N_tot and total S_z.
Half-filling sector: N_tot = N+1.
  N=1: (N_tot=2, Sz=0)   -- unique ground state
  N=2: (N_tot=3, Sz=+1/2) -- degenerate with Sz=-1/2 by SU(2); we fix +1/2
  N=3: (N_tot=4, Sz=0)   -- unique ground state

WHY THIS MATTERS:
Without sector projection, eigsh returns an arbitrary mixture of the
degenerate Sz=+1/2 and Sz=-1/2 states for N=2. This mixture changes at
each U/V value, making fidelity, Wasserstein, and spin_corr meaningless.
Projecting onto a fixed sector gives a unique ground state at each U/V.

PIPELINE
--------
1. Build H_fermion (FermionOperator).
2. Project onto (N_tot, Sz) sector; diagonalise exactly.
3. BK transform -> H_qubit.
4. XBK (r=2) -> H'(s) in Ising variables s in {+1,-1}.
5. Build counting operator C(s).
6. Quadratise to QUBO via Rosenberg substitution.
7. Dinkelbach: lambda* = min_s H'(s)/C(s), solved by SA.
   CRITICAL: H'(s*)/C(s*) evaluated on pre-quadratisation polynomials
   using only physical spins 0..n_phys-1.

OBSERVABLES
-----------
E0_exact   : exact ground-state energy in sector
E0_sa      : SA Dinkelbach approximation
error      : |E0_sa - E0_exact|
docc_exact : <n_up n_dn> from exact psi
docc_fd    : dE0/dU by finite difference (Hellmann-Feynman check)
d2E0       : d^2E0/dU^2 (Kondo crossover scale)
spin_corr  : <S_z^imp * S_z^bath1> from exact psi
fidelity   : |<psi(U_k)|psi(U_{k+1})>|^2  (consecutive U/V)
d_WGS      : 1-Wasserstein (Hamming metric) between |psi|^2 at consecutive U/V
"""

import os
import csv
import itertools
from collections import defaultdict
from math import prod

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openfermion as of
from openfermion import FermionOperator, bravyi_kitaev
from dwave.samplers import SimulatedAnnealingSampler
import dimod

# ===========================================================================
# OUTPUT
# ===========================================================================
OUT = "output/siam_local"
os.makedirs(OUT, exist_ok=True)

# ===========================================================================
# PARAMETERS
# ===========================================================================
T_HOP   = 1.0
V_HYB   = 1.0
EPS_IMP = 0.0

UV_VALS = np.arange(0.0, 8.0 + 1e-9, 0.4)
N_BATH_LIST = [1, 2, 3]

# Physical (N_tot, 2*Sz) sector for each N_bath
SECTOR = {1: (2, 0), 2: (3, 1), 3: (4, 0)}

SA_SWEEPS   = 20_000
SA_READS    = 200
DK_TOL      = 1e-6
DK_MAXITER  = 60
PENALTY_MULT = 12.0

# Finite-difference step for derivatives:
# dU = max(FD_REL * U, FD_ABS) avoids zero step at U=0
FD_REL = 0.05
FD_ABS = 0.05

# ===========================================================================
# 1. SIAM HAMILTONIAN
# ===========================================================================

def _mode(site, spin): return 2 * site + spin

def build_siam(N, U, t=T_HOP, V=V_HYB, eps_imp=EPS_IMP):
    b   = np.arange(1, N + 1)
    eps = -2.0 * t * np.cos(b * np.pi / (N + 1))
    Vb  = V * np.sqrt(2.0/(N+1)) * np.sqrt(1.0 - (eps/(2.0*t))**2)
    H   = FermionOperator()
    ui, di = _mode(0,0), _mode(0,1)
    H += FermionOperator(((ui,1),(di,1),(di,0),(ui,0)), U)
    c1 = -U/2.0 + eps_imp
    for s in range(2):
        H += FermionOperator(((_mode(0,s),1),(_mode(0,s),0)), c1)
    H += FermionOperator((), U/4.0)
    for b_site, e in enumerate(eps, 1):
        for s in range(2):
            H += FermionOperator(((_mode(b_site,s),1),(_mode(b_site,s),0)), e)
    for b_site, v in enumerate(Vb, 1):
        for s in range(2):
            im = _mode(0, s); bm = _mode(b_site, s)
            H += FermionOperator(((im,1),(bm,0)), v)
            H += FermionOperator(((bm,1),(im,0)), v)
    return H

# ===========================================================================
# 2. SECTOR BASIS
# ===========================================================================

def sector_basis(n_modes, n_elec, sz_twice):
    """
    Return array of Fock-space state integers that have exactly n_elec electrons
    and (n_up - n_dn) = sz_twice.
    Bit k of integer = occupation of mode k.
    Even modes = spin-up, odd modes = spin-dn.
    """
    mask_up = sum(1 << k for k in range(0, n_modes, 2))
    mask_dn = sum(1 << k for k in range(1, n_modes, 2))
    idx = []
    for state in range(2**n_modes):
        nu = bin(state & mask_up).count('1')
        nd = bin(state & mask_dn).count('1')
        if nu + nd == n_elec and nu - nd == sz_twice:
            idx.append(state)
    return np.array(idx, dtype=int)

# ===========================================================================
# 3. EXACT GROUND STATE (sector-projected)
# ===========================================================================

def exact_gs(H_fermion, N):
    """
    Returns (E0, psi0, basis) where psi0 is in the sector subspace.
    """
    n_modes = 2*(N+1)
    n_elec, sz2 = SECTOR[N]
    Hsp   = of.linalg.get_sparse_operator(H_fermion, n_qubits=n_modes)
    basis = sector_basis(n_modes, n_elec, sz2)
    Hsec  = Hsp[np.ix_(basis, basis)].toarray()
    vals, vecs = scipy.linalg.eigh(Hsec)
    psi = vecs[:, 0]; psi /= np.linalg.norm(psi)
    return float(vals[0]), psi, basis

# ===========================================================================
# 4. XBK DIAGONALISATION
# ===========================================================================

def xbk(H_qubit, n, r=2, p=0):
    H_ising = of.QubitOperator()
    for terms, coeff in H_qubit.terms.items():
        for j in range(r):
            Sj = -1 if j < p else 1
            for k in range(r):
                Sk = -1 if k < p else 1
                qc = -1
                Qbar = of.QubitOperator(())
                for i in range(n-1, -1, -1):
                    Zj = of.QubitOperator(f'Z{j*n+i}')
                    Zk = of.QubitOperator(f'Z{k*n+i}')
                    if len(terms) == 0 or i < terms[0][0] or i > terms[qc][0]:
                        op = (Zj*Zk + of.QubitOperator(())) * 0.5
                    elif i == terms[qc][0]:
                        g = terms[qc][1]
                        if g == 'X': op = (Zj*Zk - of.QubitOperator(())) * (-0.5)
                        elif g == 'Y': op = (Zk - Zj) * 0.5j
                        else:          op = (Zj + Zk) * 0.5
                        qc -= 1
                    Qbar *= op
                H_ising += coeff * Sj * Sk * Qbar
    return H_ising

# ===========================================================================
# 5. COUNTING OPERATOR
# ===========================================================================

def counting_op(n, r=2, p=0):
    C = of.QubitOperator()
    for signs in itertools.product([1,-1], repeat=n):
        bs = of.QubitOperator()
        for j in range(r):
            Sj = -1 if j < p else 1
            po = of.QubitOperator(())
            for i in range(n):
                Z = of.QubitOperator(f'Z{j*n+i}')
                po *= (Z * signs[i] + of.QubitOperator(())) * 0.5
            bs += po * Sj
        C += bs * bs
    return C

# ===========================================================================
# 6. POLYNOMIAL UTILITIES
# ===========================================================================

def to_dict(qop):
    d = defaultdict(float)
    for terms, coeff in qop.terms.items():
        d[tuple(idx for idx, _ in terms)] += coeff.real
    return dict(d)

def eval_poly(poly, spin, n_phys):
    """Evaluate polynomial using only physical spins 0..n_phys-1."""
    val = 0.0
    for key, coeff in poly.items():
        if any(i >= n_phys for i in key): continue
        val += coeff * (prod(spin.get(i, 0) for i in key) if key else 1)
    return val

def quadratise(poly):
    M = PENALTY_MULT * max(abs(v) for v in poly.values()) if poly else 1.0
    return dimod.higherorder.utils.make_quadratic(poly, M, dimod.SPIN)

# ===========================================================================
# 7. DINKELBACH
# ===========================================================================

def dinkelbach(H_poly, C_poly, n_phys, lam0=0.0):
    sampler = SimulatedAnnealingSampler()
    lam = lam0
    for k in range(DK_MAXITER):
        sh = defaultdict(float)
        for key, v in H_poly.items(): sh[key] += v
        for key, v in C_poly.items(): sh[key] -= lam * v
        bqm = quadratise(dict(sh))
        try:
            ss = sampler.sample(bqm, num_sweeps=SA_SWEEPS, num_reads=SA_READS)
        except Exception:
            continue
        if ss.first.energy >= -DK_TOL:
            break
        spins = dict(ss.first.sample)
        Hv = eval_poly(H_poly, spins, n_phys)
        Cv = eval_poly(C_poly, spins, n_phys)
        if abs(Cv) < 1e-12: break
        lam_new = Hv / Cv
        if abs(lam_new - lam) < DK_TOL:
            lam = lam_new; break
        lam = lam_new
    return lam, k+1

# ===========================================================================
# 8. OBSERVABLES (from exact sector-projected psi)
# ===========================================================================

def double_occupancy(psi, basis, N):
    """<n_{0,up} n_{0,dn}>  from the sector probability amplitudes."""
    docc = 0.0
    for i, state in enumerate(basis):
        n_up = (state >> 0) & 1   # mode 0
        n_dn = (state >> 1) & 1   # mode 1
        docc += abs(psi[i])**2 * n_up * n_dn
    return float(docc)

def spin_corr(psi, basis, N):
    """<S_z^imp * S_z^bath1> = (1/4)<(n_{0u}-n_{0d})(n_{1u}-n_{1d})>."""
    if N < 1: return 0.0
    c = 0.0
    for i, state in enumerate(basis):
        szi = ((state>>0)&1) - ((state>>1)&1)   # 2*S_z^imp
        szb = ((state>>2)&1) - ((state>>3)&1)   # 2*S_z^bath1
        c  += abs(psi[i])**2 * szi * szb
    return float(c) * 0.25

def fidelity(psi1, psi2):
    return float(abs(np.dot(psi1.conj(), psi2))**2)

def wasserstein_hamming(p, q, basis):
    """
    Exact 1-Wasserstein with Hamming ground metric, solved by LP.
    p, q are probability vectors over the sector basis.
    """
    from scipy.optimize import linprog
    n = len(p)
    C = np.array([[bin(int(basis[i])^int(basis[j])).count('1')
                   for j in range(n)] for i in range(n)], dtype=float)
    Aeq = np.zeros((2*n, n*n))
    beq = np.zeros(2*n)
    for i in range(n):
        Aeq[i, i*n:(i+1)*n] = 1.0; beq[i] = p[i]
    for j in range(n):
        Aeq[n+j, j::n] = 1.0; beq[n+j] = q[j]
    res = linprog(C.flatten(), A_eq=Aeq, b_eq=beq,
                  bounds=[(0,None)]*n*n, method='highs')
    return float(res.fun) if res.success else float('nan')

# ===========================================================================
# 9. SWEEP
# ===========================================================================

def run_sweep(N):
    print(f"\n{'='*60}\n  SIAM N_bath={N}  sector={SECTOR[N]}\n{'='*60}")
    r = 2
    res = {k: [] for k in ['uv','E0_sa','E0_exact','error',
                             'docc_fd','docc_exact','d2E0',
                             'spin_corr','fidelity','d_WGS']}
    prev_psi = prev_probs = prev_basis = None

    for uv in UV_VALS:
        U = uv * V_HYB
        print(f"\n  U/V={uv:.2f}")

        H = build_siam(N, U)
        E0, psi0, basis = exact_gs(H, N)
        probs = np.abs(psi0)**2

        # finite difference with protected step
        dU = max(FD_REL * U, FD_ABS) * V_HYB
        Ep, _, _ = exact_gs(build_siam(N, U+dU), N)
        Em, _, _ = exact_gs(build_siam(N, U-dU), N)
        docc_fd = (Ep - Em) / (2*dU)
        d2      = (Ep - 2*E0 + Em) / dU**2

        docc_ex = double_occupancy(psi0, basis, N)
        sc      = spin_corr(psi0, basis, N)

        if prev_psi is not None and len(prev_basis) == len(basis):
            F    = fidelity(prev_psi, psi0)
            dWGS = wasserstein_hamming(prev_probs, probs, basis)
        else:
            F = 1.0; dWGS = 0.0

        print(f"    E0={E0:.6f}  docc={docc_ex:.4f}  d2={d2:.4f}  "
              f"scorr={sc:.4f}  F={F:.4f}  W={dWGS:.4f}")

        # SA pipeline
        Hq    = bravyi_kitaev(H)
        nq    = of.count_qubits(Hq)
        nphys = r * nq
        best  = np.inf
        for p_sec in range(r+1):
            Hi   = xbk(Hq, nq, r=r, p=p_sec)
            Ci   = counting_op(nq, r=r, p=p_sec)
            lam, nitr = dinkelbach(to_dict(Hi), to_dict(Ci), nphys)
            print(f"    p={p_sec} E0_SA={lam:.6f} ({nitr} iters)")
            best = min(best, lam)

        print(f"    -> best E0_SA={best:.6f}  err={abs(best-E0):.2e}")

        res['uv'].append(float(uv))
        res['E0_sa'].append(float(best))
        res['E0_exact'].append(float(E0))
        res['error'].append(float(abs(best-E0)))
        res['docc_fd'].append(float(docc_fd))
        res['docc_exact'].append(float(docc_ex))
        res['d2E0'].append(float(d2))
        res['spin_corr'].append(float(sc))
        res['fidelity'].append(float(F))
        res['d_WGS'].append(float(dWGS))

        prev_psi = psi0; prev_probs = probs; prev_basis = basis

    return res

# ===========================================================================
# 10. PLOTTING
# ===========================================================================

def plot_all(all_res):
    colors  = {1:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c'}
    markers = {1:'o', 2:'s', 3:'^'}
    panels = [
        ('E0_exact',   r'$E_0$ (exact)',
                       'Ground-state energy (physical sector)'),
        ('E0_sa',      r'$E_0$ (SA)',
                       'SA ground-state energy'),
        ('error',      r'$|E_0^{SA}-E_0^{exact}|$',
                       'SA absolute error'),
        ('docc_exact', r'$\langle n_\uparrow n_\downarrow\rangle$',
                       'Double occupancy'),
        ('d2E0',       r'$d^2E_0/dU^2$',
                       'Second derivative of $E_0$ (Kondo crossover scale)'),
        ('spin_corr',  r'$\langle S_z^{imp}S_z^{bath}\rangle$',
                       'Impurity-bath spin correlation'),
        ('fidelity',   r'$|\langle\psi(U_k)|\psi(U_{k+1})\rangle|^2$',
                       'Quantum fidelity (consecutive $U/V$)'),
        ('d_WGS',      r'$W_1$ (Wasserstein, Hamming)',
                       'Wasserstein distance (consecutive $U/V$)'),
    ]
    for key, ylabel, title in panels:
        fig, ax = plt.subplots(figsize=(7,4))
        for N, r in all_res.items():
            ax.plot(r['uv'], r[key], color=colors[N], marker=markers[N],
                    markersize=4, label=f'$N_{{bath}}={N}$', linewidth=1.5)
        ax.set_xlabel(r'$U/V$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'{key}.pdf'), dpi=150)
        plt.close(fig)
    print(f"  Figures saved to {OUT}/")

# ===========================================================================
# 11. CSV
# ===========================================================================

def save_csv(N, res):
    cols = ['uv','E0_sa','E0_exact','error','docc_fd','docc_exact',
            'd2E0','spin_corr','fidelity','d_WGS']
    with open(os.path.join(OUT, f'siam_N{N}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(len(res['uv'])):
            w.writerow([res[c][i] for c in cols])

# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    all_res = {}
    for N in N_BATH_LIST:
        r = run_sweep(N)
        all_res[N] = r
        save_csv(N, r)
    plot_all(all_res)
    print(f"\nDone. Outputs in {OUT}/")
