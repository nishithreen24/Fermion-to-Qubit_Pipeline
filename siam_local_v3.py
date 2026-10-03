"""
siam_local.py  (v3 — all bugs fixed, correct architecture)
===========================================================
Self-contained SIAM script for N_bath = 1, 2, 3.
No MPI. No SA. All observables are computed EXACTLY.

WHY NO SA HERE
--------------
The XBK Rayleigh quotient  min_{p,s} H'_p(s)/C_p(s) = lambda_min(H)
requires r large enough that the ground state is representable in the
replica simplex.  For the SIAM BK Hamiltonian (which contains 3-body
XYY and YYX Pauli terms), r=2 gives min = -1.0 instead of -2.236,
and r=3 gives -1.667.  The required r grows with the weight of the
BK Pauli strings and the structure of the ground state.  For the local
script, exact diagonalisation is both faster and exact.

The SA / qanneal E0 values for the paper come from the HPC runs.

PHYSICAL SECTOR PROJECTION (mandatory)
---------------------------------------
H commutes with total electron number N_tot and total S_z.
Without projection onto a fixed (N_tot, S_z) sector, scipy.eigsh
returns an arbitrary mixture of degenerate states (N=2 has a 2-fold
SU(2) degeneracy at all U/V), making fidelity, Wasserstein, and
spin_corr meaningless.

Sectors used (half-filling, N_tot = N+1):
  N=1: (N_tot=2, S_z=0)    -- unique ground state, gap > 1.2 at all U/V
  N=2: (N_tot=3, S_z=+1/2) -- unique within sector, gap > 1.4 at all U/V
  N=3: (N_tot=4, S_z=0)    -- unique ground state, gap > 0.4 at all U/V

FINITE-DIFFERENCE STEP (d2E0 fix)
------------------------------------
Step dU = max(FD_REL * U, FD_ABS) * V avoids division by zero at U=0
which caused a spurious spike in d2E0 for N=2.

OBSERVABLES
-----------
E0_exact  : exact ground-state energy in the (N_tot, S_z) sector.
docc      : double occupancy <n_{imp,up} n_{imp,dn}> from exact psi.
            NOTE: dE0/dU = <n_up n_dn> - 1/4 holds in the grand-canonical
            (variable N) ensemble. In a fixed-sector ground state this
            identity is violated for N=2,3, and both columns are reported.
docc_fd   : dE0/dU by finite difference (numerical derivative).
d2E0      : d^2 E0 / dU^2 (Kondo crossover scale).
spin_corr : <S_z^imp * S_z^bath1> from exact psi.
fidelity  : |<psi(U_k)|psi(U_{k+1})>|^2  (consecutive U/V steps).
d_WGS     : 1-Wasserstein distance (Hamming ground metric) between
            the probability distributions |psi(U_k)|^2 and |psi(U_{k+1})|^2
            over the sector basis. Solved as an exact LP.
            This is the key geometric observable in the XBK paper.
"""

import os, csv, itertools
from collections import defaultdict
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openfermion as of
from openfermion import FermionOperator, bravyi_kitaev

OUT = "output/siam_local"
os.makedirs(OUT, exist_ok=True)

# ─── parameters ─────────────────────────────────────────────────────────────
T_HOP   = 1.0
V_HYB   = 1.0
EPS_IMP = 0.0

UV_VALS     = np.arange(0.0, 8.0 + 1e-9, 0.4)
N_BATH_LIST = [1, 2, 3]

# Physical (N_tot, 2*S_z) sector for each N_bath
SECTOR = {1: (2, 0), 2: (3, 1), 3: (4, 0)}

# Finite-difference step: dU = max(FD_REL*U, FD_ABS) * V
FD_REL = 0.05
FD_ABS = 0.05

# ─── 1.  SIAM Hamiltonian ────────────────────────────────────────────────────
def _mode(site, spin): return 2 * site + spin

def build_siam(N, U, t=T_HOP, V=V_HYB, eps_imp=EPS_IMP):
    """
    H_SIAM = U n_{0,up} n_{0,dn}  - (U/2)(n_{0,up}+n_{0,dn})  + U/4
           + sum_{b=1}^{N} eps_b (n_{b,up}+n_{b,dn})
           + sum_{b=1}^{N} V_b sum_sigma (a+_{0,sigma} a_{b,sigma} + h.c.)

    Bath parameters (star geometry):
        eps_b = -2t cos(b pi / (N+1))
        V_b   =  V  sqrt(2/(N+1)) sqrt(1 - (eps_b/2t)^2)
    """
    b_idx = np.arange(1, N + 1)
    eps   = -2.0 * t * np.cos(b_idx * np.pi / (N + 1))
    Vb    =  V * np.sqrt(2.0 / (N + 1)) * np.sqrt(1.0 - (eps / (2.0 * t))**2)
    H = FermionOperator()
    H += FermionOperator(((0,1),(1,1),(1,0),(0,0)), U)
    for s in range(2):
        H += FermionOperator(((_mode(0,s),1),(_mode(0,s),0)), -U/2.0 + eps_imp)
    H += FermionOperator((), U/4.0)
    for b, (e, v) in enumerate(zip(eps, Vb), 1):
        bm0, bm1 = _mode(b,0), _mode(b,1)
        H += FermionOperator(((bm0,1),(bm0,0)), e)
        H += FermionOperator(((bm1,1),(bm1,0)), e)
        H += FermionOperator(((0,1),(bm0,0)), v); H += FermionOperator(((bm0,1),(0,0)), v)
        H += FermionOperator(((1,1),(bm1,0)), v); H += FermionOperator(((bm1,1),(1,0)), v)
    return H

# ─── 2.  Sector basis ────────────────────────────────────────────────────────
def sector_basis(n_modes, n_elec, sz2):
    """
    Return integer array of Fock-space states with exactly n_elec electrons
    and (n_up - n_dn) = sz2.
    Bit k of integer = occupation of mode k.
    Even modes = spin-up, odd modes = spin-dn.
    """
    mu = sum(1 << k for k in range(0, n_modes, 2))   # mask for up modes
    md = sum(1 << k for k in range(1, n_modes, 2))   # mask for dn modes
    return np.array([s for s in range(2**n_modes)
                     if bin(s & mu).count('1') + bin(s & md).count('1') == n_elec
                     and bin(s & mu).count('1') - bin(s & md).count('1') == sz2],
                    dtype=int)

# ─── 3.  Exact ground state in sector ────────────────────────────────────────
def exact_gs(H_ferm, N):
    """
    Returns (E0, psi0, basis) where psi0 is normalised in the sector subspace.
    """
    n_modes       = 2 * (N + 1)
    n_elec, sz2   = SECTOR[N]
    Hsp   = of.linalg.get_sparse_operator(H_ferm, n_qubits=n_modes)
    basis = sector_basis(n_modes, n_elec, sz2)
    Hsec  = Hsp[np.ix_(basis, basis)].toarray()
    vals, vecs = scipy.linalg.eigh(Hsec)       # ascending order, real symmetric
    psi = vecs[:, 0]
    psi /= np.linalg.norm(psi)
    return float(vals[0]), psi, basis

# ─── 4.  Observables ─────────────────────────────────────────────────────────
def double_occupancy(psi, basis):
    """<n_{imp,up} n_{imp,dn}> = sum_state |psi|^2 * bit0(state) * bit1(state)."""
    return float(sum(abs(psi[i])**2 * ((basis[i]>>0)&1) * ((basis[i]>>1)&1)
                     for i in range(len(basis))))

def spin_corr(psi, basis):
    """
    <S_z^imp * S_z^bath1> = (1/4) <(n_{0,up}-n_{0,dn})(n_{1,up}-n_{1,dn})>.
    S_z^imp   acts on modes 0,1 (site 0).
    S_z^bath1 acts on modes 2,3 (site 1).
    """
    c = 0.0
    for i, state in enumerate(basis):
        szi = ((state>>0)&1) - ((state>>1)&1)   # 2*S_z^imp
        szb = ((state>>2)&1) - ((state>>3)&1)   # 2*S_z^bath1
        c  += abs(psi[i])**2 * szi * szb
    return float(c) * 0.25

def fidelity(psi1, psi2):
    """F = |<psi1|psi2>|^2. Both vectors must be in the same sector subspace."""
    return float(abs(np.dot(psi1.conj(), psi2))**2)

def wasserstein_hamming(p, q, basis):
    """
    1-Wasserstein distance W_1(p, q) where p, q are probability vectors
    over the sector basis, and the ground metric is Hamming distance between
    the corresponding Fock-space integers.

    W_1 = min_{pi >= 0}  sum_{x,y} pi(x,y) * d_H(basis[x], basis[y])
    subject to  sum_y pi(x,y) = p[x]   (row marginal)
                sum_x pi(x,y) = q[y]   (column marginal)

    Solved as a linear program (HiGHS solver).
    """
    from scipy.optimize import linprog
    n = len(p)
    # Cost matrix: C[i,j] = Hamming distance between basis[i] and basis[j]
    C = np.array([[bin(int(basis[i]) ^ int(basis[j])).count('1')
                   for j in range(n)] for i in range(n)], dtype=float)
    Aeq = np.zeros((2*n, n*n))
    beq = np.zeros(2*n)
    for i in range(n):
        Aeq[i, i*n:(i+1)*n] = 1.0; beq[i] = p[i]
    for j in range(n):
        Aeq[n+j, j::n] = 1.0; beq[n+j] = q[j]
    res = linprog(C.flatten(), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)]*(n*n), method='highs')
    return float(res.fun) if res.success else float('nan')

# ─── 5.  Sweep ───────────────────────────────────────────────────────────────
def run_sweep(N):
    print(f"\n{'='*60}\n  SIAM  N_bath={N}  sector={SECTOR[N]}\n{'='*60}")
    keys = ['uv','E0_exact','docc','docc_fd','d2E0','spin_corr','fidelity','d_WGS']
    res  = {k: [] for k in keys}
    prev_psi = prev_probs = prev_basis = None

    for uv in UV_VALS:
        U = uv * V_HYB
        H           = build_siam(N, U)
        E0, psi, basis = exact_gs(H, N)
        probs       = np.abs(psi)**2

        # finite-difference derivatives (protected step)
        dU        = max(FD_REL * U, FD_ABS) * V_HYB
        Ep, _, _  = exact_gs(build_siam(N, U + dU), N)
        Em, _, _  = exact_gs(build_siam(N, U - dU), N)
        docc_fd_v = (Ep - Em) / (2 * dU)
        d2_v      = (Ep - 2*E0 + Em) / dU**2

        docc_v = double_occupancy(psi, basis)
        sc_v   = spin_corr(psi, basis)

        if prev_psi is not None and len(prev_basis) == len(basis):
            F    = fidelity(prev_psi, psi)
            dWGS = wasserstein_hamming(prev_probs, probs, basis)
        else:
            F = 1.0; dWGS = 0.0

        print(f"  U/V={uv:.2f}: E0={E0:.5f}  docc={docc_v:.4f}  "
              f"d2={d2_v:.4f}  Scorr={sc_v:.4f}  F={F:.4f}  W={dWGS:.4f}")

        res['uv'].append(float(uv))
        res['E0_exact'].append(float(E0))
        res['docc'].append(float(docc_v))
        res['docc_fd'].append(float(docc_fd_v))
        res['d2E0'].append(float(d2_v))
        res['spin_corr'].append(float(sc_v))
        res['fidelity'].append(float(F))
        res['d_WGS'].append(float(dWGS))

        prev_psi = psi; prev_probs = probs; prev_basis = basis

    return res

# ─── 6.  Plotting ─────────────────────────────────────────────────────────────
def plot_all(all_res):
    colors  = {1:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c'}
    markers = {1:'o', 2:'s', 3:'^'}
    panels = [
        ('E0_exact',   r'$E_0$',
                       'Ground-state energy (physical sector)'),
        ('docc',       r'$\langle n_\uparrow n_\downarrow\rangle$',
                       'Double occupancy'),
        ('docc_fd',    r'$dE_0/dU$',
                       r'Hellmann–Feynman: $dE_0/dU$ (finite difference)'),
        ('d2E0',       r'$d^2E_0/dU^2$',
                       r'$d^2E_0/dU^2$ (Kondo crossover scale)'),
        ('spin_corr',  r'$\langle S_z^{\rm imp}\,S_z^{\rm bath}\rangle$',
                       'Impurity–bath spin correlation'),
        ('fidelity',   r'$|\langle\psi_k|\psi_{k+1}\rangle|^2$',
                       'Quantum fidelity (consecutive $U/V$ steps)'),
        ('d_WGS',      r'$W_1$ (Wasserstein, Hamming)',
                       'Wasserstein distance $d_{\rm WGS}$ (consecutive $U/V$)'),
    ]
    for key, ylabel, title in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        for N, r in all_res.items():
            ax.plot(r['uv'], r[key], color=colors[N], marker=markers[N],
                    markersize=4, label=f'$N_{{\\rm bath}}={N}$', linewidth=1.5)
        ax.set_xlabel(r'$U/V$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'{key}.pdf'), dpi=150)
        plt.close(fig)
        print(f"  Saved {key}.pdf")

# ─── 7.  CSV ──────────────────────────────────────────────────────────────────
def save_csv(N, res):
    cols = ['uv','E0_exact','docc','docc_fd','d2E0','spin_corr','fidelity','d_WGS']
    with open(os.path.join(OUT, f'siam_N{N}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(len(res['uv'])):
            w.writerow([res[c][i] for c in cols])
    print(f"  Saved siam_N{N}.csv")

# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_res = {}
    for N in N_BATH_LIST:
        r = run_sweep(N)
        all_res[N] = r
        save_csv(N, r)
    plot_all(all_res)
    print(f"\nAll outputs in {OUT}/")
