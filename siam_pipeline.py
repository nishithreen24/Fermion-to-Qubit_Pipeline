"""
siam_pipeline.py
================
Reproduces all figures from "Mapping of Fermionic Lattice Models for Ising
Solvers" (Nagpal, Kumar, Hassan 2025) but for the Single Impurity Anderson
Model (SIAM) defined in Models.py.

Add this file to the same directory as playground.py and Models.py.
Run:   python siam_pipeline.py

Figures produced:
  Fig A  – Ground state energy E0(U/t) from XBK-QUBO vs ED
  Fig B  – Numerical curvature E0''(U/t): central differences, raw and smoothed
  Fig C  – Replication factor study: |E_sim - E_ED| and wall-clock time vs r
  Fig D  – Thermal observables: Binder cumulant U4, susceptibility chi, heat
            capacity Cv, all vs effective temperature T, for N_bath = 1,2,3
  Fig E  – Site-resolved <n_{site}> impurity occupancy vs U/t
  Fig F  – Impurity double occupancy <n_up n_down> vs U/t  (Mott diagnostic)

All simulations use classical SA (Neal) exactly as in the paper.
ED is performed via exact diagonalization using openfermion sparse matrices.
"""

import time
import itertools
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from math import prod

import openfermion as op
from openfermion import bravyi_kitaev, get_sparse_operator
from openfermion import FermionOperator
import dimod
from neal import SimulatedAnnealingSampler

# ── local imports ────────────────────────────────────────────────────────────
from Models import build_siam_hamiltonian
# XBK, construct, convert_to_dict, reduction live in playground.py
# We import them directly.
from playground import XBK, construct, convert_to_dict, reduction

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 0. EXACT DIAGONALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def exact_ground_energy(H_fermion):
    """
    Return the ground state energy of a FermionOperator via sparse ED.

    Mathematically: E0 = min eigenvalue of the matrix representation of
    H_fermion in the full Fock space of dimension 2^(number of modes).

    Parameters
    ----------
    H_fermion : FermionOperator

    Returns
    -------
    E0 : float
    """
    H_sparse = get_sparse_operator(H_fermion)
    dim = H_sparse.shape[0]
    if dim <= 64:
        # dense is more reliable for small matrices
        evals = np.linalg.eigvalsh(H_sparse.toarray())
        return float(evals[0])
    else:
        evals, _ = spla.eigsh(H_sparse, k=1, which='SA')
        return float(evals[0])


def exact_spectrum(H_fermion, k=20):
    """
    Return the lowest k eigenvalues of H_fermion via sparse ED.

    Used for thermal observables: partition function Z(T) = sum_i exp(-E_i/T).

    Parameters
    ----------
    H_fermion : FermionOperator
    k         : int  number of eigenvalues to return

    Returns
    -------
    evals : np.ndarray  shape (k,), sorted ascending
    """
    H_sparse = get_sparse_operator(H_fermion)
    dim = H_sparse.shape[0]
    k = min(k, dim - 2)
    if dim <= 128:
        evals = np.linalg.eigvalsh(H_sparse.toarray())
        return evals
    else:
        evals, _ = spla.eigsh(H_sparse, k=k, which='SA')
        return np.sort(evals)


def exact_double_occupancy(H_fermion, N_bath):
    """
    Compute the ground state expectation value of the impurity double
    occupancy D = <n_{0,up} n_{0,down}>.

    D is a diagnostic for the Kondo/Mott crossover in SIAM:
    D -> 0.25 for U=0 (non-interacting), D -> 0 for large U (singly occupied).

    The operator n_{0,up} n_{0,down} in mode indices 0 (up) and 1 (down) is:
        D_op = a†_0 a_0 a†_1 a_1

    Parameters
    ----------
    H_fermion : FermionOperator  (SIAM at some U)
    N_bath    : int

    Returns
    -------
    D : float
    """
    H_sparse = get_sparse_operator(H_fermion)
    dim = H_sparse.shape[0]

    # Build double occupancy operator
    # modes 0 = impurity up, 1 = impurity down
    D_op = FermionOperator('0^ 0 1^ 1', 1.0)
    D_sparse = get_sparse_operator(D_op, n_qubits=int(np.log2(dim)))

    if dim <= 128:
        evals, evecs = np.linalg.eigh(H_sparse.toarray())
        gs = evecs[:, 0]
    else:
        evals, evecs = spla.eigsh(H_sparse, k=1, which='SA')
        gs = evecs[:, 0]

    D = float(np.real(gs.conj() @ D_sparse @ gs))
    return D


def exact_impurity_occupancy(H_fermion, N_bath):
    """
    Compute ground state <n_{0,up} + n_{0,down}> (total impurity occupancy).

    In SIAM, at particle-hole symmetry (eps_imp = -U/2), this equals 1.0
    for all U. Deviations indicate symmetry breaking or implementation errors.

    Parameters
    ----------
    H_fermion : FermionOperator
    N_bath    : int

    Returns
    -------
    n_imp : float
    """
    H_sparse = get_sparse_operator(H_fermion)
    dim = H_sparse.shape[0]
    n_qubits = int(np.log2(dim))

    n_op = FermionOperator('0^ 0', 1.0) + FermionOperator('1^ 1', 1.0)
    n_sparse = get_sparse_operator(n_op, n_qubits=n_qubits)

    if dim <= 128:
        evals, evecs = np.linalg.eigh(H_sparse.toarray())
        gs = evecs[:, 0]
    else:
        evals, evecs = spla.eigsh(H_sparse, k=1, which='SA')
        gs = evecs[:, 0]

    n_imp = float(np.real(gs.conj() @ n_sparse @ gs))
    return n_imp


# ═══════════════════════════════════════════════════════════════════════════
# 1. XBK → QUBO PIPELINE (wraps playground.py functions)
# ═══════════════════════════════════════════════════════════════════════════

def xbk_ground_energy(H_fermion, r, p,
                      num_sweeps=10000, num_reads=200,
                      lam_init=100.0, max_iter=50):
    """
    Run the full BK → XBK → QUBO → Dinkelbach pipeline and return
    the estimated ground state energy.

    This implements Algorithm 1 from the XBK paper (Xia, Bian, Kais 2021)
    and Section III E of Nagpal et al. 2025.

    The Dinkelbach iteration solves:
        min_s [ H'_p(s) - lambda * C_p(s) ] = 0
    where lambda converges to E0 = H'_p(s*) / C_p(s*).

    Parameters
    ----------
    H_fermion  : FermionOperator
    r          : int  XBK replication factor
    p          : int  XBK sector (number of negative-sign replicas)
    num_sweeps : int  SA sweeps per sample
    num_reads  : int  SA reads per solve
    lam_init   : float  initial lambda
    max_iter   : int  maximum Dinkelbach iterations

    Returns
    -------
    lam   : float  estimated ground state energy
    t_wall: float  wall-clock time in seconds
    """
    t0 = time.time()

    Q = bravyi_kitaev(H_fermion)
    n = op.count_qubits(Q)
    Q_diag = XBK(Q, n, r, p)
    C = construct(n, r, p)

    lam = lam_init
    sampler = SimulatedAnnealingSampler()

    for _ in range(max_iter):
        H_shift = Q_diag - lam * C
        H_shifted = convert_to_dict(H_shift)
        bqm = reduction(H_shifted)

        try:
            sampleset = sampler.sample(bqm, num_sweeps=num_sweeps,
                                       num_reads=num_reads)
        except ValueError:
            continue

        gse = sampleset.first.energy
        if gse >= -1e-8:
            break

        gs = dict(sampleset.first.sample)
        C_dict = convert_to_dict(C)
        bm_2 = sum(
            coeff * (prod(gs[q] for q in key) if key else 1)
            for key, coeff in C_dict.items()
        )
        if abs(bm_2) < 1e-10:
            break
        lam = lam + gse / bm_2

    t_wall = time.time() - t0
    return lam, t_wall


def best_xbk_energy(H_fermion, r, num_sweeps=10000, num_reads=200):
    """
    Run XBK over all sectors p in range(r+1) and return the minimum energy.

    E0 = min_p E0_p   (Eq. 6 in Nagpal et al.)

    Parameters
    ----------
    H_fermion : FermionOperator
    r         : int

    Returns
    -------
    E0_best : float
    t_total : float  total wall-clock time
    """
    best = np.inf
    t_total = 0.0
    for p in range(r + 1):
        lam, t = xbk_ground_energy(H_fermion, r, p,
                                   num_sweeps=num_sweeps,
                                   num_reads=num_reads)
        t_total += t
        if lam < best:
            best = lam
    return best, t_total


# ═══════════════════════════════════════════════════════════════════════════
# 2. THERMAL OBSERVABLES (classical exact, using ED spectrum)
# ═══════════════════════════════════════════════════════════════════════════

def thermal_observables(evals, T_values):
    """
    Compute thermal observables from a spectrum {E_i} as a function of T.

    The partition function is Z(T) = sum_i exp(-E_i / T).
    The thermal average of an observable A with eigenvalues {a_i} is:
        <A>(T) = sum_i a_i exp(-E_i / T) / Z(T)

    We compute:
        E_mean(T)  = <H>
        E2_mean(T) = <H^2>
        Cv(T)      = (E2_mean - E_mean^2) / T^2   (heat capacity)
        M_mean(T)  = <M>  where M_i = |sum_j s_j|  NOT applicable here;
                     for SIAM we use total spin Sz = 0 sector,
                     so we use magnetisation proxy = E_mean / T (omit)

    For Binder cumulant and susceptibility we need M^2 and M^4.
    Since SIAM doesn't have a simple order parameter, we use:
        m_i = E_i - E_i.mean()  (energy fluctuation as proxy)
    and define the susceptibility as chi = (<m^2> - <m>^2) / T.

    Binder cumulant: U4 = 1 - <m^4> / (3 * <m^2>^2)
    This is dimensionless and crosses at the transition temperature
    for different system sizes.

    Parameters
    ----------
    evals    : np.ndarray  energy eigenvalues (sorted ascending)
    T_values : np.ndarray  temperatures (positive floats)

    Returns
    -------
    results : dict with keys 'T', 'Cv', 'chi', 'U4', 'E_mean'
    """
    E = evals - evals[0]   # shift so ground state = 0 (numerical stability)
    n_states = len(E)

    Cv_arr  = np.zeros(len(T_values))
    chi_arr = np.zeros(len(T_values))
    U4_arr  = np.zeros(len(T_values))
    Em_arr  = np.zeros(len(T_values))

    for ti, T in enumerate(T_values):
        beta = 1.0 / T
        weights = np.exp(-beta * E)
        Z = weights.sum()
        weights /= Z

        # Thermal averages of energy
        E_mean  = float(np.dot(weights, E))
        E2_mean = float(np.dot(weights, E**2))

        # Heat capacity per site: Cv = (<E^2> - <E>^2) / T^2
        Cv_arr[ti] = (E2_mean - E_mean**2) / T**2

        # Energy fluctuation moments (used as proxy for order parameter)
        m = E - E_mean
        m2 = float(np.dot(weights, m**2))
        m4 = float(np.dot(weights, m**4))

        chi_arr[ti] = m2 / T
        U4_arr[ti]  = 1.0 - m4 / (3.0 * m2**2 + 1e-30)
        Em_arr[ti]  = E_mean + evals[0]   # restore original energy scale

    return {'T': T_values, 'Cv': Cv_arr, 'chi': chi_arr,
            'U4': U4_arr, 'E_mean': Em_arr}


# ═══════════════════════════════════════════════════════════════════════════
# 3. NUMERICAL CURVATURE (second derivative by central differences)
# ═══════════════════════════════════════════════════════════════════════════

def second_derivative(x, y):
    """
    Compute the numerical second derivative d^2y/dx^2 via central differences.

    For interior points i = 1, ..., n-2:
        y''_i = (y_{i+1} - 2*y_i + y_{i-1}) / h^2
    where h = x_{i+1} - x_i  (assumed uniform; averaged if not).

    For endpoints, forward/backward differences are used.

    Parameters
    ----------
    x : np.ndarray  shape (n,)  independent variable (assumed roughly uniform)
    y : np.ndarray  shape (n,)  dependent variable

    Returns
    -------
    d2y : np.ndarray  shape (n,)
    """
    n = len(x)
    d2y = np.zeros(n)

    for i in range(1, n - 1):
        h1 = x[i]   - x[i-1]
        h2 = x[i+1] - x[i]
        d2y[i] = 2.0 * (y[i+1]/h2 - y[i]*(1/h1 + 1/h2) + y[i-1]/h1) / (h1 + h2)

    # Endpoints: one-sided second-order finite difference
    # Forward: y''_0 = (y_2 - 2*y_1 + y_0) / h^2
    d2y[0]    = d2y[1]
    d2y[n-1]  = d2y[n-2]

    return d2y


def smooth(y, window=3):
    """
    Simple moving-average smoother over a window of odd size.
    Used to produce the 'smoothed curvature' curve in the paper.

    Parameters
    ----------
    y      : np.ndarray
    window : int  (must be odd)

    Returns
    -------
    y_smooth : np.ndarray  same shape as y
    """
    pad = window // 2
    y_pad = np.pad(y, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode='valid')


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN SWEEP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def sweep_U(N_bath, U_values, t=1.0, V_hyb=1.0, eps_imp=None,
            r=2, num_sweeps=10000, num_reads=200):
    """
    Sweep over U/t for SIAM with N_bath bath sites.

    At each U value, compute:
        - E0_ED    : exact ground state energy via ED
        - E0_XBK   : XBK-QUBO ground state energy (best over all sectors)
        - D_occ    : impurity double occupancy <n_up n_down> (ED)
        - n_imp    : total impurity occupancy (ED)
        - t_wall   : XBK wall-clock time

    Parameters
    ----------
    N_bath    : int    number of bath sites
    U_values  : array  values of U to sweep (in units of t)
    t         : float  hopping amplitude (sets energy scale)
    V_hyb     : float  hybridization amplitude
    eps_imp   : float  impurity level energy.  None -> -U/2 (p-h symmetry)
    r         : int    XBK replication factor
    num_sweeps: int    SA sweeps
    num_reads : int    SA reads

    Returns
    -------
    results : dict with numpy arrays indexed by U_values
    """
    E0_ED_arr  = np.zeros(len(U_values))
    E0_XBK_arr = np.zeros(len(U_values))
    D_occ_arr  = np.zeros(len(U_values))
    n_imp_arr  = np.zeros(len(U_values))
    t_wall_arr = np.zeros(len(U_values))

    for i, U in enumerate(U_values):
        eps = -U / 2.0 if eps_imp is None else eps_imp
        print(f"  U={U:.3f}  eps_imp={eps:.3f}  N_bath={N_bath}  r={r}")

        H = build_siam_hamiltonian(N=N_bath, U=U, t=t, V=V_hyb, eps_imp=eps)

        E0_ED_arr[i]  = exact_ground_energy(H)
        D_occ_arr[i]  = exact_double_occupancy(H, N_bath)
        n_imp_arr[i]  = exact_impurity_occupancy(H, N_bath)

        E0_XBK_arr[i], t_wall_arr[i] = best_xbk_energy(
            H, r=r, num_sweeps=num_sweeps, num_reads=num_reads
        )

        print(f"    ED={E0_ED_arr[i]:.6f}  XBK={E0_XBK_arr[i]:.6f}  "
              f"err={abs(E0_XBK_arr[i]-E0_ED_arr[i]):.2e}  "
              f"t={t_wall_arr[i]:.1f}s")

    return {
        'U_values'  : np.array(U_values),
        'E0_ED'     : E0_ED_arr,
        'E0_XBK'    : E0_XBK_arr,
        'D_occ'     : D_occ_arr,
        'n_imp'     : n_imp_arr,
        't_wall'    : t_wall_arr,
        'N_bath'    : N_bath,
        'r'         : r,
    }


def replication_study(N_bath, U_values, r_values,
                      t=1.0, V_hyb=1.0, eps_imp=None,
                      num_sweeps=10000, num_reads=200):
    """
    Reproduce the replication factor study (Fig. 16 in the paper):
    for each r in r_values, sweep U and record mean absolute error and
    wall-clock time.

    Parameters
    ----------
    N_bath    : int
    U_values  : array  U sweep
    r_values  : list of int  e.g. [2, 3, 4]
    t, V_hyb, eps_imp : float
    num_sweeps, num_reads : int

    Returns
    -------
    study : dict { r -> {'mean_err': float, 't_mean': float,
                          'errors': array, 'times': array} }
    """
    # First compute ED reference energies
    ED_energies = []
    for U in U_values:
        eps = -U / 2.0 if eps_imp is None else eps_imp
        H = build_siam_hamiltonian(N=N_bath, U=U, t=t, V=V_hyb, eps_imp=eps)
        ED_energies.append(exact_ground_energy(H))
    ED_energies = np.array(ED_energies)

    study = {}
    for r in r_values:
        print(f"\n  Replication study: r={r}")
        xbk_energies = np.zeros(len(U_values))
        times        = np.zeros(len(U_values))

        for i, U in enumerate(U_values):
            eps = -U / 2.0 if eps_imp is None else eps_imp
            H = build_siam_hamiltonian(N=N_bath, U=U, t=t, V=V_hyb, eps_imp=eps)
            xbk_energies[i], times[i] = best_xbk_energy(
                H, r=r, num_sweeps=num_sweeps, num_reads=num_reads
            )

        errors = np.abs(xbk_energies - ED_energies)
        study[r] = {
            'mean_err' : float(errors.mean()),
            't_mean'   : float(times.mean()),
            't_total'  : float(times.sum()),
            'errors'   : errors,
            'times'    : times,
            'E0_XBK'   : xbk_energies,
        }
        print(f"    r={r}: mean_err={errors.mean():.4f}  "
              f"t_mean={times.mean():.1f}s")

    study['ED_energies'] = ED_energies
    study['U_values']    = np.array(U_values)
    return study


# ═══════════════════════════════════════════════════════════════════════════
# 5. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_ground_energy(sweeps, save='siam_figA_ground_energy.png'):
    """
    Fig A: Ground state energy E0(U/t) from XBK-QUBO vs ED.

    Analogous to Fig 14 (left) in the paper for t-V model.
    For SIAM the x-axis is U/t (on-site Coulomb / hopping).
    The Kondo crossover occurs around U/t ~ 4-6 for typical parameters.

    Parameters
    ----------
    sweeps : list of dicts  output of sweep_U for different N_bath
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sweeps)))

    for sw, col in zip(sweeps, colors):
        U = sw['U_values']
        N = sw['N_bath']
        r = sw['r']
        ax.plot(U, sw['E0_ED'],  '-',  color=col, lw=2,
                label=f'ED N={N}')
        ax.plot(U, sw['E0_XBK'], 'o',  color=col, ms=5,
                label=f'XBK r={r} N={N}')

    ax.axvline(x=0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('U / t')
    ax.set_ylabel('Ground state energy $E_0$')
    ax.set_title('SIAM: Ground state energy vs U/t')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved {save}")


def plot_curvature(sweeps, save='siam_figB_curvature.png'):
    """
    Fig B: Numerical curvature E0''(U/t) via central differences.

    Analogous to Fig 14 (right) in the paper.
    A pronounced dip or peak in the curvature signals the Kondo crossover.

    Parameters
    ----------
    sweeps : list of dicts  output of sweep_U
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sweeps)))

    for sw, col in zip(sweeps, colors):
        U   = sw['U_values']
        N   = sw['N_bath']
        r   = sw['r']

        # Curvature of ED energy
        d2_ED  = second_derivative(U, sw['E0_ED'])
        d2_XBK = second_derivative(U, sw['E0_XBK'])

        axes[0].plot(U, d2_ED,  '-',  color=col, lw=2,  label=f'ED N={N}')
        axes[0].plot(U, d2_XBK, '--', color=col, lw=1.5, label=f'XBK r={r} N={N}')

        # Smoothed curvature
        d2_ED_sm  = smooth(d2_ED,  window=5)
        d2_XBK_sm = smooth(d2_XBK, window=5)
        axes[1].plot(U, d2_ED_sm,  '-',  color=col, lw=2,  label=f'ED N={N}')
        axes[1].plot(U, d2_XBK_sm, '--', color=col, lw=1.5, label=f'XBK r={r}')

    for ax, title in zip(axes, ['Raw curvature $E_0\'\'(U/t)$',
                                  'Smoothed curvature']):
        ax.axhline(0, color='grey', lw=0.5, ls='--')
        ax.set_xlabel('U / t')
        ax.set_ylabel("$E_0''$")
        ax.set_title(f'SIAM: {title}')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved {save}")


def plot_replication_study(study, save='siam_figC_replication.png'):
    """
    Fig C: Replication factor study (analogous to Fig 16 in the paper).

    Top-left:  E0(U/t) for different r vs ED
    Top-right: |E_sim - E_ED| on log scale
    Bottom-left:  wall-clock time per run vs r
    Bottom-right: mean error reduction when increasing r
    """
    U       = study['U_values']
    ED      = study['ED_energies']
    r_vals  = [k for k in study.keys()
               if isinstance(k, int)]
    colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(r_vals)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: energy comparison
    ax = axes[0, 0]
    ax.plot(U, ED, 'k--', lw=2, label='ED')
    for r, col in zip(r_vals, colors):
        ax.plot(U, study[r]['E0_XBK'], 'o-', color=col, ms=4,
                label=f'r={r}')
    ax.set_xlabel('U / t'); ax.set_ylabel('$E_0$')
    ax.set_title('SIAM: Energy comparison'); ax.legend(fontsize=8)

    # Top-right: absolute error on log scale
    ax = axes[0, 1]
    for r, col in zip(r_vals, colors):
        ax.semilogy(U, study[r]['errors'] + 1e-8, 'o-', color=col,
                    ms=4, label=f'r={r}')
    ax.set_xlabel('U / t'); ax.set_ylabel('$|E_{sim} - E_{ED}|$')
    ax.set_title('SIAM: Absolute error (log scale)'); ax.legend(fontsize=8)

    # Bottom-left: wall-clock time per run vs r
    ax = axes[1, 0]
    t_means = [study[r]['t_total'] for r in r_vals]
    ax.bar([str(r) for r in r_vals], t_means,
           color=colors, edgecolor='k')
    ax.set_xlabel('r'); ax.set_ylabel('Total wall-clock time (s)')
    ax.set_title('SIAM: Computation time vs r')
    for i, (r, tm) in enumerate(zip(r_vals, t_means)):
        ax.text(i, tm + 0.5, f'{tm:.1f}s', ha='center', fontsize=8)

    # Bottom-right: mean error reduction (%) when increasing r
    ax = axes[1, 1]
    reductions = []
    labels     = []
    for i in range(1, len(r_vals)):
        r_prev = r_vals[i - 1]
        r_curr = r_vals[i]
        e_prev = study[r_prev]['mean_err']
        e_curr = study[r_curr]['mean_err']
        if e_prev > 1e-10:
            pct = 100.0 * (e_prev - e_curr) / e_prev
        else:
            pct = 0.0
        reductions.append(pct)
        labels.append(f'{r_prev}→{r_curr}')
    bar_colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(reductions)))
    ax.bar(labels, reductions, color=bar_colors, edgecolor='k')
    ax.set_xlabel('r increase'); ax.set_ylabel('Mean error reduction (%)')
    ax.set_title('SIAM: Error reduction per replication step')
    for i, (lb, red) in enumerate(zip(labels, reductions)):
        ax.text(i, max(red + 1, 1), f'{red:.1f}%', ha='center', fontsize=9)

    plt.suptitle(f'SIAM Replication Factor Study', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved {save}")


def plot_thermal(thermal_results, save='siam_figD_thermal.png'):
    """
    Fig D: Thermal observables vs effective temperature T.

    Analogous to Fig 11-12 in the paper.
    For SIAM with different N_bath:
      - Binder cumulant U4 (left)
      - Energy susceptibility chi = Var(E)/T (center)
      - Heat capacity Cv (right)

    A crossing point in U4 at a common temperature signals a crossover.

    Parameters
    ----------
    thermal_results : list of tuples (N_bath, dict_from_thermal_observables)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(thermal_results)))

    for (N, res), col in zip(thermal_results, colors):
        T = res['T']
        axes[0].plot(T, res['U4'],  '-o', color=col, ms=3, label=f'N={N}')
        axes[1].plot(T, res['chi'], '-o', color=col, ms=3, label=f'N={N}')
        axes[2].plot(T, res['Cv'],  '-o', color=col, ms=3, label=f'N={N}')

    titles = ['Binder cumulant $U_4$',
              'Energy susceptibility $\\chi$',
              'Heat capacity $C_v$']
    ylabels = ['$U_4$', '$\\chi$', '$C_v$']

    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_xlabel('Effective temperature $T/t$')
        ax.set_ylabel(yl)
        ax.set_title(f'SIAM: {title}')
        ax.legend(fontsize=8)

    plt.suptitle('SIAM: Thermal observables (ED spectrum)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved {save}")


def plot_impurity_physics(sweeps, save='siam_figEF_impurity.png'):
    """
    Fig E + F: Impurity occupancy and double occupancy vs U/t.

    These are the key physical observables for SIAM:

    E: <n_imp> = <n_{0,up} + n_{0,down}> vs U/t
       At particle-hole symmetry (eps = -U/2), this = 1.0 exactly.
       This serves as a sanity check on the implementation.

    F: Double occupancy D = <n_{0,up} n_{0,down}> vs U/t
       D = 0.25 for U=0 (non-interacting half-filling)
       D -> 0 as U -> infinity (singly occupied impurity, Kondo regime)
       The crossover from D ~ 0.25 to D ~ 0 marks the onset of Kondo physics.
       The inflection point of D(U) is the Kondo crossover scale.

    Parameters
    ----------
    sweeps : list of dicts  output of sweep_U
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sweeps)))

    for sw, col in zip(sweeps, colors):
        U = sw['U_values']
        N = sw['N_bath']
        axes[0].plot(U, sw['n_imp'],  'o-', color=col, ms=5, lw=2,
                     label=f'N_bath={N}')
        axes[1].plot(U, sw['D_occ'],  'o-', color=col, ms=5, lw=2,
                     label=f'N_bath={N}')

    axes[0].axhline(1.0, color='k', ls='--', lw=1, label='PH symmetry')
    axes[0].set_xlabel('U / t')
    axes[0].set_ylabel(r'$\langle n_{imp} \rangle$')
    axes[0].set_title('SIAM: Impurity occupancy')
    axes[0].set_ylim([0, 2])
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.25, color='k', ls='--', lw=1, label='U=0 limit')
    axes[1].axhline(0.0,  color='grey', ls=':', lw=1, label='U→∞ limit')
    axes[1].set_xlabel('U / t')
    axes[1].set_ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
    axes[1].set_title('SIAM: Double occupancy (Kondo diagnostic)')
    axes[1].legend(fontsize=8)

    plt.suptitle('SIAM: Impurity physics vs U/t', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved {save}")


def plot_summary_table(sweeps, study, save='siam_table.png'):
    """
    Table: analogous to Table I and II in the paper.

    Columns: N_bath, n_modes, n_BK_qubits, n_XBK_qubits(r=2),
             mean_abs_error, runtime

    Parameters
    ----------
    sweeps : list of dicts
    study  : dict  output of replication_study
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    r_vals = [k for k in study.keys() if isinstance(k, int)]

    headers = ['N_bath', 'Modes', 'BK qubits', 'XBK(r=2) qubits',
               'Mean |err|', 'Total time (s)']

    rows = []
    for sw in sweeps:
        N  = sw['N_bath']
        n_modes   = 2 * (N + 1)
        # BK qubit count: build once
        H = build_siam_hamiltonian(N=N, U=4.0, t=1.0)
        Q = bravyi_kitaev(H)
        n_bk = op.count_qubits(Q)
        n_xbk = 2 * n_bk

        r2_err = study.get(2, {}).get('mean_err', float('nan'))
        r2_t   = study.get(2, {}).get('t_total',  float('nan'))

        rows.append([str(N), str(n_modes), str(n_bk), str(n_xbk),
                     f'{r2_err:.4f}', f'{r2_t:.1f}'])

    table = ax.table(cellText=rows, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax.set_title('SIAM Pipeline Resource Summary', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("SIAM Pipeline: Reproducing Nagpal et al. 2025 for SIAM")
    print("=" * 65)

    # ── Parameters ───────────────────────────────────────────────────────
    # U sweep range: from 0 (non-interacting) to 8t (strongly correlated)
    # The Kondo crossover T_K ~ t * exp(-pi*U / (8*Gamma)) where Gamma=pi*V^2/W
    # occurs around U/t ~ 3-6 for V=1, t=1 (W=2t).
    U_values_fine   = np.linspace(0.0, 8.0, 17).tolist()   # fine for ED
    U_values_coarse = np.linspace(0.5, 7.5,  8).tolist()   # coarse for XBK

    t_hop  = 1.0
    V_hyb  = 1.0
    r_main = 2        # main replication factor (paper uses r=2 for t-V)

    # ── FIG A + B + E + F: Ground energy sweep for N_bath = 1, 2, 3 ────
    print("\nSection 1: Ground energy sweeps")
    sweeps = []
    for N_bath in [1, 2, 3]:
        print(f"\n  N_bath = {N_bath}")
        sw = sweep_U(
            N_bath     = N_bath,
            U_values   = U_values_coarse,
            t          = t_hop,
            V_hyb      = V_hyb,
            eps_imp    = None,    # particle-hole symmetric: eps = -U/2
            r          = r_main,
            num_sweeps = 5000,
            num_reads  = 100,
        )
        sweeps.append(sw)

    plot_ground_energy(sweeps)
    plot_curvature(sweeps)
    plot_impurity_physics(sweeps)

    # ── FIG C: Replication factor study ─────────────────────────────────
    print("\nSection 2: Replication factor study (N_bath=1)")
    r_values_study = [2, 3, 4]
    study = replication_study(
        N_bath     = 1,
        U_values   = U_values_coarse,
        r_values   = r_values_study,
        t          = t_hop,
        V_hyb      = V_hyb,
        num_sweeps = 5000,
        num_reads  = 100,
    )
    plot_replication_study(study)
    plot_summary_table(sweeps, study)

    # ── FIG D: Thermal observables ───────────────────────────────────────
    print("\nSection 3: Thermal observables")
    T_values = np.logspace(-1, 1, 40)   # T from 0.1t to 10t
    thermal_results = []

    for N_bath in [1, 2, 3]:
        eps = -4.0 / 2.0    # fixed U=4 for thermal study
        H = build_siam_hamiltonian(N=N_bath, U=4.0, t=t_hop,
                                   V=V_hyb, eps_imp=eps)
        evals = exact_spectrum(H, k=min(50, 2**(2*(N_bath+1)) - 2))
        res   = thermal_observables(evals, T_values)
        thermal_results.append((N_bath, res))
        print(f"  N_bath={N_bath}: spectrum computed, {len(evals)} levels")

    plot_thermal(thermal_results)

    print("\nAll figures saved.  Pipeline complete.")