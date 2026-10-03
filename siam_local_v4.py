"""
siam_local.py  (v4 — parallelised for 4-core macOS, all bugs fixed)
=====================================================================
Self-contained SIAM script for N_bath = 1, 2, 3.
Exact diagonalisation only (no SA). All observables computed exactly.

PARALLELISATION STRATEGY (quad-core Intel i5)
----------------------------------------------
Three levels of parallelism, each chosen to match macOS process semantics:

Level 1 — across N_bath values (3 independent sweeps):
  ProcessPoolExecutor(max_workers=3) runs N=1, N=2, N=3 concurrently.
  Each worker is a separate OS process, bypassing Python's GIL and
  using one physical core each.

Level 2 — across U/V points within each N (21 points, each independent):
  ProcessPoolExecutor(max_workers=4) inside each worker process.
  Each U/V point requires 3 exact diagonalisations (E0, E+, E-) and
  one Wasserstein LP. These are all independent of each other.
  On a quad-core machine with 3 outer workers, the inner pool is capped
  at 2 to avoid oversubscription (3 × 2 = 6, leaving headroom for macOS).

  NOTE: fidelity and Wasserstein between consecutive U/V points require
  the psi vectors from adjacent points. We therefore compute all points
  in parallel first, then stitch consecutive pairs in a single serial pass.

Level 3 — within each U/V point (the 3 exact_gs calls):
  These are tiny (< 1ms each for N=1,2 and < 5ms for N=3) so the
  overhead of spawning processes exceeds the gain. They run serially
  within the worker.

WHY NO SA
---------
For SIAM N=1 (BK Hamiltonian contains 3-body XYY/YYX Pauli terms),
exhaustive enumeration over all 2^8 spin configurations shows that
min_{p,s} H'_p(s)/C_p(s) = -1.0, not the true E0 = -2.236.
With r=3 the bound improves to -1.667. The required r grows with
the complexity of the ground-state structure and is not reached at
small r for SIAM. SA results for the paper come from the HPC runs
(with qanneal, large SA_READS, and appropriate r).

SECTOR PROJECTION (mandatory)
-------------------------------
H commutes with N_tot (total electrons) and S_z.
At half-filling N_tot = N+1:
  N=1 → (N_tot=2, Sz=0)    — unique ground state
  N=2 → (N_tot=3, Sz=+1/2) — non-degenerate within sector (SU(2) gap)
  N=3 → (N_tot=4, Sz=0)    — unique ground state
Without projection, scipy.eigsh mixes degenerate Sz sectors for N=2,
making fidelity/Wasserstein/spin_corr meaningless.

FINITE-DIFFERENCE STEP
-----------------------
dU = max(FD_REL * U, FD_ABS) * V  avoids zero step at U=0.

WASSERSTEIN LP
--------------
The standard transport LP has 2n equality constraints but rank 2n-1
(sum of row marginals = sum of col marginals = 1, so one constraint
is redundant). Feeding the full 2n × n² system to HiGHS triggers
infeasibility at some U/V values due to floating-point inconsistency.
Fix: drop the last column-marginal constraint (standard in LP literature).

OBSERVABLES
-----------
E0_exact  : exact ground-state energy in (N_tot, Sz) sector
docc      : <n_{imp,up} n_{imp,dn}>  (double occupancy)
docc_fd   : dE0/dU  by finite difference  (Hellmann-Feynman check;
            identity docc_fd = docc - 1/4 holds only for variable-N)
d2E0      : d²E0/dU²  (locates Kondo crossover scale)
spin_corr : <S_z^imp · S_z^bath1>  (singlet formation signature)
fidelity  : |⟨ψ(U_k)|ψ(U_{k+1})⟩|²  (consecutive U/V steps)
d_WGS     : W_1(|ψ(U_k)|², |ψ(U_{k+1})|²) with Hamming ground metric
            (key geometric observable in the XBK paper)
"""

from __future__ import annotations
import os, csv, itertools
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── imports needed inside worker processes (must be top-level for pickling) ──
import openfermion as of
from openfermion import FermionOperator
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── output directory ────────────────────────────────────────────────────────
OUT = "output/siam_local"
os.makedirs(OUT, exist_ok=True)

# ─── global parameters ───────────────────────────────────────────────────────
T_HOP   = 1.0
V_HYB   = 1.0
EPS_IMP = 0.0

UV_VALS     = np.arange(0.0, 8.0 + 1e-9, 0.4)   # 21 points
N_BATH_LIST = [1, 2, 3]

# Physical (N_tot, 2·Sz) sector per N_bath
SECTOR = {1: (2, 0), 2: (3, 1), 3: (4, 0)}

# Finite-difference step: dU = max(FD_REL·U, FD_ABS) · V
FD_REL = 0.05
FD_ABS = 0.05

# Parallelism caps (tuned for 4-core i5 — avoid oversubscription)
OUTER_WORKERS = 3   # one per N_bath value
INNER_WORKERS = 2   # per-N worker; 3×2 = 6 total logical threads max

# ─── 1. SIAM Hamiltonian ─────────────────────────────────────────────────────
def _mode(site: int, spin: int) -> int:
    """Spin-orbital mode index: mode = 2·site + spin  (spin 0=↑, 1=↓)."""
    return 2 * site + spin


def build_siam(N: int, U: float,
               t: float = T_HOP, V: float = V_HYB,
               eps_imp: float = EPS_IMP) -> FermionOperator:
    """
    H = U n_{0↑} n_{0↓}  −  (U/2)(n_{0↑}+n_{0↓})  +  U/4
      + Σ_b  ε_b (n_{b↑}+n_{b↓})
      + Σ_b  V_b Σ_σ (a†_{0σ} a_{bσ} + h.c.)

    Bath parameters (star geometry / Wilson chain):
        ε_b = −2t cos(bπ/(N+1))
        V_b =  V √(2/(N+1)) √(1 − (ε_b/2t)²)
    """
    b_idx = np.arange(1, N + 1)
    eps   = -2.0 * t * np.cos(b_idx * np.pi / (N + 1))
    Vb    =  V * np.sqrt(2.0 / (N + 1)) * np.sqrt(1.0 - (eps / (2.0 * t))**2)

    H = FermionOperator()
    # Impurity Hubbard term (normal-ordered)
    H += FermionOperator(((0,1),(1,1),(1,0),(0,0)), U)
    # Impurity one-body (particle-hole symmetric form)
    for s in range(2):
        H += FermionOperator(((_mode(0,s),1),(_mode(0,s),0)), -U/2.0 + eps_imp)
    H += FermionOperator((), U / 4.0)
    # Bath on-site energies
    for b, (e, v) in enumerate(zip(eps, Vb), 1):
        bm0, bm1 = _mode(b, 0), _mode(b, 1)
        H += FermionOperator(((bm0,1),(bm0,0)), e)
        H += FermionOperator(((bm1,1),(bm1,0)), e)
        # Hybridisation
        H += FermionOperator(((0,1),(bm0,0)),  v)
        H += FermionOperator(((bm0,1),(0,0)),  v)
        H += FermionOperator(((1,1),(bm1,0)),  v)
        H += FermionOperator(((bm1,1),(1,0)),  v)
    return H


# ─── 2. Sector basis ─────────────────────────────────────────────────────────
def sector_basis(n_modes: int, n_elec: int, sz2: int) -> np.ndarray:
    """
    Integer array of Fock-space states with exactly n_elec electrons
    and (n_up − n_dn) = sz2.
    Bit k of an integer = occupation of mode k.
    Even-index modes = spin-up; odd-index modes = spin-down.
    """
    mask_up = sum(1 << k for k in range(0, n_modes, 2))
    mask_dn = sum(1 << k for k in range(1, n_modes, 2))
    return np.array(
        [s for s in range(2**n_modes)
         if (bin(s & mask_up).count('1') + bin(s & mask_dn).count('1') == n_elec
             and bin(s & mask_up).count('1') - bin(s & mask_dn).count('1') == sz2)],
        dtype=int
    )


# ─── 3. Exact ground state in the physical sector ────────────────────────────
def exact_gs(N: int, U: float):
    """
    Returns (E0, psi0, basis).

    E0    : lowest eigenvalue in the (N_tot, Sz) sector (float)
    psi0  : corresponding normalised eigenvector (1-D numpy array)
    basis : integer array of sector basis states
    """
    n_modes      = 2 * (N + 1)
    n_elec, sz2  = SECTOR[N]
    Hsp   = of.linalg.get_sparse_operator(build_siam(N, U), n_qubits=n_modes)
    basis = sector_basis(n_modes, n_elec, sz2)
    Hsec  = Hsp[np.ix_(basis, basis)].toarray()
    vals, vecs = scipy.linalg.eigh(Hsec)          # real symmetric → ascending
    psi = vecs[:, 0]
    psi /= np.linalg.norm(psi)
    return float(vals[0]), psi, basis


# ─── 4. Observables ──────────────────────────────────────────────────────────
def double_occupancy(psi: np.ndarray, basis: np.ndarray) -> float:
    """
    ⟨n_{imp↑} n_{imp↓}⟩ = Σ_state |ψ(state)|² · bit0(state) · bit1(state)
    Modes 0 and 1 carry spin-up and spin-down on the impurity (site 0).
    """
    return float(
        np.sum(np.abs(psi)**2
               * ((basis >> 0) & 1).astype(float)
               * ((basis >> 1) & 1).astype(float))
    )


def spin_corr(psi: np.ndarray, basis: np.ndarray) -> float:
    """
    ⟨S_z^imp · S_z^bath1⟩ = (1/4) ⟨(n_{0↑}−n_{0↓})(n_{1↑}−n_{1↓})⟩

    Modes 0,1 → impurity (site 0).
    Modes 2,3 → first bath site (site 1).
    """
    sz_imp  = ((basis >> 0) & 1).astype(float) - ((basis >> 1) & 1).astype(float)
    sz_bath = ((basis >> 2) & 1).astype(float) - ((basis >> 3) & 1).astype(float)
    return float(np.sum(np.abs(psi)**2 * sz_imp * sz_bath)) * 0.25


def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """F = |⟨ψ₁|ψ₂⟩|².  Both vectors must live in the same sector subspace."""
    return float(abs(np.dot(psi1.conj(), psi2))**2)


def wasserstein_hamming(p: np.ndarray, q: np.ndarray,
                        basis: np.ndarray) -> float:
    """
    1-Wasserstein distance between probability vectors p and q
    over the sector basis, with Hamming distance as ground metric.

        W₁(p,q) = min_{π≥0}  Σ_{x,y} π(x,y) · d_H(basis[x], basis[y])
        s.t.  Σ_y π(x,y) = p[x]   ∀x   (row marginals)
              Σ_x π(x,y) = q[y]   ∀y   (col marginals)

    The transport LP has 2n equality constraints but rank 2n−1 (sum of
    row marginals = sum of col marginals = 1, one is redundant).
    We drop the last column-marginal constraint to obtain a non-singular
    system; this is the standard fix and does not change the solution.

    Marginals are renormalised before the LP to guard against floating-
    point drift in |ψ|².
    """
    p = np.array(p, dtype=np.float64); p /= p.sum()
    q = np.array(q, dtype=np.float64); q /= q.sum()
    n = len(p)

    # Cost matrix: C[i,j] = popcount(basis[i] XOR basis[j])
    C_mat = np.array(
        [[bin(int(basis[i]) ^ int(basis[j])).count('1') for j in range(n)]
         for i in range(n)],
        dtype=np.float64
    )

    # (2n−1) × n² constraint matrix (last col-marginal dropped)
    Aeq = np.zeros((2*n - 1, n*n))
    beq = np.zeros(2*n - 1)
    for i in range(n):
        Aeq[i, i*n:(i+1)*n] = 1.0
        beq[i] = p[i]
    for j in range(n - 1):            # n−1 column marginals (last dropped)
        Aeq[n + j, j::n] = 1.0
        beq[n + j] = q[j]

    res = linprog(C_mat.flatten(), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)] * (n*n), method='highs')
    return float(res.fun) if res.success else float('nan')


# ─── 5. Worker: compute ONE U/V point ────────────────────────────────────────
def _compute_point(args):
    """
    Compute all quantities for a single (N, uv) pair.
    Returns a dict of scalar results plus psi and basis (needed for
    the consecutive-pair observables fidelity and d_WGS).

    This function is called inside a worker process, so all imports
    must be available at module level.
    """
    N, uv = args
    U  = uv * V_HYB
    dU = max(FD_REL * U, FD_ABS) * V_HYB

    E0, psi, basis = exact_gs(N, U)
    Ep, _,   _     = exact_gs(N, U + dU)
    Em, _,   _     = exact_gs(N, U - dU)

    docc_fd_v = (Ep - Em) / (2 * dU)
    d2_v      = (Ep - 2*E0 + Em) / dU**2
    docc_v    = double_occupancy(psi, basis)
    sc_v      = spin_corr(psi, basis)

    return {
        'uv':      float(uv),
        'E0':      E0,
        'docc':    docc_v,
        'docc_fd': docc_fd_v,
        'd2E0':    d2_v,
        'spin_corr': sc_v,
        'psi':     psi,         # kept for fidelity / Wasserstein
        'basis':   basis,       # same for all points within one N, but cheap
    }


# ─── 6. Sweep for one N_bath (runs inside an outer worker process) ────────────
def run_sweep(N: int) -> dict:
    """
    Parallelise over U/V points using an inner ProcessPoolExecutor.
    Returns a dict of lists (one value per U/V point) plus arrays for
    the consecutive-pair observables (fidelity, d_WGS).
    """
    import sys
    print(f"[N={N}] starting sweep ({len(UV_VALS)} points)", flush=True)

    # ── parallel computation of all independent per-point quantities ──────────
    args = [(N, uv) for uv in UV_VALS]
    point_results = [None] * len(UV_VALS)

    with ProcessPoolExecutor(max_workers=INNER_WORKERS) as pool:
        futures = {pool.submit(_compute_point, a): idx
                   for idx, a in enumerate(args)}
        for fut in as_completed(futures):
            idx = futures[fut]
            point_results[idx] = fut.result()

    # Sort by index (as_completed does not preserve order)
    point_results.sort(key=lambda r: r['uv'])

    # ── serial pass for consecutive-pair observables (fidelity, d_WGS) ────────
    fid_vals  = [1.0]   # first point has no predecessor → 1
    dwgs_vals = [0.0]
    for i in range(1, len(point_results)):
        psi_prev   = point_results[i-1]['psi']
        psi_curr   = point_results[i]['psi']
        basis      = point_results[i]['basis']
        p_prev     = np.abs(psi_prev)**2
        p_curr     = np.abs(psi_curr)**2
        fid_vals.append(fidelity(psi_prev, psi_curr))
        dwgs_vals.append(wasserstein_hamming(p_prev, p_curr, basis))

    # ── assemble output ────────────────────────────────────────────────────────
    res = {
        'uv':        [r['uv']        for r in point_results],
        'E0_exact':  [r['E0']        for r in point_results],
        'docc':      [r['docc']      for r in point_results],
        'docc_fd':   [r['docc_fd']   for r in point_results],
        'd2E0':      [r['d2E0']      for r in point_results],
        'spin_corr': [r['spin_corr'] for r in point_results],
        'fidelity':  fid_vals,
        'd_WGS':     dwgs_vals,
    }

    for i, uv in enumerate(res['uv']):
        print(f"  [N={N}] U/V={uv:.2f}  E0={res['E0_exact'][i]:.5f}  "
              f"docc={res['docc'][i]:.4f}  d2={res['d2E0'][i]:.4f}  "
              f"Scorr={res['spin_corr'][i]:.4f}  "
              f"F={res['fidelity'][i]:.4f}  W={res['d_WGS'][i]:.4f}",
              flush=True)
    return res


# ─── 7. Plotting ──────────────────────────────────────────────────────────────
def plot_all(all_res: dict) -> None:
    colors  = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
    markers = {1: 'o',       2: 's',       3: '^'}
    panels = [
        ('E0_exact',   r'$E_0$',
                       'Ground-state energy (physical sector)'),
        ('docc',       r'$\langle n_\uparrow n_\downarrow\rangle$',
                       'Double occupancy'),
        ('docc_fd',    r'$dE_0/dU$',
                       r'$dE_0/dU$ (finite difference)'),
        ('d2E0',       r'$d^2E_0/dU^2$',
                       r'$d^2E_0/dU^2$ (Kondo crossover scale)'),
        ('spin_corr',  r'$\langle S_z^{\rm imp}\,S_z^{\rm bath}\rangle$',
                       'Impurity–bath spin correlation'),
        ('fidelity',   r'$|\langle\psi_k|\psi_{k+1}\rangle|^2$',
                       'Quantum fidelity (consecutive $U/V$ steps)'),
        ('d_WGS',      r'$W_1$ (Wasserstein, Hamming)',
                       r'$d_{\rm WGS}$ (consecutive $U/V$ steps)'),
    ]
    for key, ylabel, title in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        for N, r in all_res.items():
            ax.plot(r['uv'], r[key],
                    color=colors[N], marker=markers[N],
                    markersize=4, linewidth=1.5,
                    label=f'$N_{{\\rm bath}}={N}$')
        ax.set_xlabel(r'$U/V$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'{key}.pdf'), dpi=150)
        plt.close(fig)
        print(f"  Saved {key}.pdf")


# ─── 8. CSV export ────────────────────────────────────────────────────────────
def save_csv(N: int, res: dict) -> None:
    cols = ['uv', 'E0_exact', 'docc', 'docc_fd', 'd2E0',
            'spin_corr', 'fidelity', 'd_WGS']
    path = os.path.join(OUT, f'siam_N{N}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(res['uv'])):
            w.writerow([res[c][i] for c in cols])
    print(f"  Saved siam_N{N}.csv")


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    t0 = time.time()

    all_res = {}

    # Outer parallelism: run N=1, N=2, N=3 concurrently on separate cores.
    # macOS requires the 'spawn' start method (default on macOS since Python 3.8).
    # Guard with __name__=='__main__' to prevent recursive spawning.
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as pool:
        futures = {pool.submit(run_sweep, N): N for N in N_BATH_LIST}
        for fut in as_completed(futures):
            N = futures[fut]
            all_res[N] = fut.result()

    # Sort results by N for consistent CSV / plot order
    for N in N_BATH_LIST:
        save_csv(N, all_res[N])

    plot_all(all_res)

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.1f}s.  Outputs in {OUT}/")
