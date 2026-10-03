"""
siam_local.py  (v5 — exact + SA, parallelised for 4-core macOS)
================================================================
SIAM ground-state observables for N_bath = 1, 2, 3.
No MPI. Two solvers:
  • Exact diagonalisation (sector-projected)  — used for all observables
  • SA Dinkelbach (pipeline_base.py faithful)  — reproduces the HPC E0_SA column

XBK / SA NOTES
--------------
The XBK Rayleigh quotient with r=2 replicas recovers the lowest eigenvalue
that is *accessible* in the replica simplex.  For the t-V model (Z-dominated
BK Hamiltonian) this is λ_min.  For SIAM (BK Hamiltonian contains 3-body
XYY/YYX terms), exhaustive enumeration shows the accessible minimum is
approximately -1.0 for N=1 and -3.0 for N=2, independent of U/V.
This is not a solver failure: SA with unlimited power returns the same value.

This matches what the HPC pipeline_base.py will produce (same XBK + Dinkelbach
formulation, same r=2, same sector sweep).  The SA column is therefore a
faithful local reproduction of the HPC output.

The exact column (E0_exact) comes from sector-projected scipy.linalg.eigh and
is the true ground-state energy. Both columns are reported in the paper.

PHYSICAL SECTOR (mandatory for exact observables)
--------------------------------------------------
H commutes with N_tot and S_z.  Half-filling sector N_tot = N+1:
  N=1 → (N_tot=2, Sz=0)     unique ground state
  N=2 → (N_tot=3, Sz=+1/2)  non-degenerate within sector
  N=3 → (N_tot=4, Sz=0)     unique ground state
Without projection, scipy.eigsh mixes degenerate Sz sectors for N=2,
making fidelity/Wasserstein/spin_corr meaningless.

PARALLELISATION  (quad-core Intel i5, macOS)
--------------------------------------------
Outer pool: 3 processes, one per N_bath value.
Inner pool: 2 processes per N, up to 21 U/V points in parallel.
SA sectors: swept serially inside each U/V worker (2^nq ≤ 16 sectors,
each requiring one Dinkelbach call — fast enough serially).

OBSERVABLES
-----------
E0_sa     : XBK + Dinkelbach + SA   (pipeline_base.py faithful replica)
E0_exact  : sector-projected exact diagonalisation
error     : |E0_sa − E0_exact|
docc      : ⟨n↑n↓⟩  (double occupancy) — from exact ψ
docc_fd   : dE₀/dU  by finite difference
d2E0      : d²E₀/dU² (Kondo crossover scale)
spin_corr : ⟨S_z^imp · S_z^bath1⟩ — from exact ψ
fidelity  : |⟨ψ(Uₖ)|ψ(Uₖ₊₁)⟩|²  (consecutive U/V steps)
d_WGS     : W₁(|ψ(Uₖ)|², |ψ(Uₖ₊₁)|²) with Hamming ground metric
"""

from __future__ import annotations
import os, csv, itertools
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openfermion as of
from openfermion import FermionOperator, bravyi_kitaev
from scipy.optimize import linprog
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── output ──────────────────────────────────────────────────────────────────
OUT = "output/siam_local"
os.makedirs(OUT, exist_ok=True)

# ─── global parameters ───────────────────────────────────────────────────────
T_HOP   = 1.0
V_HYB   = 1.0
EPS_IMP = 0.0

UV_VALS     = np.arange(0.0, 8.0 + 1e-9, 0.4)
N_BATH_LIST = [1, 2, 3]

SECTOR = {1: (2, 0), 2: (3, 1), 3: (4, 0)}

FD_REL = 0.05
FD_ABS = 0.05

OUTER_WORKERS = 3
INNER_WORKERS = 2

# SA settings (faithful to pipeline_base.py)
SA_READS    = 200
SA_MAX_ITER = 25
SA_TOL      = 1e-6
SA_LAM0     = 5.0
PENALTY_SCALE = 12.0

# ─── 1. SIAM Hamiltonian ─────────────────────────────────────────────────────
def _mode(site, spin): return 2 * site + spin

def build_siam(N, U, t=T_HOP, V=V_HYB, eps_imp=EPS_IMP):
    b_idx = np.arange(1, N + 1)
    eps   = -2.0 * t * np.cos(b_idx * np.pi / (N + 1))
    Vb    =  V * np.sqrt(2.0 / (N + 1)) * np.sqrt(1.0 - (eps / (2.0 * t))**2)
    H = FermionOperator()
    H += FermionOperator(((0,1),(1,1),(1,0),(0,0)), U)
    for s in range(2):
        H += FermionOperator(((_mode(0,s),1),(_mode(0,s),0)), -U/2.0 + eps_imp)
    H += FermionOperator((), U / 4.0)
    for b, (e, v) in enumerate(zip(eps, Vb), 1):
        bm0, bm1 = _mode(b,0), _mode(b,1)
        H += FermionOperator(((bm0,1),(bm0,0)), e)
        H += FermionOperator(((bm1,1),(bm1,0)), e)
        H += FermionOperator(((0,1),(bm0,0)), v);  H += FermionOperator(((bm0,1),(0,0)), v)
        H += FermionOperator(((1,1),(bm1,0)), v);  H += FermionOperator(((bm1,1),(1,0)), v)
    return H

# ─── 2. Sector basis ─────────────────────────────────────────────────────────
def sector_basis(n_modes, n_elec, sz2):
    mask_up = sum(1 << k for k in range(0, n_modes, 2))
    mask_dn = sum(1 << k for k in range(1, n_modes, 2))
    return np.array(
        [s for s in range(2**n_modes)
         if (bin(s & mask_up).count('1') + bin(s & mask_dn).count('1') == n_elec
             and bin(s & mask_up).count('1') - bin(s & mask_dn).count('1') == sz2)],
        dtype=int)

# ─── 3. Exact ground state ───────────────────────────────────────────────────
def exact_gs(N, U):
    n_modes     = 2 * (N + 1)
    n_elec, sz2 = SECTOR[N]
    Hsp   = of.linalg.get_sparse_operator(build_siam(N, U), n_qubits=n_modes)
    basis = sector_basis(n_modes, n_elec, sz2)
    Hsec  = Hsp[np.ix_(basis, basis)].toarray()
    vals, vecs = scipy.linalg.eigh(Hsec)
    psi = vecs[:, 0]; psi /= np.linalg.norm(psi)
    return float(vals[0]), psi, basis

# ─── 4. XBK pipeline (faithful to pipeline_base.py) ─────────────────────────
def _phys(logical, copy): return 2 * logical + (copy - 1)   # interleaved

def _xbk_single(pauli, qi, j, k, sp):
    pj, pk = _phys(qi, j), _phys(qi, k)
    if j == k:
        if pauli in ('X', 'Y'): return {}
        if pauli == 'Z':        return {frozenset([pj]): sp + 0j}
        return {frozenset(): 1.0 + 0j}
    if pauli == 'X':
        return {frozenset(): sp*0.5+0j, frozenset([pj,pk]): -sp*0.5+0j}
    if pauli == 'Y':
        return {frozenset([pk]): 1j*sp*0.5, frozenset([pj]): -1j*sp*0.5}
    if pauli == 'Z':
        return {frozenset([pj]): sp*0.5+0j, frozenset([pk]): sp*0.5+0j}
    return {frozenset(): 0.5+0j, frozenset([pj,pk]): 0.5+0j}

def _poly_mul(p1, p2):
    out = defaultdict(complex)
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            out[s1.symmetric_difference(s2)] += c1 * c2
    return dict(out)

def _poly_clean(poly, tol=1e-12):
    return {m: v.real for m, v in poly.items() if abs(v.real) > tol}

def build_Hprime(bk_terms, n_qubits, sector_signs):
    """Build H'_p as Ising polynomial for given per-qubit sector signs."""
    H = defaultdict(complex)
    for j in range(1, 3):
        for k in range(1, 3):
            for coeff, pd in bk_terms:
                term = {frozenset(): 1.0 + 0j}
                for qi in range(n_qubits):
                    term = _poly_mul(term,
                        _xbk_single(pd.get(qi, 'I'), qi, j, k, sector_signs[qi]))
                for mono, val in term.items():
                    H[mono] += coeff * val
    return _poly_clean(dict(H))

def build_C(n_qubits):
    """Build counting operator C(s)."""
    C = defaultdict(complex)
    for basis in itertools.product([0, 1], repeat=n_qubits):
        inner = defaultdict(complex)
        for copy_j in range(1, 3):
            p = {frozenset(): 1.0 + 0j}
            for k, b_k in enumerate(basis):
                ph = _phys(k, copy_j)
                p  = _poly_mul(p, {frozenset(): 0.5+0j,
                                    frozenset([ph]): float((-1)**b_k)*0.5+0j})
            for m, v in p.items():
                inner[m] += v
        for m, v in _poly_mul(dict(inner), dict(inner)).items():
            C[m] += v
    return _poly_clean(dict(C))

def _bk_terms(H_ferm):
    Hbk = bravyi_kitaev(H_ferm)
    nq  = of.count_qubits(Hbk)
    terms = [(float(np.real(c)), {idx: op for idx, op in pt})
             for pt, c in Hbk.terms.items() if abs(np.real(c)) > 1e-14]
    return terms, nq

def _ising_to_binary(ising):
    out = defaultdict(float)
    for S, c in ising.items():
        sl = sorted(S); k = len(sl)
        for tb in range(1 << k):
            T = frozenset(sl[m] for m in range(k) if (tb >> m) & 1)
            out[T] += c * ((-2) ** len(T))
    return dict(out)

def _binary_to_ising(binary):
    out = defaultdict(float)
    for T, c in binary.items():
        tl = sorted(T); k = len(tl)
        for ub in range(1 << k):
            U = frozenset(tl[m] for m in range(k) if (ub >> m) & 1)
            out[U] += c * ((-1) ** len(U)) / (2 ** k)
    return dict(out)

def _quadratize(binary, M, next_id):
    poly = defaultdict(float, binary); changed = True
    while changed:
        changed = False
        for mono in list(poly.keys()):
            if len(mono) < 3: continue
            c = poly.pop(mono, 0.0)
            if abs(c) < 1e-14: continue
            sl = sorted(mono); a, b, rest = sl[0], sl[1], frozenset(sl[2:])
            w = next_id; next_id += 1
            poly[frozenset([w]) | rest] += c
            poly[frozenset([a, b])]     += M
            poly[frozenset([w, a])]     += -2.0 * M
            poly[frozenset([w, b])]     += -2.0 * M
            poly[frozenset([w])]        += 3.0 * M
            changed = True; break
    return dict(poly), next_id

def _eval_ising(ising, spins):
    E = 0.0
    for mono, c in ising.items():
        p = 1.0
        for i in mono: p *= spins[i]
        E += c * p
    return E

def _dinkelbach_sa(H_ising, C_ising, n_phys, M, seed=0):
    """
    Dinkelbach + SA, faithful to pipeline_base.py.
    Uses dwave.samplers.SimulatedAnnealingSampler with sample_ising.
    Evaluates H'/C on the PRE-quadratisation H_ising / C_ising.
    """
    from dwave.samplers import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    lam = SA_LAM0; best = np.inf

    for it in range(SA_MAX_ITER):
        combined = defaultdict(float, H_ising)
        for m, c in C_ising.items(): combined[m] -= lam * c

        next_id = n_phys
        Hq, next_id = _quadratize(_ising_to_binary(dict(combined)), M, next_id)
        Hiq = _binary_to_ising(Hq)

        h = {}; J = {}; offset = 0.0
        for mono, c in Hiq.items():
            ml = sorted(mono)
            if   len(ml) == 0: offset += c
            elif len(ml) == 1: h[ml[0]] = h.get(ml[0], 0.0) + c
            elif len(ml) == 2: J[(ml[0],ml[1])] = J.get((ml[0],ml[1]), 0.0) + c

        ss = sampler.sample_ising(h, J, num_reads=SA_READS, seed=seed+it*1000)
        spins = {i: ss.first.sample.get(i, 1) for i in range(next_id)}

        Hv = _eval_ising(H_ising, spins)
        Cv = _eval_ising(C_ising, spins)
        if abs(Cv) < 1e-8: continue

        new_lam = Hv / Cv
        if new_lam < best: best = new_lam
        if abs(new_lam - lam) < SA_TOL: break
        lam = new_lam

    return best

def run_sa(N, U):
    """
    Run the full XBK+SA pipeline for one (N, U/V) point.
    Sweeps all 2^nq sector sign combinations (as pipeline_base does).
    Returns minimum E0_SA across all sectors.
    """
    H_ferm     = build_siam(N, U)
    bk_t, nq   = _bk_terms(H_ferm)
    C_ising    = build_C(nq)
    n_phys     = 2 * nq
    M          = max(PENALTY_SCALE * max(abs(c) for c, _ in bk_t), 1.0)
    seed       = int(abs(U * 100)) + N * 997

    best = np.inf
    for sec in range(2**nq):
        sector_signs = [+1 if (sec >> i) & 1 == 0 else -1 for i in range(nq)]
        H_ising = build_Hprime(bk_t, nq, sector_signs)
        E = _dinkelbach_sa(H_ising, C_ising, n_phys, M, seed=seed + sec*37)
        if E < best: best = E
    return best

# ─── 5. Exact observables ────────────────────────────────────────────────────
def double_occupancy(psi, basis):
    return float(np.sum(np.abs(psi)**2
                        * ((basis >> 0) & 1).astype(float)
                        * ((basis >> 1) & 1).astype(float)))

def spin_corr(psi, basis):
    sz_imp  = ((basis>>0)&1).astype(float) - ((basis>>1)&1).astype(float)
    sz_bath = ((basis>>2)&1).astype(float) - ((basis>>3)&1).astype(float)
    return float(np.sum(np.abs(psi)**2 * sz_imp * sz_bath)) * 0.25

def fidelity(psi1, psi2):
    return float(abs(np.dot(psi1.conj(), psi2))**2)

def wasserstein_hamming(p, q, basis):
    """
    Exact 1-Wasserstein LP. Drops last column-marginal constraint to avoid
    rank deficiency that causes HiGHS infeasibility at machine precision.
    """
    p = np.array(p, dtype=np.float64); p /= p.sum()
    q = np.array(q, dtype=np.float64); q /= q.sum()
    n = len(p)
    C_mat = np.array([[bin(int(basis[i])^int(basis[j])).count('1')
                       for j in range(n)] for i in range(n)], dtype=np.float64)
    Aeq = np.zeros((2*n - 1, n*n)); beq = np.zeros(2*n - 1)
    for i in range(n):
        Aeq[i, i*n:(i+1)*n] = 1.0; beq[i] = p[i]
    for j in range(n - 1):
        Aeq[n+j, j::n] = 1.0; beq[n+j] = q[j]
    res = linprog(C_mat.flatten(), A_eq=Aeq, b_eq=beq,
                  bounds=[(0,None)]*(n*n), method='highs')
    return float(res.fun) if res.success else float('nan')

# ─── 6. Per-point worker ─────────────────────────────────────────────────────
def _compute_point(args):
    """Compute all quantities for one (N, uv) pair. Called in worker process."""
    N, uv = args
    U  = uv * V_HYB
    dU = max(FD_REL * U, FD_ABS) * V_HYB

    E0,   psi,  basis = exact_gs(N, U)
    Ep,   _,    _     = exact_gs(N, U + dU)
    Em,   _,    _     = exact_gs(N, U - dU)

    E0_sa = run_sa(N, U)

    return {
        'uv':      float(uv),
        'E0':      E0,
        'E0_sa':   E0_sa,
        'error':   abs(E0_sa - E0),
        'docc':    double_occupancy(psi, basis),
        'docc_fd': (Ep - Em) / (2 * dU),
        'd2E0':    (Ep - 2*E0 + Em) / dU**2,
        'spin_corr': spin_corr(psi, basis),
        'psi':     psi,
        'basis':   basis,
    }

# ─── 7. Sweep for one N_bath ─────────────────────────────────────────────────
def run_sweep(N):
    print(f"[N={N}] starting ({len(UV_VALS)} points)", flush=True)

    args = [(N, uv) for uv in UV_VALS]
    point_results = [None] * len(UV_VALS)

    with ProcessPoolExecutor(max_workers=INNER_WORKERS) as pool:
        futures = {pool.submit(_compute_point, a): idx for idx, a in enumerate(args)}
        for fut in as_completed(futures):
            point_results[futures[fut]] = fut.result()

    point_results.sort(key=lambda r: r['uv'])

    # Consecutive-pair observables (serial — require adjacent psi vectors)
    fid_vals  = [1.0]
    dwgs_vals = [0.0]
    for i in range(1, len(point_results)):
        p1  = np.abs(point_results[i-1]['psi'])**2
        p2  = np.abs(point_results[i]['psi'])**2
        bas = point_results[i]['basis']
        fid_vals.append(fidelity(point_results[i-1]['psi'], point_results[i]['psi']))
        dwgs_vals.append(wasserstein_hamming(p1, p2, bas))

    res = {
        'uv':        [r['uv']        for r in point_results],
        'E0_sa':     [r['E0_sa']     for r in point_results],
        'E0_exact':  [r['E0']        for r in point_results],
        'error':     [r['error']     for r in point_results],
        'docc':      [r['docc']      for r in point_results],
        'docc_fd':   [r['docc_fd']   for r in point_results],
        'd2E0':      [r['d2E0']      for r in point_results],
        'spin_corr': [r['spin_corr'] for r in point_results],
        'fidelity':  fid_vals,
        'd_WGS':     dwgs_vals,
    }

    for i, uv in enumerate(res['uv']):
        print(f"  [N={N}] U/V={uv:.2f}  "
              f"E0_sa={res['E0_sa'][i]:.4f}  "
              f"E0_exact={res['E0_exact'][i]:.5f}  "
              f"err={res['error'][i]:.2e}  "
              f"docc={res['docc'][i]:.4f}  "
              f"d2={res['d2E0'][i]:.4f}  "
              f"Scorr={res['spin_corr'][i]:.4f}  "
              f"F={res['fidelity'][i]:.4f}  "
              f"W={res['d_WGS'][i]:.4f}", flush=True)
    return res

# ─── 8. Plotting ──────────────────────────────────────────────────────────────
def plot_all(all_res):
    colors  = {1:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c'}
    markers = {1:'o', 2:'s', 3:'^'}
    panels = [
        ('E0_exact',  r'$E_0$ (exact)',           'Ground-state energy (exact, physical sector)'),
        ('E0_sa',     r'$E_0$ (SA)',               'Ground-state energy (XBK + SA Dinkelbach)'),
        ('error',     r'$|E_0^{SA}-E_0^{exact}|$','SA absolute error'),
        ('docc',      r'$\langle n_\uparrow n_\downarrow\rangle$', 'Double occupancy'),
        ('docc_fd',   r'$dE_0/dU$',               r'$dE_0/dU$ (finite difference)'),
        ('d2E0',      r'$d^2E_0/dU^2$',           r'$d^2E_0/dU^2$ (Kondo crossover scale)'),
        ('spin_corr', r'$\langle S_z^{\rm imp}S_z^{\rm bath}\rangle$',
                                                   'Impurity–bath spin correlation'),
        ('fidelity',  r'$|\langle\psi_k|\psi_{k+1}\rangle|^2$',
                                                   'Quantum fidelity (consecutive $U/V$)'),
        ('d_WGS',     r'$W_1$ (Wasserstein, Hamming)',
                                                   r'$d_{\rm WGS}$ (consecutive $U/V$)'),
    ]
    for key, ylabel, title in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        for N, r in all_res.items():
            ax.plot(r['uv'], r[key], color=colors[N], marker=markers[N],
                    markersize=4, linewidth=1.5, label=f'$N_{{\\rm bath}}={N}$')
        ax.set_xlabel(r'$U/V$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'{key}.pdf'), dpi=150)
        plt.close(fig)
        print(f"  Saved {key}.pdf")

# ─── 9. CSV ───────────────────────────────────────────────────────────────────
def save_csv(N, res):
    cols = ['uv','E0_sa','E0_exact','error','docc','docc_fd',
            'd2E0','spin_corr','fidelity','d_WGS']
    with open(os.path.join(OUT, f'siam_N{N}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(len(res['uv'])):
            w.writerow([res[c][i] for c in cols])
    print(f"  Saved siam_N{N}.csv")

# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    t0 = time.time()

    all_res = {}
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as pool:
        futures = {pool.submit(run_sweep, N): N for N in N_BATH_LIST}
        for fut in as_completed(futures):
            all_res[futures[fut]] = fut.result()

    for N in N_BATH_LIST:
        save_csv(N, all_res[N])
    plot_all(all_res)

    print(f"\nFinished in {time.time()-t0:.1f}s.  Outputs in {OUT}/")
