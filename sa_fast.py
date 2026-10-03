"""
sa_fast.py — numpy-vectorised simulated annealing + Dinkelbach (NEW FILE).

A drop-in faster alternative to xbk_pipeline.sa_run / .dinkelbach that does NOT
modify xbk_pipeline.py.  Two sources of speedup, both via numpy (no numba, no
new dependency -- works on the cluster's anaconda as-is):

  1. Replica batching: all `n_restarts` SA chains run simultaneously as rows of
     an (R x n_spins) array, so one numpy op advances all restarts at once.
  2. Monomial vectorisation: the per-spin local field (the expensive inner sum
     over the ~|H| 2^n r^2 Ising terms touching a spin) is computed as a single
     batched numpy contraction instead of a Python loop.

The Ising polynomial is the same k-local format used by xbk_pipeline:
    dict { frozenset(spin indices) -> float }.

Identical algorithm (Metropolis, geometric cooling); results match the pure-
Python SA up to RNG.  Use dinkelbach_fast() exactly like xbk_pipeline.dinkelbach
(quadratic=False path).
"""
import math
import numpy as np

from xbk_pipeline import _eval            # reused for the ratio (called O(iters))


# ─────────────────────────────────────────────────────────────────────────────
# Compile the Ising poly into numpy index structures
# ─────────────────────────────────────────────────────────────────────────────

def _compile(ising, n_spins):
    """Return (const, touch, Eidx, Ecf).

    Spin index `n_spins` is a SENTINEL column held at +1, used to pad ragged
    monomials so every row has equal width (padding multiplies by 1).

    touch[i] = (idx (M_i x d_i) int, cf (M_i,) float): for each monomial that
               contains spin i, the OTHER spins in it (padded) and its coeff.
    Eidx/Ecf : all non-constant monomials padded, for full-energy evaluation.
    """
    const = 0.0
    touch_lists = [[] for _ in range(n_spins)]
    nz = []
    for mono, c in ising.items():
        if abs(c) == 0.0:
            continue
        ml = sorted(mono)
        if not ml:
            const += c
            continue
        nz.append((ml, c))
        for i in ml:
            touch_lists[i].append(([j for j in ml if j != i], c))

    def pad(lst):
        if not lst:
            return (np.full((0, 1), n_spins, dtype=np.int64), np.zeros(0))
        d = max(1, max(len(o) for o, _ in lst))
        idx = np.full((len(lst), d), n_spins, dtype=np.int64)
        cf = np.empty(len(lst), dtype=np.float64)
        for r, (o, c) in enumerate(lst):
            idx[r, :len(o)] = o
            cf[r] = c
        return idx, cf

    touch = [pad(touch_lists[i]) for i in range(n_spins)]
    # full-energy structure
    if nz:
        d = max(len(m) for m, _ in nz)
        Eidx = np.full((len(nz), d), n_spins, dtype=np.int64)
        Ecf = np.empty(len(nz), dtype=np.float64)
        for r, (m, c) in enumerate(nz):
            Eidx[r, :len(m)] = m
            Ecf[r] = c
    else:
        Eidx, Ecf = np.full((0, 1), n_spins, dtype=np.int64), np.zeros(0)
    return const, touch, Eidx, Ecf


def _energy(ext, Eidx, Ecf, const):
    """Batched total energy for ext (R x n_spins+1, last col = +1). Returns (R,)."""
    R = ext.shape[0]
    if Eidx.shape[0] == 0:
        return np.full(R, const)
    return ext[:, Eidx].prod(axis=2) @ Ecf + const


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised multi-restart SA
# ─────────────────────────────────────────────────────────────────────────────

def sa_run_fast(ising, n_spins, n_restarts, n_sweeps, T_start, T_end, seed):
    """All n_restarts chains in lockstep via numpy. Returns (best_spins, best_E)
    with best_spins a {spin: +/-1} dict, matching xbk_pipeline.sa_run's output."""
    rng = np.random.default_rng(seed)
    const, touch, Eidx, Ecf = _compile(ising, n_spins)
    R = max(1, n_restarts)

    ext = np.ones((R, n_spins + 1))                    # last col = +1 sentinel
    ext[:, :n_spins] = rng.integers(0, 2, (R, n_spins)) * 2 - 1
    E = _energy(ext, Eidx, Ecf, const)
    best_E = E.copy()
    best_spins = ext[:, :n_spins].copy()

    alpha = (T_end / T_start) ** (1.0 / max(n_sweeps, 1))
    T = T_start
    for _ in range(n_sweeps):
        for i in range(n_spins):                       # systematic sweep
            idx, cf = touch[i]
            if idx.shape[0] == 0:
                continue
            field = ext[:, idx].prod(axis=2) @ cf       # (R,)
            dE = -2.0 * ext[:, i] * field               # (R,)
            # Metropolis: accept if dE<0 or rand < exp(-dE/T); clip avoids overflow
            p = np.exp(np.minimum(-dE / T, 0.0))
            acc = (dE < 0) | (rng.random(R) < p)
            ext[acc, i] *= -1.0
            E[acc] += dE[acc]
            improved = E < best_E
            if improved.any():
                best_E[improved] = E[improved]
                best_spins[improved] = ext[improved, :n_spins]
        T *= alpha

    b = int(np.argmin(best_E))
    return {i: int(best_spins[b, i]) for i in range(n_spins)}, float(best_E[b])


# ─────────────────────────────────────────────────────────────────────────────
# Dinkelbach using the fast SA (k-local, no quadratisation) — mirrors
# xbk_pipeline.dinkelbach(quadratic=False).  Ratio is evaluated on the physical
# polynomials with the physical spins only.
# ─────────────────────────────────────────────────────────────────────────────

def dinkelbach_fast(H_ising, C_ising, n_phys, seed,
                    n_restarts=10, n_sweeps=2000, max_iter=30, tol=1e-6):
    """Minimise H'(s)/C(s) over n_phys physical spins. Returns best E0 (or +inf)."""
    lam = 0.0
    best_E0 = float('inf')
    base = max((abs(c) for c in H_ising.values()), default=1.0) \
        + max((abs(c) for c in C_ising.values()), default=1.0)

    for it in range(max_iter):
        combined = dict(H_ising)
        for m, c in C_ising.items():
            combined[m] = combined.get(m, 0.0) - lam * c
        combined = {m: v for m, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            break

        peak = max(abs(v) for v in combined.values())
        T0 = max(2.0 * max(base, peak), 5.0)
        spins, _ = sa_run_fast(combined, n_phys, n_restarts, n_sweeps,
                               T0, 1e-3, seed + it * 1009)

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
