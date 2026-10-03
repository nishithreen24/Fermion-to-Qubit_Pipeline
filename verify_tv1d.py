"""
verify_tv1d.py
==============
Standalone verification of the BK->XBK->QUBO->SA pipeline on the
2-site periodic spinless t-V model, using dwave-neal's
SimulatedAnnealingSampler as the inner minimiser.

This script is self-contained — it does NOT import pipeline_base.py.
It is meant to be run directly on the HPC to verify that the pipeline
produces correct ground state energies before relying on qanneal.

Install dependency:
    pip install dwave-neal

Expected results (exact, from Bethe ansatz at L=2 half-filling):
    V/t = 0.0  ->  E0 = -2.0
    V/t = 1.0  ->  E0 = -2.5
    V/t = 2.0  ->  E0 = -3.0
    V/t = 4.0  ->  E0 = -4.0

All results should match to < 1e-10 absolute error.
"""

import sys
import numpy as np
from collections import defaultdict
from itertools import product as iproduct

from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator
from openfermion.utils import count_qubits
# Try to import neal from whichever package provides it.
# On the IMSC cluster, neal may be available as:
#   - standalone:  pip install dwave-neal  -> from neal import ...
#   - via Ocean SDK: pip install dwave-ocean-sdk -> from neal import ...
#   - via dimod:   pip install dwave-samplers -> from dwave.samplers import ...
# This block tries each in order and falls back to a pure-Python SA if none found.
try:
    from neal import SimulatedAnnealingSampler
    _NEAL_SOURCE = "neal"
except ImportError:
    try:
        from dwave.samplers import SimulatedAnnealingSampler
        _NEAL_SOURCE = "dwave.samplers"
    except ImportError:
        SimulatedAnnealingSampler = None
        _NEAL_SOURCE = "fallback"

if SimulatedAnnealingSampler is None:
    # Pure-Python fallback SA using dimod BQM if available, else hand-rolled
    print("WARNING: neither neal nor dwave.samplers found.")
    print("         Falling back to hand-rolled SA.")
    print("         To use neal: pip install dwave-neal")
    import random, math

    class SimulatedAnnealingSampler:
        """Minimal fallback SA sampler with same interface as neal."""
        def sample_ising(self, h, J, num_reads=100, seed=42, **kwargs):
            rng = random.Random(seed)
            n = 0
            if h: n = max(n, max(h.keys()) + 1)
            if J: n = max(n, max(max(e) for e in J.keys()) + 1)
            best_E, best_s = float('inf'), {}

            for _ in range(num_reads):
                s = {i: rng.choice([-1, 1]) for i in range(n)}
                E = sum(h.get(i, 0)*s[i] for i in s) + \
                    sum(v*s[a]*s[b] for (a,b),v in J.items())
                T = 2.0
                for step in range(1000):
                    i = rng.randint(0, n-1)
                    dE = -2*s[i]*(h.get(i,0) + sum(J.get((min(i,j),max(i,j)),0)*s[j]
                                                     for j in s if j != i))
                    if dE < 0 or rng.random() < math.exp(-dE/T):
                        s[i] = -s[i]; E += dE
                    T *= 0.995
                if E < best_E:
                    best_E, best_s = E, dict(s)

            class _Resp:
                class first:
                    sample = best_s
                    energy = best_E
            return _Resp()

print(f"Using SA from: {_NEAL_SOURCE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Hamiltonian
# ─────────────────────────────────────────────────────────────────────────────

def build_tV(t, V, L=2):
    H = FermionOperator()
    for j in range(L):
        jp = (j + 1) % L
        H += FermionOperator(((j, 1), (jp, 0)), -t)
        H += FermionOperator(((jp, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (j, 0), (jp, 1), (jp, 0)), V)
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator(((jp, 1), (jp, 0)), -V / 2)
        H += FermionOperator((), V / 4)
    return H


def exact_E0(H_fermion):
    return float(np.linalg.eigvalsh(get_sparse_operator(H_fermion).toarray())[0])


# ─────────────────────────────────────────────────────────────────────────────
# 2. BK transformation
# ─────────────────────────────────────────────────────────────────────────────

def bk_transform(H_f):
    H_bk = bravyi_kitaev(H_f)
    n = count_qubits(H_bk)
    terms = []
    for pauli_tuple, coeff in H_bk.terms.items():
        c = float(np.real(coeff))
        if abs(c) > 1e-14:
            terms.append((c, {idx: op for idx, op in pauli_tuple}))
    return terms, n


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ising polynomial arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def _poly_mul(p1, p2):
    out = defaultdict(complex)
    for s1, c1 in p1.items():
        for s2, c2 in p2.items():
            out[s1.symmetric_difference(s2)] += c1 * c2
    return dict(out)


def _poly_clean(poly, tol=1e-12):
    return {m: v.real for m, v in poly.items() if abs(v.real) > tol}


# ─────────────────────────────────────────────────────────────────────────────
# 4. XBK
# ─────────────────────────────────────────────────────────────────────────────

def _phys(i, copy):
    return 2 * i + (copy - 1)


def _xbk_single(pauli, qi, j, k, sp):
    pj, pk = _phys(qi, j), _phys(qi, k)
    if j == k:
        if pauli in ('X', 'Y'): return {}
        if pauli == 'Z': return {frozenset([pj]): sp + 0j}
        return {frozenset(): 1.0 + 0j}
    if pauli == 'X':
        return {frozenset(): sp*0.5+0j, frozenset([pj,pk]): -sp*0.5+0j}
    if pauli == 'Y':
        return {frozenset([pk]): 1j*sp*0.5, frozenset([pj]): -1j*sp*0.5}
    if pauli == 'Z':
        return {frozenset([pj]): sp*0.5+0j, frozenset([pk]): sp*0.5+0j}
    return {frozenset(): 0.5+0j, frozenset([pj,pk]): 0.5+0j}


def build_Hprime(bk_terms, n_qubits, sector_signs):
    H_prime = defaultdict(complex)
    for j in range(1, 3):
        for k in range(1, 3):
            for coeff, pd in bk_terms:
                term = {frozenset(): 1.0 + 0j}
                for qi in range(n_qubits):
                    term = _poly_mul(term, _xbk_single(pd.get(qi,'I'), qi, j, k, sector_signs[qi]))
                for mono, val in term.items():
                    H_prime[mono] += coeff * val
    return _poly_clean(dict(H_prime))


def build_counting_op(n_qubits):
    C = defaultdict(complex)
    for basis in iproduct([0,1], repeat=n_qubits):
        inner = defaultdict(complex)
        for copy_j in range(1, 3):
            p = {frozenset(): 1.0 + 0j}
            for k, b_k in enumerate(basis):
                ph = _phys(k, copy_j)
                factor = {frozenset(): 0.5+0j, frozenset([ph]): float((-1)**b_k)*0.5+0j}
                p = _poly_mul(p, factor)
            for m, v in p.items():
                inner[m] += v
        for m, v in _poly_mul(dict(inner), dict(inner)).items():
            C[m] += v
    return _poly_clean(dict(C))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Quadratisation
# ─────────────────────────────────────────────────────────────────────────────

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


def quadratize(binary, M, next_id):
    poly = defaultdict(float, binary)
    changed = True
    while changed:
        changed = False
        for mono in list(poly.keys()):
            if len(mono) < 3: continue
            c = poly.pop(mono, 0.0)
            if abs(c) < 1e-14: continue
            sl = sorted(mono)
            a, b, rest = sl[0], sl[1], frozenset(sl[2:])
            w = next_id; next_id += 1
            poly[frozenset([w]) | rest] += c
            poly[frozenset([a,b])]      += M
            poly[frozenset([w,a])]      += -2.0*M
            poly[frozenset([w,b])]      += -2.0*M
            poly[frozenset([w])]        += 3.0*M
            changed = True; break
    return dict(poly), next_id


# ─────────────────────────────────────────────────────────────────────────────
# 6. Neal-based Dinkelbach
# ─────────────────────────────────────────────────────────────────────────────

def _eval(ising, spins):
    E = 0.0
    for mono, c in ising.items():
        p = 1.0
        for i in mono: p *= spins[i]
        E += c * p
    return E


def _to_bqm(ising, n_spins):
    """Convert Ising polynomial to a dict suitable for neal."""
    h = {}
    J = {}
    offset = 0.0
    for mono, c in ising.items():
        if len(mono) == 0:
            offset += c
        elif len(mono) == 1:
            i = next(iter(mono))
            h[i] = h.get(i, 0.0) + c
        elif len(mono) == 2:
            i, j = sorted(mono)
            J[(i,j)] = J.get((i,j), 0.0) + c
    return h, J, offset


def neal_minimize(ising, n_spins, num_reads=100, seed=42):
    """Minimise the Ising polynomial using dwave-neal."""
    h, J, offset = _to_bqm(ising, n_spins)
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_ising(
        h, J,
        num_reads=num_reads,
        seed=seed,
    )
    best = response.first
    spins = {i: best.sample.get(i, 1) for i in range(n_spins)}
    return spins, best.energy + offset


def dinkelbach_neal(H_ising, C_ising, n_spins, num_reads=100,
                    max_iter=25, tol=1e-6, seed=0):
    lam = 5.0
    best_E0 = float('inf')
    for it in range(max_iter):
        combined = defaultdict(float, H_ising)
        for m, c in C_ising.items():
            combined[m] -= lam * c
        spins, _ = neal_minimize(dict(combined), n_spins,
                                  num_reads=num_reads, seed=seed + it)
        H_val = _eval(H_ising, spins)
        C_val = _eval(C_ising, spins)
        if abs(C_val) < 1e-8: continue
        new_lam = H_val / C_val
        if new_lam < best_E0: best_E0 = new_lam
        if abs(new_lam - lam) < tol: break
        lam = new_lam
    return best_E0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full pipeline for one (t, V)
# ─────────────────────────────────────────────────────────────────────────────

def run_one(t, V, penalty_scale=12.0, num_reads=200, seed=0):
    H_f = build_tV(t, V, L=2)
    bk_terms, n_qubits = bk_transform(H_f)
    n_phys = n_qubits * 2
    M = penalty_scale * max(abs(t), abs(V), 1.0)
    C_ising = build_counting_op(n_qubits)
    best_E0 = float('inf')
    for sec in range(2 ** n_qubits):
        sector_signs = [+1 if (sec >> i) & 1 == 0 else -1 for i in range(n_qubits)]
        H_ising = build_Hprime(bk_terms, n_qubits, sector_signs)
        next_id = n_phys
        H_quad, next_id = quadratize(_ising_to_binary(H_ising), M, next_id)
        C_quad, next_id = quadratize(_ising_to_binary(C_ising),  M, next_id)
        E0 = dinkelbach_neal(_binary_to_ising(H_quad), _binary_to_ising(C_quad),
                              next_id, num_reads=num_reads, seed=seed + sec*37)
        if E0 < best_E0: best_E0 = E0
    return best_E0, exact_E0(H_f)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sweep V/t from 0 to 4
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    t = 1.0
    V_values = np.round(np.arange(0.0, 4.01, 0.2), 6)
    out_dir = "output/tv1d_neal_verify"
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'V/t':>5}  {'E_neal':>12}  {'E_exact':>12}  {'|err|':>10}")
    print("─" * 46)

    E_neal_list  = []
    E_exact_list = []
    errors       = []

    for V in V_values:
        E_sa, E_ex = run_one(t, V, penalty_scale=12.0, num_reads=200, seed=42)
        err = abs(E_sa - E_ex)
        E_neal_list.append(E_sa)
        E_exact_list.append(E_ex)
        errors.append(err)
        print(f"{V/t:>5.2f}  {E_sa:>12.6f}  {E_ex:>12.6f}  {err:>10.2e}")

    # Save CSV
    csv_path = os.path.join(out_dir, "tv1d_neal_verify.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["V_over_t", "E_neal", "E_exact", "abs_error"])
        for V, E_sa, E_ex, err in zip(V_values, E_neal_list, E_exact_list, errors):
            w.writerow([f"{V/t:.2f}", f"{E_sa:.10f}", f"{E_ex:.10f}", f"{err:.4e}"])

    print(f"\nMax |error|:  {max(errors):.2e}")
    print(f"Mean |error|: {sum(errors)/len(errors):.2e}")
    print(f"Results saved to {csv_path}")

    # Plot
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
        "font.size": 12, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05}, sharex=True)

    ax1.plot(V_values, E_exact_list, color="#1a1a2e", lw=2.2,
             label="Exact diagonalisation")
    ax1.scatter(V_values, E_neal_list, color="#e63946", s=55, marker="o",
                edgecolors="white", lw=0.8, zorder=4, label="XBK + neal SA")
    ax1.axvline(2.0, color="#888", ls="--", lw=1.0, alpha=0.7)
    ax1.set_ylabel(r"$E_0$  (units of $t$)", fontsize=13)
    ax1.set_ylim(-4.3, -1.7)
    ax1.legend(fontsize=11, framealpha=0.95)
    ax1.set_title(r"2-site periodic $t$-$V$ model — pipeline verification (neal SA)",
                  fontsize=12)

    safe_err = np.where(np.array(errors) < 1e-15, 1e-16, errors)
    ax2.semilogy(V_values, safe_err, color="#457b9d", lw=1.6,
                 marker="s", ms=5, mfc="white", mew=1.2,
                 label=r"$|E_\mathrm{SA} - E_\mathrm{ED}|$")
    ax2.axhline(1e-10, color="#ccc", ls=":", lw=0.9)
    ax2.set_ylabel(r"$|E_\mathrm{SA} - E_\mathrm{ED}|$", fontsize=12)
    ax2.set_xlabel(r"$V/t$", fontsize=13)
    ax2.set_ylim(1e-15, 1e-8)
    ax2.legend(fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xlim(-0.1, 4.2)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.savefig(os.path.join(out_dir, "tv1d_neal_verify.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved to {out_dir}/tv1d_neal_verify.png")
