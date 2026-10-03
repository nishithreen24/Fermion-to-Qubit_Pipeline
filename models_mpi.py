"""
models_mpi.py — general MPI-parallel XBK ground-state sweep for ANY model in
the xbk_pipeline registry (tv1d/tv2d, hubbard1d/2d, siam, pam, kitaev, j1j2_*).

Generalises siam_mpi.py: instead of sweeping (U, N_bath) for the SIAM, it sweeps
a model's natural (ratio, size) grid, where `ratio` is the dimensionless coupling
(V/t, U/t, Delta/t, J2/J1) and `size` is L (chain length / linear lattice dim /
bath count).  Fermionic models go through BK; spin models (j1j2_*) skip it.

Same robustness as siam_mpi.py:
  * one MPI rank per (ratio, size, sector, p) task,
  * per-rank incremental checkpoint files (survive a wall-clock timeout),
  * --reduce-partials to assemble the CSV/PNG after a kill,
  * configurable light SA budget (--restarts/--sweeps).

Run (SLURM, see models.slurm):
    mpirun python models_mpi.py --model kitaev --ratios 0 0.5 1 1.5 2 --L 4 6 8

Dry run / sanity:
    python models_mpi.py --model j1j2_1d --L 4 6 --ratios 0 0.5 1 --dry-run
"""
import argparse
import glob
import math
import os
import time
from collections import defaultdict

import numpy as np

import xbk_pipeline as xp

try:
    from mpi4py import MPI
    _HAVE_MPI = True
except ImportError:
    _HAVE_MPI = False


# ─────────────────────────────────────────────────────────────────────────────
# Model dispatch: (name, ratio, L) -> Hamiltonian.  ratio is the swept coupling;
# t / J1 / V_hyb are fixed at 1, matching the project's model_*.py conventions.
# ─────────────────────────────────────────────────────────────────────────────

def model_hamiltonian(name, ratio, L):
    if name == "tv1d":      return xp.tv1d(L, 1.0, ratio)
    if name == "tv2d":      return xp.tv2d(L, 1.0, ratio)
    if name == "hubbard1d": return xp.hubbard1d(L, 1.0, ratio)
    if name == "hubbard2d": return xp.hubbard2d(L, 1.0, ratio)
    if name == "hubbard2d_2x3": return xp.hubbard2d_rect(2, 3, 1.0, ratio)  # L ignored (fixed 2x3, OBC)
    if name == "siam":      return xp.siam(L, U=ratio)
    if name == "pam":       return xp.pam(L, 1.0, 1.0, ratio)
    if name == "kitaev":    return xp.kitaev(L, 1.0, ratio)
    if name == "j1j2_1d":   return xp.j1j2_1d(L, 1.0, ratio)
    if name == "j1j2_2d":   return xp.j1j2_2d(L, 1.0, ratio)
    raise ValueError(f"unknown model {name!r}")


# sensible default (ratio range, L list, axis label) per model; override on CLI.
DEFAULTS = {
    "tv1d":      (np.round(np.arange(0, 4.01, 0.5), 3), [3, 4],    "V/t"),
    "tv2d":      (np.round(np.arange(0, 4.01, 0.5), 3), [2, 3],    "V/t"),
    "hubbard1d": (np.round(np.arange(0, 8.01, 1.0), 3), [2, 3],    "U/t"),
    "hubbard2d": (np.round(np.arange(0, 8.01, 1.0), 3), [2],       "U/t"),
    "hubbard2d_2x3": (np.round(np.arange(0, 8.01, 1.0), 3), [2],   "U/t"),  # L ignored (fixed 2x3)
    "siam":      (np.round(np.arange(0, 8.01, 1.0), 3), [2, 3, 4], "U/t"),
    "pam":       (np.round(np.arange(0, 8.01, 1.0), 3), [2, 3],    "U/t"),
    "kitaev":    (np.round(np.arange(0, 2.01, 0.25), 3), [4, 6, 8], "Delta/t"),
    "j1j2_1d":   (np.round(np.arange(0, 1.01, 0.1), 3), [4, 6, 8], "J2/J1"),
    "j1j2_2d":   (np.round(np.arange(0, 1.01, 0.2), 3), [2, 3],    "J2/J1"),
}

_MODEL = None                       # set in main(), used by solve() for exact ED
_FAST = False                       # --fast -> numpy-vectorised SA (sa_fast.py)
_SA_RESTARTS = None                 # explicit override (None -> size-adaptive)
_SA_SWEEPS = None


def _budget_for(n, r=2):
    """No-compromise SA budget keyed on the physical spin count m = n*r.

    We do NOT lighten the budget to fit a wall-clock limit (use klong's 7 days
    if a run is slow).  10 restarts to escape local minima, and an anneal length
    that GROWS with the spin count (longer schedule for harder problems):

        restarts = 10
        sweeps   = max(2000, 50 * m)

    Measured single-task times (~15 Dinkelbach iters): kitaev L8 / hubbard2d L2
    at r=6 (m=36) ~6 h; tv2d L3 at r=6 (m=48) ~4 days (-> klong).  Override with
    explicit --restarts/--sweeps for convergence checks (e.g. run B and 2B).
    """
    m = n * r
    return 10, max(2000, 50 * m)


# ─────────────────────────────────────────────────────────────────────────────
# Task construction and solve
# ─────────────────────────────────────────────────────────────────────────────

def build_tasks(name, ratios, Ls, r, seed=3, penalty=12.0, want_exact=True):
    """Flatten (ratio, L, sector, p) into one task list + exact-ED tasks.

    Task tuple formats:
        ('xbk',   ratio, L, si, p, terms, n, r, seed, penalty, const)
        ('exact', ratio, L)
    """
    fermionic = name not in xp.SPIN_MODELS
    tasks = []
    for ratio in ratios:
        for L in Ls:
            if want_exact:
                tasks.append(('exact', ratio, L))
            H = model_hamiltonian(name, ratio, L)
            H_q = xp.bk_transform(H) if fermionic else H
            for si, sec in enumerate(xp.z2_taper(H_q)):
                terms, n = xp._qubitop_to_terms(sec)
                if n == 0:
                    const = float(np.real(sec.terms.get((), 0.0)))
                    tasks.append(('xbk', ratio, L, si, 0, (), 0, r, 0, penalty, const))
                    continue
                for p in range(math.ceil(r / 2) + 1):
                    tasks.append(('xbk', ratio, L, si, p, terms, n, r,
                                  seed + si * 101 + p * 37, penalty, 0.0))
    return tasks


def solve(task):
    """Execute one task. Returns (kind, (ratio, L), value)."""
    kind = task[0]
    if kind == 'exact':
        _, ratio, L = task
        E = xp.exact_ground_energy(model_hamiltonian(_MODEL, ratio, L))
        return ('exact', (ratio, L), E)

    _, ratio, L, si, p, terms, n, r, seed, penalty, const = task
    if n == 0:
        return ('xbk', (ratio, L), const)
    H_is = xp.build_xbk_ising(terms, n, r, p)
    C_is = xp.build_counting(n, r, p)
    if not C_is:
        return ('xbk', (ratio, L), math.inf)
    # budget: explicit --restarts/--sweeps override the size-adaptive default,
    # filling any unset one from _budget_for so both are always concrete ints.
    br, bs = _budget_for(n, r)
    restarts = _SA_RESTARTS if _SA_RESTARTS is not None else br
    sweeps = _SA_SWEEPS if _SA_SWEEPS is not None else bs

    if _FAST:                                          # numpy-vectorised SA
        import sa_fast
        E0 = sa_fast.dinkelbach_fast(H_is, C_is, n * r, seed,
                                     n_restarts=restarts, n_sweeps=sweeps)
    else:
        E0 = xp.dinkelbach(H_is, C_is, n * r, penalty, seed, quadratic=False,
                           n_restarts=restarts, n_sweeps=sweeps)
    return ('xbk', (ratio, L), E0)


# ─────────────────────────────────────────────────────────────────────────────
# Reduce / checkpoint / output  (mirrors siam_mpi.py)
# ─────────────────────────────────────────────────────────────────────────────

def reduce_results(gathered):
    best_xbk = defaultdict(lambda: math.inf)
    exact = {}
    for kind, key, val in gathered:
        if kind == 'exact':
            exact[key] = val
        else:
            best_xbk[key] = min(best_xbk[key], val)
    return best_xbk, exact


def _partial_path(csv, rank):
    return f"{csv}.partial.rank{rank}"


def _append_partial(fh, result):
    kind, (ratio, L), val = result
    fh.write(f"{kind},{float(ratio)},{int(L)},{float(val)!r}\n")
    fh.flush()
    os.fsync(fh.fileno())


def read_partials(csv):
    out = []
    for path in sorted(glob.glob(f"{csv}.partial.rank*")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                kind, ratio, L, val = line.split(",", 3)
                out.append((kind, (float(ratio), int(L)), float(val)))
    return out


def write_csv(path, ratios, Ls, best_xbk, exact, label):
    with open(path, 'w') as fh:
        fh.write(f"ratio,L,E0_XBK,E0_exact,abs_err  # ratio={label}\n")
        for ratio in ratios:
            for L in Ls:
                e = best_xbk.get((ratio, L), math.inf)
                ex = exact.get((ratio, L))
                if not math.isfinite(e) and ex is None:
                    continue
                es = '' if not math.isfinite(e) else f"{e:.6f}"
                exs = '' if ex is None else f"{ex:.6f}"
                err = '' if (ex is None or not math.isfinite(e)) \
                    else f"{abs(e-ex):.6f}"
                fh.write(f"{ratio},{L},{es},{exs},{err}\n")


def save_plot(path, ratios, Ls, best_xbk, exact, label, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(Ls), 1)))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for color, L in zip(cmap, Ls):
        xs = [r for r in ratios if math.isfinite(best_xbk.get((r, L), math.inf))]
        if xs:
            ax.plot(xs, [best_xbk[(r, L)] for r in xs], '-o', color=color,
                    label=f"L={L} (XBK)")
        xe = [r for r in ratios if (r, L) in exact]
        if xe:
            ax.plot(xe, [exact[(r, L)] for r in xe], '--', color=color, alpha=0.6)
    ax.set_xlabel(label)
    ax.set_ylabel(r"$E_0$")
    ax.set_title(f"{model}: XBK (solid) vs exact (dashed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _MODEL, _FAST, _SA_RESTARTS, _SA_SWEEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(xp.MODELS))
    ap.add_argument("--ratios", type=float, nargs="+", default=None,
                    help="coupling values to sweep (default: per-model)")
    ap.add_argument("--L", type=int, nargs="+", default=None,
                    help="system sizes to sweep (default: per-model)")
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--penalty", type=float, default=12.0)
    ap.add_argument("--restarts", type=int, default=None,
                    help="SA restarts (default: size-adaptive per tapered n)")
    ap.add_argument("--sweeps", type=int, default=None,
                    help="SA sweeps (default: size-adaptive per tapered n)")
    ap.add_argument("--fast", action="store_true",
                    help="use the numpy-vectorised SA (sa_fast.py): ~8x faster, "
                         "identical results; lets the n>=8 models reach high r")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reduce-partials", action="store_true")
    args = ap.parse_args()

    _MODEL = args.model
    _FAST = args.fast
    _SA_RESTARTS = args.restarts if args.restarts and args.restarts > 0 else None
    _SA_SWEEPS = args.sweeps if args.sweeps and args.sweeps > 0 else None

    d_ratios, d_Ls, label = DEFAULTS.get(
        args.model, (np.round(np.arange(0, 4.01, 0.5), 3), [2, 3], "ratio"))
    ratios = list(args.ratios) if args.ratios is not None else list(d_ratios)
    Ls = list(args.L) if args.L is not None else list(d_Ls)
    csv = args.csv or f"{args.model}_xbk_sweep.csv"
    png = args.png or f"{args.model}_xbk_sweep.png"

    if args.reduce_partials:
        flat = read_partials(csv)
        best_xbk, exact = reduce_results(flat)
        print(f"[models_mpi] reduced {len(flat)} partial results from "
              f"{csv}.partial.rank*", flush=True)
        write_csv(csv, ratios, Ls, best_xbk, exact, label)
        try:
            save_plot(png, ratios, Ls, best_xbk, exact, label, args.model)
            print(f"[models_mpi] wrote {csv} and {png}", flush=True)
        except Exception as exc:
            print(f"[models_mpi] wrote {csv} (plot skipped: {exc})", flush=True)
        return

    if _HAVE_MPI and not args.dry_run:
        comm = MPI.COMM_WORLD
        rank, nranks = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nranks = None, 0, 1

    tasks = build_tasks(args.model, ratios, Ls, args.r, seed=args.seed,
                        penalty=args.penalty, want_exact=not args.no_exact)
    if rank == 0:
        nx = sum(1 for t in tasks if t[0] == 'xbk')
        bud = "size-adaptive" if _SA_RESTARTS is None and _SA_SWEEPS is None \
            else f"{_SA_RESTARTS}x{_SA_SWEEPS}"
        print(f"[models_mpi] model={args.model}  ratios={ratios}  L={Ls}  "
              f"r={args.r}  budget={bud}  SA={'fast' if _FAST else 'pure-python'}",
              flush=True)
        print(f"[models_mpi] {len(tasks)} tasks ({nx} xbk + {len(tasks)-nx} "
              f"exact) across {nranks} rank(s)", flush=True)
    if args.dry_run:
        return

    my_tasks = [t for i, t in enumerate(tasks) if i % nranks == rank]
    t0 = time.time()
    local = []
    with open(_partial_path(csv, rank), 'w') as pf:
        for j, t in enumerate(my_tasks):
            res = solve(t)
            local.append(res)
            _append_partial(pf, res)
            if rank == 0 and (j + 1) % max(1, len(my_tasks) // 10) == 0:
                print(f"[rank0] {j+1}/{len(my_tasks)} local tasks "
                      f"({time.time()-t0:.0f}s)", flush=True)

    gathered = comm.gather(local, root=0) if comm is not None else [local]
    if rank != 0:
        return

    flat = [item for part in gathered for item in part]
    best_xbk, exact = reduce_results(flat)
    print(f"\n[models_mpi] DONE in {time.time()-t0:.0f}s on {nranks} ranks\n",
          flush=True)
    print(f"{'ratio':>7} {'L':>3} {'E0_XBK':>10} {'E0_exact':>10} {'err':>8}",
          flush=True)
    for ratio in ratios:
        for L in Ls:
            e = best_xbk.get((ratio, L), math.inf)
            ex = exact.get((ratio, L))
            err = '-' if (ex is None or not math.isfinite(e)) else f"{abs(e-ex):.4f}"
            exs = '-' if ex is None else f"{ex:.4f}"
            es = 'inf' if not math.isfinite(e) else f"{e:.4f}"
            print(f"{ratio:>7} {L:>3} {es:>10} {exs:>10} {err:>8}", flush=True)

    write_csv(csv, ratios, Ls, best_xbk, exact, label)
    try:
        save_plot(png, ratios, Ls, best_xbk, exact, label, args.model)
        print(f"\n[models_mpi] wrote {csv} and {png}", flush=True)
    except Exception as exc:
        print(f"\n[models_mpi] wrote {csv} (plot skipped: {exc})", flush=True)


if __name__ == "__main__":
    main()
