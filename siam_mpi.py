"""
siam_mpi.py — MPI-parallel XBK ground-state sweep for the SIAM on Kamet HPC.

Distributes the entire (U, N_bath, sector, p) task cross-product across MPI
ranks (one rank per core).  Each task is one independent XBK / Dinkelbach / SA
solve; exact diagonalisation references are distributed as tasks too, so rank 0
is never a bottleneck.  Results are gathered on rank 0, reduced by min over
(sector, p) for each (U, N_bath), and written to CSV + PNG.

Parallelism model (measured on an 8-core laptop: 24 tasks / 267 s, slowest
single task ~233 s).  Wall-clock ~= max(task_seconds, total_task_seconds /
ncores).  With ~360 tasks for the full r=2 sweep, 192 Kamet cores finish in
roughly the time of the single slowest task (~N_bath=5).

Run (under SLURM, see siam_kamet.sh):
    mpirun python siam_mpi.py --baths 2 3 4 5 --U 0 2 4 6 8 --r 2

Single-process sanity check / dry run (login node):
    python siam_mpi.py --baths 2 --U 4 --r 2 --dry-run
"""
import argparse
import glob
import math
import os
import sys
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
# Task construction — every rank builds the identical list, then strides over it
# ─────────────────────────────────────────────────────────────────────────────

def build_tasks(baths, U_values, r, seed=3, penalty=12.0, want_exact=True):
    """Flatten (U, N_bath, sector, p) into one task list, plus exact-ED tasks.

    Task tuple formats:
        ('xbk',   U, L, si, p, terms, n, r, seed, penalty, const)
        ('exact', U, L)
    Building this is cheap (BK + tapering on small operators), so it is safe to
    do redundantly on every rank rather than scatter large polynomials.
    """
    tasks = []
    for U in U_values:
        for L in baths:
            if want_exact:
                tasks.append(('exact', U, L))
            H_bk = xp.bk_transform(xp.siam(L, U=U))
            for si, sec in enumerate(xp.z2_taper(H_bk)):
                terms, n = xp._qubitop_to_terms(sec)
                if n == 0:
                    const = float(np.real(sec.terms.get((), 0.0)))
                    tasks.append(('xbk', U, L, si, 0, (), 0, r, 0, penalty, const))
                    continue
                for p in range(math.ceil(r / 2) + 1):
                    tasks.append(('xbk', U, L, si, p, terms, n, r,
                                  seed + si * 101 + p * 37, penalty, 0.0))
    return tasks


# SA budget for the inner Dinkelbach solve, set from CLI in main(). None -> the
# heavy adaptive budget (150 sweeps/spin, 2 restarts/spin); a fixed light budget
# (e.g. 4 restarts, 600 sweeps) is far cheaper and still a valid upper bound,
# which is what lets the N_bath=4,5 tasks finish in the wall-clock limit.
_SA_RESTARTS = None
_SA_SWEEPS = None


def solve(task):
    """Execute one task. Returns (kind, (U, L), value)."""
    kind = task[0]
    if kind == 'exact':
        _, U, L = task
        E = xp.exact_ground_energy(xp.siam(L, U=U))
        return ('exact', (U, L), E)

    _, U, L, si, p, terms, n, r, seed, penalty, const = task
    if n == 0:
        return ('xbk', (U, L), const)
    H_is = xp.build_xbk_ising(terms, n, r, p)
    C_is = xp.build_counting(n, r, p)
    if not C_is:
        return ('xbk', (U, L), math.inf)
    E0 = xp.dinkelbach(H_is, C_is, n * r, penalty, seed, quadratic=False,
                       n_restarts=_SA_RESTARTS, n_sweeps=_SA_SWEEPS)
    return ('xbk', (U, L), E0)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def reduce_results(gathered, U_values, baths):
    best_xbk = defaultdict(lambda: math.inf)
    exact = {}
    for kind, key, val in gathered:
        if kind == 'exact':
            exact[key] = val
        else:
            best_xbk[key] = min(best_xbk[key], val)
    return best_xbk, exact


# ─────────────────────────────────────────────────────────────────────────────
# Incremental checkpointing — every rank appends each finished result to its own
# partial file (flushed immediately).  A wall-clock TIMEOUT then never destroys
# all progress: whatever completed survives on disk and can be assembled with
# --reduce-partials.  (Job 54502 hit the 2 h limit mid-run and lost everything
# because results were only written after the final barrier-gather.)
# ─────────────────────────────────────────────────────────────────────────────

def _partial_path(csv, rank):
    return f"{csv}.partial.rank{rank}"


def _append_partial(fh, result):
    kind, (U, L), val = result
    fh.write(f"{kind},{float(U)},{int(L)},{float(val)!r}\n")
    fh.flush()
    os.fsync(fh.fileno())


def read_partials(csv):
    """Collect all (kind, (U,L), value) results from every partial rank file."""
    out = []
    for path in sorted(glob.glob(f"{csv}.partial.rank*")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                kind, U, L, val = line.split(",", 3)
                out.append((kind, (float(U), int(L)), float(val)))
    return out


def write_csv(path, U_values, baths, best_xbk, exact):
    with open(path, 'w') as fh:
        fh.write("U,N_bath,E0_XBK,E0_exact,abs_err\n")
        for U in U_values:
            for L in baths:
                e = best_xbk.get((U, L), math.inf)
                ex = exact.get((U, L))
                if not math.isfinite(e) and ex is None:
                    continue                       # nothing computed for this cell
                es = '' if not math.isfinite(e) else f"{e:.6f}"
                exs = '' if ex is None else f"{ex:.6f}"
                err = '' if (ex is None or not math.isfinite(e)) \
                    else f"{abs(e-ex):.6f}"
                fh.write(f"{U},{L},{es},{exs},{err}\n")


def save_plot(path, U_values, baths, best_xbk, exact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(U_values)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for color, U in zip(cmap, U_values):
        pts = [(L, best_xbk.get((U, L), math.inf)) for L in baths]
        pts = [(L, e) for L, e in pts if math.isfinite(e)]   # skip missing cells
        if pts:
            ax1.plot([L for L, _ in pts], [e for _, e in pts],
                     '-o', color=color, label=f"U={U}")
        if all((U, L) in exact for L in baths):
            ye = [exact[(U, L)] for L in baths]
            ax1.plot(baths, ye, '--', color=color, alpha=0.6)
            err = [abs(best_xbk[(U, L)] - exact[(U, L)]) for L in baths]
            ax2.plot(baths, err, '-o', color=color, label=f"U={U}")
    ax1.set_xlabel("N_bath"); ax1.set_ylabel(r"$E_0$")
    ax1.set_title("SIAM ground-state energy (solid: XBK, dashed: exact)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("N_bath"); ax2.set_ylabel(r"$|E_0^{XBK}-E_0^{exact}|$")
    ax2.set_title("XBK r=2 error vs bath size"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baths", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--U", type=float, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--penalty", type=float, default=12.0)
    ap.add_argument("--restarts", type=int, default=None,
                    help="SA restarts per Dinkelbach iter (default: adaptive "
                         "heavy budget; set e.g. 4 for a fast confirmation run)")
    ap.add_argument("--sweeps", type=int, default=None,
                    help="SA sweeps per restart (default: adaptive; set e.g. 600)")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--csv", default="siam_xbk_sweep.csv")
    ap.add_argument("--png", default="siam_xbk_sweep.png")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the task list and print counts, then exit")
    ap.add_argument("--reduce-partials", action="store_true",
                    help="assemble final CSV/PNG from existing .partial.rank* "
                         "files (use after a timeout) and exit")
    args = ap.parse_args()

    global _SA_RESTARTS, _SA_SWEEPS
    _SA_RESTARTS, _SA_SWEEPS = args.restarts, args.sweeps

    # Salvage path: rebuild results from whatever ranks wrote before a timeout.
    if args.reduce_partials:
        flat = read_partials(args.csv)
        best_xbk, exact = reduce_results(flat, args.U, args.baths)
        n_xbk = sum(1 for k, *_ in flat if k == 'xbk')
        print(f"[siam_mpi] reduced {len(flat)} partial results "
              f"({n_xbk} xbk) from {args.csv}.partial.rank*", flush=True)
        write_csv(args.csv, args.U, args.baths, best_xbk, exact)
        try:
            save_plot(args.png, args.U, args.baths, best_xbk, exact)
            print(f"[siam_mpi] wrote {args.csv} and {args.png}", flush=True)
        except Exception as exc:
            print(f"[siam_mpi] wrote {args.csv} (plot skipped: {exc})", flush=True)
        return

    if _HAVE_MPI and not args.dry_run:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nranks = comm.Get_size()
    else:
        comm = None
        rank, nranks = 0, 1

    tasks = build_tasks(args.baths, args.U, args.r, seed=args.seed,
                        penalty=args.penalty, want_exact=not args.no_exact)

    if rank == 0:
        nx = sum(1 for t in tasks if t[0] == 'xbk')
        ne = sum(1 for t in tasks if t[0] == 'exact')
        print(f"[siam_mpi] r={args.r}  baths={args.baths}  U={args.U}", flush=True)
        print(f"[siam_mpi] {len(tasks)} tasks ({nx} xbk + {ne} exact) "
              f"across {nranks} rank(s)", flush=True)

    if args.dry_run:
        return

    # static round-robin assignment; deterministic, no scatter of big objects
    my_tasks = [t for i, t in enumerate(tasks) if i % nranks == rank]
    t0 = time.time()
    local = []
    # checkpoint each result to this rank's partial file as soon as it finishes,
    # so a wall-clock timeout never loses completed work (--reduce-partials).
    with open(_partial_path(args.csv, rank), 'w') as pf:
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
    best_xbk, exact = reduce_results(flat, args.U, args.baths)
    wall = time.time() - t0
    print(f"\n[siam_mpi] DONE in {wall:.0f}s on {nranks} ranks\n", flush=True)

    print(f"{'U':>5} {'N_bath':>6} {'E0_XBK':>10} {'E0_exact':>10} {'err':>8}",
          flush=True)
    for U in args.U:
        for L in args.baths:
            e = best_xbk[(U, L)]
            ex = exact.get((U, L))
            err = '-' if ex is None else f"{abs(e-ex):.4f}"
            exs = '-' if ex is None else f"{ex:.4f}"
            print(f"{U:>5} {L:>6} {e:>10.4f} {exs:>10} {err:>8}", flush=True)

    write_csv(args.csv, args.U, args.baths, best_xbk, exact)
    try:
        save_plot(args.png, args.U, args.baths, best_xbk, exact)
        print(f"\n[siam_mpi] wrote {args.csv} and {args.png}", flush=True)
    except Exception as exc:                       # plotting optional on cluster
        print(f"\n[siam_mpi] wrote {args.csv} (plot skipped: {exc})", flush=True)


if __name__ == "__main__":
    main()
