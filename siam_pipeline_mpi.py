"""
siam_pipeline_mpi.py
====================
Parallelised version of siam_pipeline.py using mpi4py + OpenMPI.

DESIGN
------
All computationally expensive work is embarrassingly parallel over
independent (N_bath, U, r, p) task tuples.  The parallelisation is
implemented as a coordinator / worker pattern over MPI ranks:

  Rank 0  – coordinator
    • Builds a flat list of Task dicts.
    • Scatters tasks to all ranks using a dynamic work-queue loop
      (MPI.ANY_SOURCE receive so fast workers get more tasks).
    • Gathers all Result dicts.
    • Performs all plotting and file IO after gather.

  Rank 1..N-1  – workers
    • Wait in a receive loop.
    • Execute exactly one task per message (either XBK or ED).
    • Send back a Result dict.
    • Exit when they receive a SENTINEL message.

WHY THIS DESIGN
---------------
Static scatter (comm.scatter / MPI.Scatter) assigns fixed tasks per
rank and load-balances poorly because XBK tasks have highly variable
runtime (Dinkelbach convergence speed depends on U/t).  The dynamic
queue means fast ranks pick up extra tasks automatically.

TASK TYPES
----------
Each task dict has a 'kind' key:

  kind = 'xbk'
    Keys: N_bath, U, t, V, eps_imp, r, p, num_sweeps, num_reads
    Returns: E0_xbk (float), t_wall (float)

  kind = 'ed'
    Keys: N_bath, U, t, V, eps_imp
    Returns: E0_ed, D_occ, n_imp, evals (for thermal)

ED tasks are cheap (seconds) and XBK tasks are expensive (minutes).
Both are in the same queue so the coordinator does not need to
distinguish them; workers handle both.

USAGE
-----
Local (laptop/workstation):
    mpirun -n 8 python siam_pipeline_mpi.py

HPC (SLURM):
    See siam_job.slurm for the job script.
    sbatch siam_job.slurm

OUTPUT
------
All figures saved by rank 0 to ./siam_output/:
    siam_figA_ground_energy.png
    siam_figB_curvature.png
    siam_figC_replication.png
    siam_figD_thermal.png
    siam_figEF_impurity.png
    siam_table.png
    siam_results.npz   (raw results for further analysis)

DEPENDENCIES
------------
    mpi4py, openfermion, numpy, scipy, matplotlib, dimod, neal
    playground.py, Models.py  (same directory)
"""

import os
import sys
import time
import warnings
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import prod

import openfermion as op
from openfermion import bravyi_kitaev, get_sparse_operator, FermionOperator
import dimod
from neal import SimulatedAnnealingSampler
from mpi4py import MPI

from Models import build_siam_hamiltonian
from playground import XBK, construct, convert_to_dict, reduction

warnings.filterwarnings('ignore')

# ── MPI setup ────────────────────────────────────────────────────────────────
COMM  = MPI.COMM_WORLD
RANK  = COMM.Get_rank()
NPROC = COMM.Get_size()

# Special tag values used in the dynamic work queue
TAG_TASK     = 1   # coordinator → worker: here is a task
TAG_RESULT   = 2   # worker → coordinator: here is a result
TAG_SENTINEL = 3   # coordinator → worker: no more tasks, exit

# Output directory (created by rank 0)
OUTDIR = 'siam_output'


# ═══════════════════════════════════════════════════════════════════════════
# 0. LOGGING HELPER
# ═══════════════════════════════════════════════════════════════════════════


import multiprocessing as mp
import os

def _sa_worker(args):
    """
    Run a single SA call with a subset of reads.
    Returns (best_energy, best_sample_dict).
    Each call uses a different random seed so results are independent.
    """
    bqm, num_sweeps, num_reads, seed = args
    from neal import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    ss = sampler.sample(bqm,
                        num_sweeps=num_sweeps,
                        num_reads=num_reads,
                        seed=seed)
    return ss.first.energy, dict(ss.first.sample)


def sa_parallel(bqm, num_sweeps, num_reads, n_threads=None):
    """
    Run num_reads independent SA reads distributed across n_threads processes.

    Mathematically this is identical to running num_reads reads sequentially:
    both return the minimum energy found across num_reads independent Markov
    chains started from different random bitstrings.

    The parallel version is faster by a factor of ~n_threads.

    Parameters
    ----------
    bqm        : dimod BQM
    num_sweeps : int   sweeps per read
    num_reads  : int   total independent restarts
    n_threads  : int   number of parallel processes
                       defaults to os.cpu_count() (respects SLURM_CPUS_PER_TASK)

    Returns
    -------
    best_energy : float
    best_sample : dict
    """
    if n_threads is None:
        # SLURM sets SLURM_CPUS_PER_TASK; fall back to os.cpu_count()
        n_threads = int(os.environ.get('SLURM_CPUS_PER_TASK',
                                        os.cpu_count() or 1))

    # Divide reads as evenly as possible across threads
    # e.g. 100 reads, 4 threads → [25, 25, 25, 25]
    base  = num_reads // n_threads
    extra = num_reads %  n_threads
    split = [base + (1 if i < extra else 0) for i in range(n_threads)]

    # Each thread gets a different seed so they explore different regions
    import random
    base_seed = random.randint(0, 2**31)
    args = [
        (bqm, num_sweeps, split[i], base_seed + i)
        for i in range(n_threads)
        if split[i] > 0
    ]

    if n_threads == 1 or len(args) == 1:
        # No overhead from spawning processes for a single thread
        e, s = _sa_worker(args[0])
        return e, s

    with mp.Pool(processes=n_threads) as pool:
        results = pool.map(_sa_worker, args)

    # Return the best result across all threads
    best_energy, best_sample = min(results, key=lambda x: x[0])
    return best_energy, best_sample
def log(msg):
    """Print a timestamped message prefixed with the MPI rank."""
    t = time.strftime('%H:%M:%S')
    print(f"[{t}] rank {RANK:3d}: {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TASK EXECUTORS  (run on any rank)
# ═══════════════════════════════════════════════════════════════════════════

def run_ed_task(task):
    """
    Execute a single ED task.

    Computes for one (N_bath, U) point:
      - E0_ed  : ground state energy via sparse eigenvalue
      - D_occ  : impurity double occupancy <n_up n_down>
      - n_imp  : total impurity occupancy <n_up + n_down>
      - evals  : full spectrum (for thermal observables)

    Parameters
    ----------
    task : dict  keys N_bath, U, t, V, eps_imp

    Returns
    -------
    result : dict  same keys plus E0_ed, D_occ, n_imp, evals
    """
    N    = task['N_bath']
    U    = task['U']
    t    = task['t']
    V    = task['V']
    eps  = task['eps_imp'] if task['eps_imp'] is not None else -U / 2.0

    H = build_siam_hamiltonian(N=N, U=U, t=t, V=V, eps_imp=eps)
    H_sp = get_sparse_operator(H)
    dim  = H_sp.shape[0]
    n_q  = int(np.log2(dim))

    # Dense eigensolution for small systems, sparse for larger
    if dim <= 256:
        evals_full, evecs = np.linalg.eigh(H_sp.toarray())
        gs = evecs[:, 0]
        evals = evals_full
    else:
        k = min(30, dim - 2)
        evals_k, evecs = spla.eigsh(H_sp, k=k, which='SA')
        idx = np.argsort(evals_k)
        evals_k = evals_k[idx]
        evecs   = evecs[:, idx]
        gs      = evecs[:, 0]
        evals   = evals_k

    E0_ed = float(evals[0])

    # Double occupancy  D = <n_up n_down>
    D_op = FermionOperator('0^ 0 1^ 1', 1.0)
    D_sp = get_sparse_operator(D_op, n_qubits=n_q)
    D_occ = float(np.real(gs.conj() @ (D_sp @ gs)))

    # Total impurity occupancy  <n_up + n_down>
    n_op = FermionOperator('0^ 0', 1.0) + FermionOperator('1^ 1', 1.0)
    n_sp = get_sparse_operator(n_op, n_qubits=n_q)
    n_imp = float(np.real(gs.conj() @ (n_sp @ gs)))

    result = dict(task)
    result.update({
        'E0_ed' : E0_ed,
        'D_occ' : D_occ,
        'n_imp' : n_imp,
        'evals' : evals.tolist(),   # list so it pickle-serialises cleanly
    })
    return result


def run_xbk_task(task):
    """
    Execute a single XBK task: one (N_bath, U, r, p) point.

    Implements the Dinkelbach iteration:
        phi_p(lambda) = min_s [ H'_p(s) - lambda * C_p(s) ] = 0

    The root lambda = E0_p is approached monotonically from above.

    Parameters
    ----------
    task : dict  keys N_bath, U, t, V, eps_imp, r, p, num_sweeps, num_reads

    Returns
    -------
    result : dict  same keys plus E0_xbk (float), t_wall (float)
    """
    N         = task['N_bath']
    U         = task['U']
    t         = task['t']
    V         = task['V']
    eps       = task['eps_imp'] if task['eps_imp'] is not None else -U / 2.0
    r         = task['r']
    p         = task['p']
    num_sw    = task['num_sweeps']
    num_rd    = task['num_reads']

    t0 = time.time()

    H      = build_siam_hamiltonian(N=N, U=U, t=t, V=V, eps_imp=eps)
    Q      = bravyi_kitaev(H)
    n      = op.count_qubits(Q)
    Q_diag = XBK(Q, n, r, p)
    C      = construct(n, r, p)

    lam     = 100.0
    sampler = SimulatedAnnealingSampler()

    for _ in range(50):
        H_shift   = Q_diag - lam * C
        H_shifted = convert_to_dict(H_shift)
        bqm       = reduction(H_shifted)
        gse, gs = sa_parallel(bqm, num_sw, num_rd)
        if gse >= -1e-8:
            break
        C_dict = convert_to_dict(C)
        bm_2  = sum(
            coeff * (prod(gs[q] for q in key) if key else 1)
            for key, coeff in C_dict.items()
        )
        if abs(bm_2) < 1e-10:
            break
        lam = lam + gse / bm_2

    t_wall = time.time() - t0

    result = dict(task)
    result.update({'E0_xbk': lam, 't_wall': t_wall})
    return result


def execute_task(task):
    """
    Dispatch a task to the appropriate executor based on task['kind'].

    Parameters
    ----------
    task : dict

    Returns
    -------
    result : dict
    """
    if task['kind'] == 'ed':
        return run_ed_task(task)
    elif task['kind'] == 'xbk':
        return run_xbk_task(task)
    else:
        raise ValueError(f"Unknown task kind: {task['kind']}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. MPI COORDINATOR (rank 0)
# ═══════════════════════════════════════════════════════════════════════════

def coordinator(task_list):
    """
    Dynamic work queue.  Sends tasks to workers one at a time;
    whenever a worker finishes and sends back a result, the coordinator
    sends it the next pending task.  When the queue is exhausted, sends
    SENTINEL to each worker.

    This is the standard MPI master/worker pattern — it adapts
    automatically to heterogeneous task runtimes and heterogeneous
    hardware (some nodes faster than others).

    Parameters
    ----------
    task_list : list of dicts

    Returns
    -------
    results : list of dicts  (one per task, order may differ from task_list)
    """
    n_workers = NPROC - 1
    if n_workers == 0:
        # Only one process: run everything sequentially on rank 0
        log("Only 1 MPI rank — running sequentially on rank 0")
        results = []
        for i, task in enumerate(task_list):
            log(f"  task {i+1}/{len(task_list)}: {task['kind']} "
                f"N={task['N_bath']} U={task['U']:.2f} "
                + (f"r={task['r']} p={task['p']}" if 'r' in task else ''))
            results.append(execute_task(task))
        return results

    pending  = list(task_list)   # tasks not yet sent
    results  = []
    in_flight = 0                # number of tasks currently with workers

    # Seed each worker with its first task
    for w in range(1, NPROC):
        if pending:
            task = pending.pop(0)
            COMM.send(task, dest=w, tag=TAG_TASK)
            in_flight += 1
            log(f"  sent task to worker {w}: "
                f"{task['kind']} N={task['N_bath']} U={task['U']:.2f}"
                + (f" r={task['r']} p={task['p']}" if 'r' in task else ''))

    # Dynamic work queue: listen for results, send next task
    while in_flight > 0:
        status = MPI.Status()
        result = COMM.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT,
                           status=status)
        worker = status.Get_source()
        results.append(result)
        in_flight -= 1
        log(f"  result from worker {worker} "
            f"({len(results)}/{len(task_list)} done)")

        if pending:
            task = pending.pop(0)
            COMM.send(task, dest=worker, tag=TAG_TASK)
            in_flight += 1

    # Send SENTINEL to all workers
    for w in range(1, NPROC):
        COMM.send(None, dest=w, tag=TAG_SENTINEL)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. MPI WORKER (ranks 1..N-1)
# ═══════════════════════════════════════════════════════════════════════════

def worker():
    """
    Worker event loop.  Waits for tasks from rank 0; executes them;
    sends results back.  Exits cleanly on SENTINEL.
    """
    while True:
        status = MPI.Status()
        task = COMM.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag  = status.Get_tag()

        if tag == TAG_SENTINEL:
            log("received SENTINEL, exiting")
            return

        log(f"executing {task['kind']} N={task['N_bath']} U={task['U']:.2f}"
            + (f" r={task['r']} p={task['p']}" if 'r' in task else ''))

        result = execute_task(task)
        COMM.send(result, dest=0, tag=TAG_RESULT)


# ═══════════════════════════════════════════════════════════════════════════
# 4. TASK BUILDER (run on rank 0 only)
# ═══════════════════════════════════════════════════════════════════════════

def build_task_list(N_bath_values, U_values, r_values,
                    t=1.0, V=1.0, eps_imp=None,
                    num_sweeps=5000, num_reads=100):
    """
    Build the complete flat list of tasks for the coordinator.

    Each task is a plain dict so it is pickle-serialisable by mpi4py.

    ED tasks (one per (N_bath, U)):
        Computes E0, D_occ, n_imp, evals.

    XBK tasks (one per (N_bath, U, r, p)):
        Computes E0_xbk for sector p with replication factor r.
        p ranges over 0..r (all sectors), exactly as in the paper.

    Parameters
    ----------
    N_bath_values : list of int
    U_values      : list of float
    r_values      : list of int  e.g. [2, 3, 4]
    t, V          : float
    eps_imp       : float or None  (None → -U/2 per point)
    num_sweeps    : int
    num_reads     : int

    Returns
    -------
    tasks : list of dicts
    """
    tasks = []

    # ED tasks — one per (N_bath, U)
    for N in N_bath_values:
        for U in U_values:
            tasks.append({
                'kind'    : 'ed',
                'N_bath'  : N,
                'U'       : float(U),
                't'       : t,
                'V'       : V,
                'eps_imp' : eps_imp,
            })

    # XBK tasks — one per (N_bath, U, r, p)
    for N in N_bath_values:
        for U in U_values:
            for r in r_values:
                for p in range(r + 1):
                    tasks.append({
                        'kind'      : 'xbk',
                        'N_bath'    : N,
                        'U'         : float(U),
                        't'         : t,
                        'V'         : V,
                        'eps_imp'   : eps_imp,
                        'r'         : r,
                        'p'         : p,
                        'num_sweeps': num_sweeps,
                        'num_reads' : num_reads,
                    })

    return tasks


# ═══════════════════════════════════════════════════════════════════════════
# 5. RESULT ASSEMBLER (rank 0 only)
# ═══════════════════════════════════════════════════════════════════════════

def assemble_results(raw_results, N_bath_values, U_values, r_values):
    """
    Assemble flat result list into structured numpy arrays.

    For each (N_bath, r), assembles:
        E0_ed[U_idx]    – ED ground energy
        E0_xbk[U_idx]   – best XBK energy (min over p)
        D_occ[U_idx]    – double occupancy (ED)
        n_imp[U_idx]    – impurity occupancy (ED)
        t_wall[U_idx]   – total XBK wall-clock time (sum over p)
        evals           – spectrum at each U (list of arrays, for thermal)

    Parameters
    ----------
    raw_results   : list of dicts
    N_bath_values : list of int
    U_values      : list of float
    r_values      : list of int

    Returns
    -------
    assembled : dict  keyed by (N_bath, r) with structured arrays
    """
    n_U = len(U_values)
    U_arr = np.array(U_values)

    assembled = {}

    # Index ED results by (N_bath, U_idx)
    ed_map = {}
    for res in raw_results:
        if res['kind'] != 'ed':
            continue
        N   = res['N_bath']
        u_i = int(np.argmin(np.abs(U_arr - res['U'])))
        ed_map[(N, u_i)] = res

    # For each (N_bath, r), gather XBK results by (U_idx, p)
    for N in N_bath_values:
        # ED arrays (same for all r)
        E0_ed   = np.full(n_U, np.nan)
        D_occ   = np.full(n_U, np.nan)
        n_imp   = np.full(n_U, np.nan)
        evals_list = [None] * n_U

        for u_i in range(n_U):
            key = (N, u_i)
            if key in ed_map:
                r_ed = ed_map[key]
                E0_ed[u_i]      = r_ed['E0_ed']
                D_occ[u_i]      = r_ed['D_occ']
                n_imp[u_i]      = r_ed['n_imp']
                evals_list[u_i] = np.array(r_ed['evals'])

        for r in r_values:
            E0_xbk  = np.full(n_U, np.nan)
            t_wall  = np.zeros(n_U)

            # For each (U, p), collect XBK energies; keep min over p
            xbk_by_u_p = {}
            for res in raw_results:
                if res['kind'] != 'xbk':
                    continue
                if res['N_bath'] != N or res['r'] != r:
                    continue
                u_i = int(np.argmin(np.abs(U_arr - res['U'])))
                p   = res['p']
                xbk_by_u_p[(u_i, p)] = res

            for u_i in range(n_U):
                energies = []
                for p in range(r + 1):
                    key = (u_i, p)
                    if key in xbk_by_u_p:
                        energies.append(xbk_by_u_p[key]['E0_xbk'])
                        t_wall[u_i] += xbk_by_u_p[key]['t_wall']
                if energies:
                    E0_xbk[u_i] = min(energies)

            assembled[(N, r)] = {
                'U_values'  : U_arr,
                'E0_ed'     : E0_ed,
                'E0_xbk'    : E0_xbk,
                'D_occ'     : D_occ,
                'n_imp'     : n_imp,
                't_wall'    : t_wall,
                'evals_list': evals_list,
                'N_bath'    : N,
                'r'         : r,
            }

    return assembled


# ═══════════════════════════════════════════════════════════════════════════
# 6. ANALYSIS FUNCTIONS (rank 0 only)
# ═══════════════════════════════════════════════════════════════════════════

def second_derivative(x, y):
    """
    Central-difference second derivative d^2y/dx^2.

    For interior points i = 1..n-2:
        y''_i = 2*(y_{i+1}/h2 - y_i*(1/h1+1/h2) + y_{i-1}/h1) / (h1+h2)
    where h1 = x_i - x_{i-1}, h2 = x_{i+1} - x_i.
    Endpoints are filled by nearest interior value.
    """
    n  = len(x)
    d2 = np.zeros(n)
    for i in range(1, n - 1):
        h1 = x[i]   - x[i-1]
        h2 = x[i+1] - x[i]
        d2[i] = 2*(y[i+1]/h2 - y[i]*(1/h1 + 1/h2) + y[i-1]/h1) / (h1 + h2)
    d2[0]   = d2[1]
    d2[n-1] = d2[n-2]
    return d2


def smooth(y, window=5):
    """Moving-average smoother. window must be odd."""
    pad    = window // 2
    y_pad  = np.pad(y, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode='valid')


def thermal_observables(evals, T_values):
    """
    Compute Binder cumulant U4, susceptibility chi, heat capacity Cv
    from a discrete spectrum.

    Z(T) = sum_i exp(-E_i/T)
    <A>  = sum_i A_i exp(-E_i/T) / Z(T)
    Cv   = Var(E) / T^2
    chi  = Var(E) / T        (energy fluctuation susceptibility)
    U4   = 1 - <m^4> / (3 <m^2>^2)  where m_i = E_i - <E>
    """
    E   = np.array(evals, dtype=float)
    E  -= E[0]   # shift for numerical stability

    Cv_arr = np.zeros(len(T_values))
    chi_arr = np.zeros(len(T_values))
    U4_arr  = np.zeros(len(T_values))
    Em_arr  = np.zeros(len(T_values))

    for ti, T in enumerate(T_values):
        w   = np.exp(-E / T)
        Z   = w.sum()
        w  /= Z
        Em  = float(w @ E)
        E2  = float(w @ E**2)
        m   = E - Em
        m2  = float(w @ m**2)
        m4  = float(w @ m**4)
        Cv_arr[ti]  = (E2 - Em**2) / T**2
        chi_arr[ti] = m2 / T
        U4_arr[ti]  = 1.0 - m4 / (3.0 * m2**2 + 1e-30)
        Em_arr[ti]  = Em + evals[0]

    return {'T': T_values, 'Cv': Cv_arr, 'chi': chi_arr,
            'U4': U4_arr, 'E_mean': Em_arr}


# ═══════════════════════════════════════════════════════════════════════════
# 7. PLOTTING FUNCTIONS (rank 0 only)
# ═══════════════════════════════════════════════════════════════════════════

def _savefig(name):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"saved {path}")


def plot_ground_energy(assembled, N_bath_values, r_main):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_bath_values)))
    for N, col in zip(N_bath_values, colors):
        sw = assembled[(N, r_main)]
        U  = sw['U_values']
        ax.plot(U, sw['E0_ed'],  '-',  color=col, lw=2,
                label=f'ED N={N}')
        ax.plot(U, sw['E0_xbk'], 'o',  color=col, ms=5,
                label=f'XBK r={r_main} N={N}')
    ax.set_xlabel('U / t'); ax.set_ylabel('$E_0$')
    ax.set_title('SIAM: Ground state energy vs U/t')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig('siam_figA_ground_energy.png')


def plot_curvature(assembled, N_bath_values, r_main):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_bath_values)))
    for N, col in zip(N_bath_values, colors):
        sw = assembled[(N, r_main)]
        U  = sw['U_values']
        d2_ed  = second_derivative(U, sw['E0_ed'])
        d2_xbk = second_derivative(U, sw['E0_xbk'])
        axes[0].plot(U, d2_ed,  '-',  color=col, lw=2,  label=f'ED N={N}')
        axes[0].plot(U, d2_xbk, '--', color=col, lw=1.5,label=f'XBK N={N}')
        axes[1].plot(U, smooth(d2_ed),  '-',  color=col, lw=2,  label=f'ED N={N}')
        axes[1].plot(U, smooth(d2_xbk), '--', color=col, lw=1.5,label=f'XBK N={N}')
    for ax, title in zip(axes, ["Raw $E_0''$", "Smoothed $E_0''$"]):
        ax.axhline(0, color='grey', lw=0.5, ls='--')
        ax.set_xlabel('U / t'); ax.set_ylabel("$E_0''$")
        ax.set_title(f'SIAM: {title}'); ax.legend(fontsize=7)
    plt.tight_layout()
    _savefig('siam_figB_curvature.png')


def plot_replication_study(assembled, N_study, U_values, r_values):
    U  = np.array(U_values)
    ED = assembled[(N_study, r_values[0])]['E0_ed']
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(r_values)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: energy
    ax = axes[0, 0]
    ax.plot(U, ED, 'k--', lw=2, label='ED')
    for r, col in zip(r_values, colors):
        ax.plot(U, assembled[(N_study, r)]['E0_xbk'], 'o-',
                color=col, ms=4, label=f'r={r}')
    ax.set_xlabel('U / t'); ax.set_ylabel('$E_0$')
    ax.set_title(f'SIAM N_bath={N_study}: Energy vs r')
    ax.legend(fontsize=8)

    # Top-right: absolute error log scale
    ax = axes[0, 1]
    mean_errs = []
    for r, col in zip(r_values, colors):
        err = np.abs(assembled[(N_study, r)]['E0_xbk'] - ED)
        ax.semilogy(U, err + 1e-8, 'o-', color=col, ms=4, label=f'r={r}')
        mean_errs.append(float(err.mean()))
    ax.set_xlabel('U / t'); ax.set_ylabel('$|E_{XBK}-E_{ED}|$')
    ax.set_title('Absolute error (log scale)')
    ax.legend(fontsize=8)

    # Bottom-left: total wall-clock time vs r
    ax = axes[1, 0]
    t_totals = [assembled[(N_study, r)]['t_wall'].sum() for r in r_values]
    ax.bar([str(r) for r in r_values], t_totals,
           color=colors, edgecolor='k')
    ax.set_xlabel('r'); ax.set_ylabel('Total time (s)')
    ax.set_title('Computation time vs r')
    for i, (r, tm) in enumerate(zip(r_values, t_totals)):
        ax.text(i, tm + 0.2, f'{tm:.1f}s', ha='center', fontsize=8)

    # Bottom-right: error reduction %
    ax = axes[1, 1]
    reds, labs = [], []
    for i in range(1, len(r_values)):
        pct = 100*(mean_errs[i-1]-mean_errs[i])/(mean_errs[i-1]+1e-12)
        reds.append(pct); labs.append(f'{r_values[i-1]}→{r_values[i]}')
    bar_c = plt.cm.Greens(np.linspace(0.4, 0.9, len(reds)))
    ax.bar(labs, reds, color=bar_c, edgecolor='k')
    ax.set_xlabel('r step'); ax.set_ylabel('Error reduction (%)')
    ax.set_title('Error reduction per replication step')
    for i, (lb, rd) in enumerate(zip(labs, reds)):
        ax.text(i, max(rd+1,1), f'{rd:.1f}%', ha='center', fontsize=9)

    plt.suptitle('SIAM Replication Factor Study', fontsize=13)
    plt.tight_layout()
    _savefig('siam_figC_replication.png')


def plot_thermal(assembled, N_bath_values, r_main, T_values):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_bath_values)))
    for N, col in zip(N_bath_values, colors):
        sw = assembled[(N, r_main)]
        # Use the spectrum at the mid-point U index (U=4.0 ≈ index 4 for 8-pt grid)
        mid = len(sw['U_values']) // 2
        evals = sw['evals_list'][mid]
        if evals is None:
            continue
        res = thermal_observables(evals, T_values)
        axes[0].plot(res['T'], res['U4'],  '-o', color=col, ms=3, label=f'N={N}')
        axes[1].plot(res['T'], res['chi'], '-o', color=col, ms=3, label=f'N={N}')
        axes[2].plot(res['T'], res['Cv'],  '-o', color=col, ms=3, label=f'N={N}')
    for ax, title, yl in zip(axes,
            ['Binder $U_4$', 'Susceptibility $\\chi$', 'Heat capacity $C_v$'],
            ['$U_4$', '$\\chi$', '$C_v$']):
        ax.set_xlabel('T / t'); ax.set_ylabel(yl)
        ax.set_title(f'SIAM: {title}'); ax.legend(fontsize=8)
    plt.suptitle('SIAM: Thermal observables', fontsize=13)
    plt.tight_layout()
    _savefig('siam_figD_thermal.png')


def plot_impurity_physics(assembled, N_bath_values, r_main):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_bath_values)))
    for N, col in zip(N_bath_values, colors):
        sw = assembled[(N, r_main)]
        U  = sw['U_values']
        axes[0].plot(U, sw['n_imp'], 'o-', color=col, ms=5, lw=2,
                     label=f'N_bath={N}')
        axes[1].plot(U, sw['D_occ'], 'o-', color=col, ms=5, lw=2,
                     label=f'N_bath={N}')
    axes[0].axhline(1.0,  color='k',    ls='--', lw=1, label='PH symmetry')
    axes[0].set_xlabel('U / t'); axes[0].set_ylabel(r'$\langle n_{imp}\rangle$')
    axes[0].set_title('SIAM: Impurity occupancy'); axes[0].set_ylim([0, 2])
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.25, color='k',    ls='--', lw=1, label='U=0 limit')
    axes[1].axhline(0.0,  color='grey', ls=':',  lw=1, label='U→∞ limit')
    axes[1].set_xlabel('U / t'); axes[1].set_ylabel(r'$\langle n_\uparrow n_\downarrow\rangle$')
    axes[1].set_title('SIAM: Double occupancy (Kondo diagnostic)')
    axes[1].legend(fontsize=8)
    plt.suptitle('SIAM: Impurity physics vs U/t', fontsize=13)
    plt.tight_layout()
    _savefig('siam_figEF_impurity.png')


def plot_summary_table(assembled, N_bath_values, r_values):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis('off')
    headers = ['N_bath', 'Modes', 'BK qubits', 'XBK(r=2)',
               'Mean |err| r=2', 'Time r=2 (s)']
    rows = []
    for N in N_bath_values:
        H    = build_siam_hamiltonian(N=N, U=4.0, t=1.0)
        Q    = bravyi_kitaev(H)
        n_bk = op.count_qubits(Q)
        sw   = assembled.get((N, 2), None)
        if sw is not None:
            err = float(np.nanmean(np.abs(sw['E0_xbk'] - sw['E0_ed'])))
            t_t = float(sw['t_wall'].sum())
        else:
            err, t_t = float('nan'), float('nan')
        rows.append([str(N), str(2*(N+1)), str(n_bk), str(2*n_bk),
                     f'{err:.4f}', f'{t_t:.1f}'])
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    ax.set_title('SIAM Pipeline Resource Summary', fontsize=12, pad=20)
    plt.tight_layout()
    _savefig('siam_table.png')


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── PARAMETERS (edit here or pass via config file) ───────────────────
    N_BATH_VALUES  = [1, 2, 3]
    U_VALUES       = np.linspace(0.5, 7.5, 8).tolist()
    R_VALUES       = [2, 3, 4]          # replication factors to study
    R_MAIN         = 2                  # primary r for ground/thermal figures
    T_HOP          = 1.0
    V_HYB          = 1.0
    EPS_IMP        = None               # None → particle-hole symmetry -U/2
    NUM_SWEEPS     = 5000
    NUM_READS      = 100
    T_THERMAL      = np.logspace(-1, 1, 40)
    # ─────────────────────────────────────────────────────────────────────

    if RANK == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        log(f"Starting SIAM MPI pipeline on {NPROC} ranks")
        log(f"N_bath={N_BATH_VALUES}  U={U_VALUES}  r={R_VALUES}")
        log(f"Total XBK tasks: "
            f"{len(N_BATH_VALUES) * len(U_VALUES) * sum(r+1 for r in R_VALUES)}")

        # Build task list on coordinator
        tasks = build_task_list(
            N_bath_values = N_BATH_VALUES,
            U_values      = U_VALUES,
            r_values      = R_VALUES,
            t             = T_HOP,
            V             = V_HYB,
            eps_imp       = EPS_IMP,
            num_sweeps    = NUM_SWEEPS,
            num_reads     = NUM_READS,
        )
        log(f"Total tasks queued: {len(tasks)}")

        # Run coordinator (distributes work, collects results)
        raw_results = coordinator(tasks)

        log("All tasks complete.  Assembling results...")
        assembled = assemble_results(raw_results, N_BATH_VALUES,
                                     U_VALUES, R_VALUES)

        log("Generating figures...")
        plot_ground_energy(assembled,   N_BATH_VALUES, R_MAIN)
        plot_curvature(assembled,       N_BATH_VALUES, R_MAIN)
        plot_replication_study(assembled, N_BATH_VALUES[0], U_VALUES, R_VALUES)
        plot_thermal(assembled,         N_BATH_VALUES, R_MAIN, T_THERMAL)
        plot_impurity_physics(assembled, N_BATH_VALUES, R_MAIN)
        plot_summary_table(assembled,   N_BATH_VALUES, R_VALUES)

        # Save raw results for later analysis
        np.savez(os.path.join(OUTDIR, 'siam_results.npz'),
                 allow_pickle=True, assembled=assembled)
        log("Pipeline complete.  All outputs in ./" + OUTDIR)

    else:
        # All non-zero ranks enter the worker loop
        worker()