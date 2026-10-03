# Running the SIAM XBK sweep on Kamet HPC

Confirmation run first: **r = 2**, baths 2–5, U = 0,2,4,6,8.

The sweep is embarrassingly parallel with **two levels** (matching your
`test.slurm` layout):

- **MPI across tasks** — every `(U, N_bath, sector, p)` is an independent XBK /
  Dinkelbach / SA solve, one per MPI rank.
- **multiprocessing across SA restarts within a task** — each rank uses its
  `-c` (`SLURM_CPUS_PER_TASK`) cores to run the SA restarts of its current
  solve in parallel. This stops a single heavy task (SIAM N_bath=5) from
  bottlenecking the pool. `xbk_pipeline.sa_run` reads `SLURM_CPUS_PER_TASK`
  automatically.

`siam.slurm` is configured as `-N 3 -n 48 -c 4` = **192 cores** (kshort max):
48 ranks, 4 SA-restart workers each. Full r=2 sweep ≈ **180 tasks**, expected
wall-clock well under the 2-hour limit set in the script.

You run the `ssh`/`sbatch` commands; I have no cluster access.

---

## Files to copy to Kamet

From this `Pipeline/` directory:

```
xbk_pipeline.py     # pipeline: BK -> Z2 taper -> XBK -> Dinkelbach -> SA (+ siam model)
fieldmath.py        # GF(2) linear algebra used by Z2 tapering (REQUIRED import)
siam_mpi.py         # MPI driver
siam.slurm          # SLURM batch script (same form as test.slurm)
```

## Step 1 — Copy into your working dir

`siam.slurm` uses `-D /home/shassan/XBKReal`. Put the files there (adjust if
your account/dir differ):

```bash
scp xbk_pipeline.py fieldmath.py siam_mpi.py siam.slurm \
    shassan@10.96.72.200:/home/shassan/XBKReal/
```

## Step 2 — Check the Python environment

`siam.slurm` loads the same modules as `test.slurm` — `anaconda/2025.06-0` and
`openmpi/5.0.8` — so the conda env you already use for `pipeline_base.py`
should have everything (this pipeline needs `openfermion numpy scipy mpi4py`;
it does **not** need `neal`). Verify on the login node:

```bash
ssh shassan@10.96.72.200
cd /home/shassan/XBKReal
module purge && module load anaconda/2025.06-0 openmpi/5.0.8
python3 -c "import openfermion, numpy, scipy, mpi4py; print('deps OK')"
python3 siam_mpi.py --baths 2 3 4 5 --U 0 2 4 6 8 --r 2 --dry-run
# expect: "180 tasks (160 xbk + 20 exact)"
```

If `openfermion` is missing from the base env, install it into your env once:
`pip install openfermion` (login node has network). `mpi4py` is already present
(your t-V pipeline uses it).

## Step 3 — Edit the batch script (if needed)

`siam.slurm` already has `--mail-user=nishithr@imsc.res.in` and
`-D /home/shassan/XBKReal`. Change them only if your account differs. To run a
different scope without editing the file, use env overrides (see Knobs).

## Step 4 — Submit

```bash
sbatch siam.slurm
squeue -u $USER                 # ST: R=running, PD=pending
tail -f Out_<jobid>.log         # live progress (rank 0 prints every ~10%)
```

## Step 5 — Collect results

In `/home/shassan/XBKReal/`:

```
siam_xbk_sweep_job<JOBID>.csv   # U, N_bath, E0_XBK, E0_exact, abs_err
siam_xbk_sweep_job<JOBID>.png   # E0 vs N_bath (one curve per U) + error panel
```

Copy back:

```bash
scp shassan@10.96.72.200:/home/shassan/XBKReal/siam_xbk_sweep_job*.{csv,png} .
```

---

## Knobs (env overrides, no file edit needed)

```bash
RVAL=4 sbatch siam.slurm                       # r=4 (higher accuracy, ~260 tasks)
UVALS="4" sbatch siam.slurm                    # single U, just bath scaling
BATHS="2 3 4" UVALS="0 4 8" sbatch siam.slurm  # smaller sweep
```

To change the core layout, edit the `#SBATCH` lines:

| Want | Change |
|------|--------|
| More restart parallelism per task (lowers N=5 floor) | raise `-c` (e.g. `-c 8`), lower `-n` to keep `n*c` ≤ 192 |
| More task parallelism | raise `-n`, lower `-c` |
| r=4 at N_bath=5 (heavy single tasks) | `-p kmedium`, `-t 1-00:00:00`, and a higher `-c` (e.g. 8) |

`-n × -c` must stay ≤ the partition max (192 for kshort).

## Notes / caveats

- **r=2 is a loose but valid bound** (never below the true E0). Confirmed on the
  laptop: e.g. N_bath=2, U=4 → XBK −3.0000 vs exact −3.3764. It is the *true*
  r=2 optimum (brute-force verified), not an SA miss. Use `RVAL=4` for accuracy
  — that's the main reason to be on the cluster.
- **Wall-clock floor** is the single slowest task (N_bath=5, high U); the `-c`
  restart parallelism shrinks it by up to `-c`×.
- **SA budget** is the heavy adaptive setting (`150 sweeps/spin`,
  `2 restarts/spin`), kept as you asked.
- The `--map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core` flags (same as your
  `test.slurm`) give each rank its `-c` cores for the multiprocessing pool.
