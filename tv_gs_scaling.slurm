#!/bin/bash
#SBATCH -J tv_gs
#SBATCH -D /home/shassan/XBKReal
#SBATCH -p kshort
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -c 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nishithr@imsc.res.in
#SBATCH -o logs/tv_gs_%j.log

# Ground-state ENERGIES for the Levy (SQA: SSE + PIMC) and Yip (SA-sampled
# subspace diagonalisation) t-V methods -- the honest solvers that replace the
# ED "cheat" in tv_levy.py / tv_yip.py.  Mirrors tv_scaling.slurm.
#
# ── parameters (override with env vars) ──────────────────────────────────────
LVALS="${LVALS:-3 4 5 6 7 8}"        # validation set; add 10 12 14 16 ... for scaling
VVAL="${VVAL:-2.0}"
TVAL="${TVAL:-1.0}"
BETA="${BETA:-32.0}"                 # large enough that finite-T bias < MC error
MPI="${MPI:-96}"                     # PIMC Trotter steps (dtau = BETA/MPI)
WARMUP="${WARMUP:-5000}"
MEASURES="${MEASURES:-20000}"
SEED="${SEED:-1}"
READS="${READS:-400}"                # Yip: # SA sampling chains
SWEEPS="${SWEEPS:-300}"              # Yip: SA sweeps per chain
MAXSUP="${MAXSUP:-40000}"            # Yip: subspace cap
# ─────────────────────────────────────────────────────────────────────────────

module purge
module load anaconda/2025.06-0 openmpi/5.0.8

CODEDIR="/home/shassan/XBKReal/code"       # tv_gs_common.py, tv_levy_sqa.py, tv_yip_sa.py live here
OUTDIR="/home/shassan/XBKReal/results"
JOBID=$SLURM_JOB_ID

mkdir -p "$OUTDIR" logs
echo "tv_gs job $JOBID  L=$LVALS  V=$VVAL  beta=$BETA  M=$MPI"

# tv_*_sqa.py import tv_gs_common as a sibling module -> run from CODEDIR
cd "$CODEDIR"

pids=()
for L in $LVALS; do
    # Levy: SSE (no Trotter error) + PIMC (Trotter world-line), both estimators
    python3 "$CODEDIR/tv_levy_sqa.py" \
        --L $L --t $TVAL --V $VVAL --beta $BETA --M $MPI \
        --warmup $WARMUP --measures $MEASURES --seed $SEED --method both \
        --csv "${OUTDIR}/tv_levy_sqa_L${L}_job${JOBID}.csv" \
        > "${OUTDIR}/tv_levy_sqa_L${L}_job${JOBID}.stdout" 2>&1 &
    pids+=($!)

    # Yip: SA-sampled subspace diagonalisation (selected-CI convergence ladder)
    python3 "$CODEDIR/tv_yip_sa.py" \
        --L $L --t $TVAL --V $VVAL --reads $READS --sweeps $SWEEPS \
        --max_support $MAXSUP --seed $SEED \
        --csv "${OUTDIR}/tv_yip_sa_L${L}_job${JOBID}.csv" \
        > "${OUTDIR}/tv_yip_sa_L${L}_job${JOBID}.stdout" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do wait $pid; done

# ── summary table ─────────────────────────────────────────────────────────────
python3 - <<PYEOF
import glob, re

jobid = "$JOBID"
outdir = "$OUTDIR"

def hdr_of(f):
    with open(f) as fh: return fh.readline()

print("\n=== LEVY SQA ground-state energy (SSE = exact-in-dtau, PIMC = Trotter) ===")
print(f"{'L':>3} {'ED':>12} {'SSE':>12} {'PIMC':>12}")
for f in sorted(glob.glob(f"{outdir}/tv_levy_sqa_L*_job{jobid}.csv")):
    m = re.search(r'_L(\d+)_', f)
    if not m: continue
    hdr = hdr_of(f)
    ed = re.search(r'E0_ed=([-\d.]+|nan)', hdr)
    rows = {}
    with open(f) as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(',')
            if len(p) >= 3: rows[p[0]] = p[1]
    edv = ed.group(1) if ed else 'nan'
    print(f"{m.group(1):>3} {edv:>12} {rows.get('SSE','--'):>12} {rows.get('PIMC','--'):>12}")

print("\n=== YIP SA-subspace (final E_var and support fraction) ===")
print(f"{'L':>3} {'ED':>12} {'E_var':>12} {'|B|':>10} {'B/sector':>10}")
for f in sorted(glob.glob(f"{outdir}/tv_yip_sa_L*_job{jobid}.csv")):
    m = re.search(r'_L(\d+)_', f)
    if not m: continue
    hdr = hdr_of(f)
    ed = re.search(r'E0_ed=([-\d.]+|nan)', hdr)
    ev = re.search(r'E_var_final=([-\d.]+)', hdr)
    last = None
    with open(f) as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(',')
            if len(p) >= 4: last = p
    edv = ed.group(1) if ed else 'nan'
    evv = ev.group(1) if ev else '--'
    if last:
        print(f"{m.group(1):>3} {edv:>12} {evv:>12} {last[0]:>10} {float(last[3]):>10.4f}")
PYEOF

echo "done job $JOBID"
