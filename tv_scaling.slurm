#!/bin/bash
#SBATCH -J tv_scaling
#SBATCH -D /home/shassan/XBKReal
#SBATCH -p kshort
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -c 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nishithr@imsc.res.in
#SBATCH -o logs/tv_scaling_%j.log

# ── parameters (override with env vars) ──────────────────────────────────────
LVALS="${LVALS:-3 4 5 6 7 8}"
VVAL="${VVAL:-2.0}"
NPTS="${NPTS:-201}"
# ─────────────────────────────────────────────────────────────────────────────

module purge
module load anaconda/2025.06-0 openmpi/5.0.8

CODEDIR="/home/shassan/XBKReal/code"
OUTDIR="/home/shassan/XBKReal/results"
JOBID=$SLURM_JOB_ID

mkdir -p "$OUTDIR"
echo "tv_scaling job $JOBID  L=$LVALS  V=$VVAL"

pids=()
for L in $LVALS; do
    python3 "$CODEDIR/tv_levy.py" \
        --L $L --t 1.0 --V $VVAL --npts $NPTS \
        --csv "${OUTDIR}/tv_levy_L${L}_job${JOBID}.csv" &
    pids+=($!)

    python3 "$CODEDIR/tv_yip.py" \
        --L $L --t 1.0 --V $VVAL --seeds 40 \
        --csv "${OUTDIR}/tv_yip_L${L}_job${JOBID}.csv" &
    pids+=($!)
done

for pid in "${pids[@]}"; do wait $pid; done

# ── summary table ─────────────────────────────────────────────────────────────
python3 - <<PYEOF
import glob, re

jobid = "$JOBID"
outdir = "$OUTDIR"

print("\n=== LEVY gap_min (N-sector) and T_req ===")
print(f"{'L':>3} {'gap_min':>10} {'T_req':>10}")
for f in sorted(glob.glob(f"{outdir}/tv_levy_L*_job{jobid}.csv")):
    m = re.search(r'_L(\d+)_', f)
    if not m: continue
    with open(f) as fh: hdr = fh.readline()
    gm = re.search(r'gapmin_sec=([\d.]+)', hdr)
    if gm:
        g = float(gm.group(1))
        T = 1/g**2 if g > 0 else float('inf')
        print(f"{m.group(1):>3} {g:>10.5f} {T:>10.2f}")

print("\n=== YIP S_required for eps=0.01 ===")
print(f"{'L':>3} {'PR':>8} {'S(eps=0.01)':>14}")
for f in sorted(glob.glob(f"{outdir}/tv_yip_L*_job{jobid}.csv")):
    m = re.search(r'_L(\d+)_', f)
    if not m: continue
    with open(f) as fh:
        hdr = fh.readline()
        rows = [line.split(',') for line in fh if not line.startswith('#')]
    pr = re.search(r'PR=([\d.]+)', hdr)
    S_need = next((int(r[0]) for r in rows if len(r)>1 and float(r[1])<=0.01), None)
    prv = float(pr.group(1)) if pr else float('nan')
    print(f"{m.group(1):>3} {prv:>8.2f} {'%d' % S_need if S_need else '>max':>14}")
PYEOF

echo "done job $JOBID"
