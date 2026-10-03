#!/bin/bash
###############################################################################
#  SLURM Job Script — XBK ground-state for the 2x3 Hubbard cluster (n=10).
#  hubbard2d_2x3 is a fixed-geometry model (Lx=2, Ly=3, OBC, plain U n_up n_dn);
#  the --L flag is ignored by the builder.  12 modes -> 12 BK qubits -> n=10 after
#  tapering, 4 sectors x (ceil(r/2)+1) p-projections per ratio.
#
#  Sweeps every r in RLIST (each a separate mpirun writing its own CSV), then
#  prints a combined convergence summary.  The 35,840-term (r=2) / much larger
#  (r>=4) n=10 polys make each task ~30 min (r=2) to many hours (r>=4), so
#  --fast (numpy-vectorised SA) is essential and the wide rank count matters.
#
#  Defaults target kmedium with r=2 and r=4 over the full U/t sweep.  For r=6
#  (frontier) override RLIST + resources on the sbatch line (see header notes).
#
#  kmedium (r=2 & r=4, full sweep):   sbatch hubbard23.slurm
#  klong   (r=6, one ratio):
#      RLIST="6" RATIOS="4" sbatch -p klong -N 3 -n 192 -t 7-00:00:00 hubbard23.slurm
#
#  EDIT account/dir if not shassan.
###############################################################################
#SBATCH -J xbk_hub23
#SBATCH -p kmedium
#SBATCH -N 2
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=2G
#SBATCH -t 2-00:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishithr@imsc.res.in
#SBATCH -D /home/shassan/XBKReal
#SBATCH -o Out_%j.log
#SBATCH -e Err_%j.log
###############################################################################
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge
module load anaconda/2025.06-0
module load openmpi/5.0.8

MODEL="hubbard2d_2x3"
RLIST="${RLIST:-2 4}"
RATIOS="${RATIOS:-0 1 2 3 4 5 6 7 8}"
RESTARTS="${RESTARTS:-10}"
SWEEPS="${SWEEPS:-2000}"

echo "======================================================================"
echo "  XBK 2x3 Hubbard (n=10):  r in [$RLIST]   U/t in [$RATIOS]"
echo "  ranks=$SLURM_NTASKS  budget=${RESTARTS}x${SWEEPS}  fast SA"
echo "  Job $SLURM_JOB_ID  start $(date '+%F %T')"
echo "======================================================================"

for f in models_mpi.py xbk_pipeline.py fieldmath.py sa_fast.py; do
    [[ ! -f "$f" ]] && echo "ERROR: missing '$f'" >&2 && exit 1
done
python3 -c "import openfermion, numpy, scipy, mpi4py" \
    || { echo "ERROR: missing python deps"; exit 1; }

START=$(date +%s)
for R in $RLIST; do
    CSV="${MODEL}_r${R}_job${SLURM_JOB_ID}.csv"
    PNG="${MODEL}_r${R}_job${SLURM_JOB_ID}.png"
    echo ""
    echo ">>> r=$R  ($(date '+%T'))  -> $CSV"
    mpirun -np $SLURM_NTASKS \
        --map-by slot:PE=$SLURM_CPUS_PER_TASK \
        --bind-to core \
        python3 models_mpi.py \
            --model "$MODEL" --L 2 --ratios $RATIOS --r "$R" \
            --restarts "$RESTARTS" --sweeps "$SWEEPS" --fast \
            --csv "$CSV" --png "$PNG"
done

echo ""
echo "======================================================================"
echo "  2x3 HUBBARD SUMMARY  (columns: U/t  E0_XBK  E0_exact  abs_err)"
for R in $RLIST; do
    CSV="${MODEL}_r${R}_job${SLURM_JOB_ID}.csv"
    echo "  --- r=$R ---"
    awk -F, 'NR>1 && NF>=5 {printf "   %4s  %10s  %10s  %8s\n", $1, $3, $4, $5}' "$CSV" 2>/dev/null
done
echo "  total wall: $(( $(date +%s) - START ))s"
echo "======================================================================"
