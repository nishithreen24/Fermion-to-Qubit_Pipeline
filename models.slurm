#!/bin/bash
###############################################################################
#  SLURM Job Script — general XBK ground-state sweep for ANY model
#  BK -> Z2 taper -> XBK -> Dinkelbach -> SA, swept over (ratio, L).
#  Pure MPI: one rank per core; SA serial per rank (no nested fork on IB).
#
#  Pick the model and (optionally) the sweep via environment variables:
#     MODEL=kitaev sbatch models.slurm
#     MODEL=j1j2_1d LVALS="4 6 8" RATIOS="0 0.2 0.4 0.6 0.8 1.0" sbatch models.slurm
#  Supported MODELs: tv1d tv2d hubbard1d hubbard2d siam pam kitaev j1j2_1d j1j2_2d
###############################################################################
#SBATCH -J xbk_model
#SBATCH -p kshort
#SBATCH -N 2
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=2G
#SBATCH -t 0-12:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishithr@imsc.res.in
#SBATCH -D /home/shassan/XBKReal
#SBATCH -o Out_%j.log
#SBATCH -e Err_%j.log
###############################################################################
echo "$SLURM_JOB_NODELIST" > hostfile_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge
module load anaconda/2025.06-0
module load openmpi/5.0.8

MODEL="${MODEL:-kitaev}"
RVAL="${RVAL:-2}"
CSV="${CSV:-${MODEL}_xbk_sweep_job${SLURM_JOB_ID}.csv}"
PNG="${PNG:-${MODEL}_xbk_sweep_job${SLURM_JOB_ID}.png}"

echo "======================================================================"
echo "  XBK MODEL SWEEP — $MODEL  (job $SLURM_JOB_ID)"
echo "  ranks=$SLURM_NTASKS  r=$RVAL  budget=${RESTARTS:-adaptive}x${SWEEPS:-}"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')   python: $(which python3)"
echo "======================================================================"

for f in models_mpi.py xbk_pipeline.py fieldmath.py sa_fast.py; do
    [[ ! -f "$f" ]] && echo "ERROR: missing '$f'" >&2 && exit 1
done
python3 -c "import openfermion, numpy, scipy, mpi4py" \
    || { echo "ERROR: missing python deps"; exit 1; }

# Optional overrides; omit to use models_mpi.py defaults (size-adaptive budget,
# per-model ratio/L grid).
EXTRA=""
[[ -n "$RATIOS"   ]] && EXTRA="$EXTRA --ratios $RATIOS"
[[ -n "$LVALS"    ]] && EXTRA="$EXTRA --L $LVALS"
[[ -n "$RESTARTS" ]] && EXTRA="$EXTRA --restarts $RESTARTS"
[[ -n "$SWEEPS"   ]] && EXTRA="$EXTRA --sweeps $SWEEPS"
[[ -n "$FAST"     ]] && EXTRA="$EXTRA --fast"     # FAST=1 -> numpy-vectorised SA

START=$(date +%s)
mpirun -np $SLURM_NTASKS \
    --map-by slot:PE=$SLURM_CPUS_PER_TASK \
    --bind-to core \
    python3 models_mpi.py \
        --model "$MODEL" \
        --r $RVAL \
        --csv "$CSV" \
        --png "$PNG" \
        $EXTRA
EXIT_CODE=$?
echo ""
echo "======================================================================"
echo "  JOB COMPLETED | Exit=$EXIT_CODE | Time=$(( $(date +%s) - START ))s"
echo "  Results: $CSV / $PNG   (salvage a timeout with --reduce-partials)"
echo "======================================================================"
exit $EXIT_CODE
