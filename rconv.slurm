#!/bin/bash
###############################################################################
#  SLURM Job Script — XBK replication-convergence curve at fixed (model,L,coupling)
#  Sweeps r = 2,4,6,8,10,12 for ONE n=6 instance to settle whether the bound
#  breaks its plateau and what r* reaches a target accuracy.  Each r is a
#  separate mpirun writing its own CSV; a summary is printed at the end.
#
#  Default: Kitaev L=6, Delta/t=0.5 (n=6, moderately entangled; plateaued at
#  r=4=r=6 in the local test).  Override via MODEL/LVAL/RATIO/RLIST.
#
#  EDIT account/dir if not shassan, then:  sbatch rconv.slurm
###############################################################################
#SBATCH -J xbk_rconv
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

MODEL="${MODEL:-kitaev}"
LVAL="${LVAL:-6}"
RATIO="${RATIO:-0.5}"
RLIST="${RLIST:-2 4 6 8 10 12}"
# Fixed, well-converged SA budget so the gap reflects the ANSATZ, not the
# solver, at every r (the point of a convergence study). --fast is essential
# at r=10,12 (n*r up to 72 spins).
RESTARTS="${RESTARTS:-8}"
SWEEPS="${SWEEPS:-2000}"

echo "======================================================================"
echo "  XBK r-convergence: model=$MODEL  L=$LVAL  ratio=$RATIO  r in [$RLIST]"
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
    CSV="${MODEL}_rconv_L${LVAL}_r${R}_job${SLURM_JOB_ID}.csv"
    echo ""
    echo ">>> r=$R  ($(date '+%T'))  -> $CSV"
    mpirun -np $SLURM_NTASKS \
        --map-by slot:PE=$SLURM_CPUS_PER_TASK \
        --bind-to core \
        python3 models_mpi.py \
            --model "$MODEL" --L "$LVAL" --ratios "$RATIO" --r "$R" \
            --restarts "$RESTARTS" --sweeps "$SWEEPS" --fast \
            --csv "$CSV" --png "${MODEL}_rconv_L${LVAL}_r${R}_job${SLURM_JOB_ID}.png"
done

echo ""
echo "======================================================================"
echo "  CONVERGENCE SUMMARY  ($MODEL L=$LVAL ratio=$RATIO)"
echo "    r    E0_XBK      E0_exact    |dE|"
for R in $RLIST; do
    CSV="${MODEL}_rconv_L${LVAL}_r${R}_job${SLURM_JOB_ID}.csv"
    # data row = last non-header line; columns: ratio,L,E0_XBK,E0_exact,abs_err
    row=$(grep -E "^${RATIO}," "$CSV" 2>/dev/null | tail -1)
    awk -v r="$R" -F, 'NF>=5{printf "  %3d  %10s  %10s  %8s\n", r, $3, $4, $5}' <<< "$row"
done
echo "  total wall: $(( $(date +%s) - START ))s"
echo "======================================================================"
