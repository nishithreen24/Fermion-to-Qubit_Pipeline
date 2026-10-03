#!/bin/bash
###############################################################################
#  SLURM Job Script — 1D t-V XBK Pipeline  (verification run, L = 3 only)
#  Spinless 1D t-V model  |  OpenMPI + mpi4py + neal
###############################################################################
#SBATCH -J tvtest                      # Job name
#SBATCH -p kshort                      # Partition (12-hr wall-time)
#SBATCH -N 1                           # Nodes
#SBATCH -n 16                          # Total MPI tasks (minimum allowed on Kamet)
#SBATCH -c 4                           # CPUs per MPI task (SA chains via multiprocessing)
#SBATCH --ntasks-per-node=16           # All 16 ranks on 1 node
#SBATCH --mem-per-cpu=1G
#SBATCH -t 0-01:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shassan@imsc.res.in
#SBATCH -D /home/shassan/XBK           # working directory
#SBATCH -o logs/Out_%j.log
#SBATCH -e logs/Err_%j.log
###############################################################################
# 0. Setup
###############################################################################
mkdir -p logs output/tvtest
echo "$SLURM_JOB_NODELIST" > logs/hostfile_${SLURM_JOB_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
###############################################################################
# 1. Load modules
###############################################################################
module purge
module load anaconda/2025.06-0
module load openmpi/5.0.8
echo "======================================================================"
echo "  1D t-V VERIFY (L=3) — JOB STARTED"
echo "======================================================================"
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Nodes          : $SLURM_JOB_NODELIST"
echo "  Tasks (MPI)    : $SLURM_NTASKS"
echo "  CPUs/task      : $SLURM_CPUS_PER_TASK"
echo "  Start Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python         : $(which python3)"
echo "======================================================================"
###############################################################################
# 2. Verify required files
###############################################################################
for f in tvtest.py pipeline_base.py Models.py; do
    [[ ! -f "$f" ]] && echo "ERROR: Missing '$f'" >&2 && exit 1
done
echo "[CHECK] All required files present."
###############################################################################
# 3. Configurable parameters (override via environment variables)
###############################################################################
L_FIXED="${L_FIXED:-3}"
VT_MIN="${VT_MIN:-0.0}"
VT_MAX="${VT_MAX:-4.0}"
VT_POINTS="${VT_POINTS:-5}"
NUM_SWEEPS="${NUM_SWEEPS:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-output/tvtest/run_job${SLURM_JOB_ID}}"
echo ""
echo "[CONFIG] L=$L_FIXED  V/t=[$VT_MIN,$VT_MAX] x$VT_POINTS"
echo "[CONFIG] Sweeps=$NUM_SWEEPS  Output=$OUTPUT_DIR"
echo ""
###############################################################################
# 4. Run
###############################################################################
START=$(date +%s)
mpirun -np $SLURM_NTASKS \
    --map-by slot:PE=$SLURM_CPUS_PER_TASK \
    --bind-to core \
    python3 tvtest.py
EXIT_CODE=$?
ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "======================================================================"
echo "  JOB COMPLETED | Exit=$EXIT_CODE | Time=${ELAPSED}s | Results=$OUTPUT_DIR"
echo "======================================================================"
exit $EXIT_CODE
