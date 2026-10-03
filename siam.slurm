#!/bin/bash
###############################################################################
#  SLURM Job Script — SIAM XBK Pipeline  (r=2 confirmation run)
#  Single Impurity Anderson Model  |  BK -> Z2 taper -> XBK -> Dinkelbach -> SA
#  Pure MPI: one rank per core across (U,N_bath,sector,p) tasks.
#  (Nested multiprocessing inside MPI ranks corrupts OpenMPI+InfiniBand RDMA at
#   the first fork, so SA restarts run serially per rank; with 128 ranks and 180
#   tasks each rank gets ~1-2 tasks, so task-level MPI already isolates the heavy
#   N_bath=5 solves onto their own cores.)
###############################################################################
#SBATCH -J siam_xbk                     # Job name
#SBATCH -p kshort                       # Partition (12-hr wall-time)
#SBATCH -N 2                            # Nodes
#SBATCH -n 128                          # Total MPI tasks (ranks) = 2 nodes x 64 cores
#SBATCH -c 1                            # CPUs per MPI task (pure MPI; SA runs serially)
#SBATCH --ntasks-per-node=64            # 64 ranks/node x 2 nodes = 128 cores
#SBATCH --mem-per-cpu=2G
#SBATCH -t 0-12:00:00                   # kshort max; N_bath=5 single tasks are slow
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishithr@imsc.res.in
#SBATCH -D /home/shassan/XBKReal        # working directory (holds the .py files)
#SBATCH -o Out_%j.log                   # written to the -D dir (no subdir needed:
#SBATCH -e Err_%j.log                   #   SLURM opens these BEFORE the body runs)
###############################################################################
# 0. Setup
###############################################################################
echo "$SLURM_JOB_NODELIST" > hostfile_${SLURM_JOB_ID}
# Pure MPI (one rank per core); SA runs serially per rank. Keep BLAS single-
# threaded so each rank stays on its one core and never oversubscribes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
###############################################################################
# 1. Load modules
###############################################################################
module purge
module load anaconda/2025.06-0
module load openmpi/5.0.8
echo "======================================================================"
echo "  SIAM XBK (r=2 confirmation) — JOB STARTED"
echo "======================================================================"
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Nodes          : $SLURM_JOB_NODELIST"
echo "  Tasks (MPI)    : $SLURM_NTASKS"
echo "  CPUs/task      : $SLURM_CPUS_PER_TASK  (pure MPI; 1 core per rank)"
echo "  Start Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python         : $(which python3)"
echo "======================================================================"
###############################################################################
# 2. Verify required files
###############################################################################
for f in siam_mpi.py xbk_pipeline.py fieldmath.py; do
    [[ ! -f "$f" ]] && echo "ERROR: Missing '$f'" >&2 && exit 1
done
echo "[CHECK] All required files present."
# Quick dependency check (openfermion / mpi4py must be importable)
python3 -c "import openfermion, numpy, scipy, mpi4py" \
    || { echo "ERROR: missing python deps (openfermion/numpy/scipy/mpi4py)"; exit 1; }
echo "[CHECK] Python dependencies importable."
###############################################################################
# 3. Configurable parameters (override via environment variables)
###############################################################################
BATHS="${BATHS:-2 3 4 5}"
UVALS="${UVALS:-0 2 4 6 8}"
RVAL="${RVAL:-2}"
# Light fixed SA budget: a single N_bath=5 task is ~44 min at 4x600 (vs hours
# under the heavy adaptive default, which is why job 54502 timed out). Still a
# valid upper bound. Raise for tighter energies once a run completes.
RESTARTS="${RESTARTS:-4}"
SWEEPS="${SWEEPS:-600}"
CSV="${CSV:-siam_xbk_sweep_job${SLURM_JOB_ID}.csv}"
PNG="${PNG:-siam_xbk_sweep_job${SLURM_JOB_ID}.png}"
echo ""
echo "[CONFIG] baths=[$BATHS]  U=[$UVALS]  r=$RVAL  restarts=$RESTARTS sweeps=$SWEEPS"
echo "[CONFIG] csv=$CSV  png=$PNG"
echo ""
###############################################################################
# 4. Run
###############################################################################
START=$(date +%s)
mpirun -np $SLURM_NTASKS \
    --map-by slot:PE=$SLURM_CPUS_PER_TASK \
    --bind-to core \
    python3 siam_mpi.py \
        --baths $BATHS \
        --U $UVALS \
        --r $RVAL \
        --restarts $RESTARTS \
        --sweeps $SWEEPS \
        --csv "$CSV" \
        --png "$PNG"
EXIT_CODE=$?
ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "======================================================================"
echo "  JOB COMPLETED | Exit=$EXIT_CODE | Time=${ELAPSED}s"
echo "  Results: $CSV  /  $PNG"
echo "======================================================================"
exit $EXIT_CODE
