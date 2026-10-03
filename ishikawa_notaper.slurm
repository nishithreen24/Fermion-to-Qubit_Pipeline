#!/bin/bash
###############################################################################
#  SLURM Job Script — XBK + ISHIKAWA reduction, NO-TAPER test
#  Runs run_ishikawa_test.py, which includes a no-taper t-V L=2 case (n=2) that
#  genuinely exercises quadratization (the tapered L=2 collapses to 0 qubits).
#  Compares exact ED vs k-local / Rosenberg / Ishikawa SA paths; all must agree.
###############################################################################
#SBATCH -J xbk_ish_notaper
#SBATCH -p kshort
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -c 1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2G
#SBATCH -t 0-00:30:00
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

echo "======================================================================"
echo "  XBK + ISHIKAWA (no-taper, exercises quadratization) — job $SLURM_JOB_ID"
echo "  start $(date '+%F %T')   python $(which python3)"
echo "======================================================================"

for f in xbk_pipeline.py fieldmath.py ishikawa.py run_ishikawa_test.py; do
    [[ ! -f "$f" ]] && echo "ERROR: missing '$f'" >&2 && exit 1
done
python3 -c "import openfermion, numpy, scipy" \
    || { echo "ERROR: missing python deps"; exit 1; }
echo "[CHECK] files and deps present."

python3 run_ishikawa_test.py
RC=$?

echo ""
echo "======================================================================"
echo "  RESULT: exit=$RC  (rows marked OK = Ishikawa matches exact/Rosenberg)"
echo "  finished $(date '+%F %T')"
echo "======================================================================"
exit $RC
