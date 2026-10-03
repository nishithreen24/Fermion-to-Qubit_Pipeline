#!/bin/bash
###############################################################################
#  SLURM Job Script — XBK REPLICATION-CONVERGENCE ladder for the 2x3 Hubbard
#  (n=10 after tapering) at a SINGLE coupling U/t, sweeping r = 2,4,6,8,10.
#
#  WHY THIS JOB EXISTS
#  -------------------
#  The full-sweep runs (hubbard23.slurm) showed the 2x3 XBK bound frozen at
#  E0=-1 (r=2) / -2 (r=4), far above ED (-5.18 at U=4).  An independent exact
#  search CONFIRMED -1.0 IS the true rank-2 optimum: this is genuine rank
#  starvation (the n=10 2D ground state needs r ~ n), NOT a solver failure.
#  This job measures how the bound recovers as r -> n, which is the honest
#  r-convergence result the benchmark needs for its 2D limitation claim.
#
#  DISTINGUISHING TRUE BOUND FROM SOLVER STALL
#  -------------------------------------------
#  At each r we run TWO budgets:  B (=2x the size-adaptive default) and 2B
#  (=4x default).  If E0(B) == E0(2B) to a few mK, the value is solver-
#  converged and reflects the ANSATZ.  If 2B is lower, the SA has NOT yet
#  found the rank-r optimum at that r and the point must be rerun harder
#  (bump BOOST).  The summary prints both and flags any gap > TOL.
#
#  Size-adaptive default is restarts=10, sweeps=max(2000,50*m), m=n*r.
#  Here n=10 so m = 20,40,60,80,100 for r=2..10; sweeps scale with m already.
#  We override restarts (not sweeps) to 2x/4x that default for the two budgets,
#  keeping the m-scaled anneal length intact.
#
#  COST WARNING
#  ------------
#  r=10 is m=100 physical spins with a very large quadratized QUBO; a single
#  Dinkelbach chain there is the frontier of this SA setup.  r=8,10 at 2B/4B
#  may take many hours to >1 day each -> this is a klong job.  If wall time is
#  tight, submit r=2,4,6 first (RLIST="2 4 6") and r=8,10 as a follow-up.
#
#  RUN
#  ---
#      sbatch hub23_rconv.slurm                      # U/t=4, r=2..10, B & 2B
#      RATIO=6 RLIST="2 4 6 8 10" sbatch hub23_rconv.slurm
#      RLIST="2 4 6" sbatch -p kmedium -t 2-00:00:00 hub23_rconv.slurm
#
#  EDIT account/dir if not shassan.
###############################################################################
#SBATCH -J xbk_h23rc
#SBATCH -p klong
#SBATCH -N 3
#SBATCH -n 192
#SBATCH -c 1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=2G
#SBATCH -t 7-00:00:00
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
RATIO="${RATIO:-4}"                 # single coupling U/t for the ladder
RLIST="${RLIST:-2 4 6 8 10}"        # replication factors to test
# Two budgets per r for the convergence certificate.  BASE multiplies the
# size-adaptive restart count (default 10); BOOST is the harder cross-check.
BASE="${BASE:-2}"                   # B  = BASE  x adaptive restarts
BOOST="${BOOST:-4}"                 # 2B = BOOST x adaptive restarts
TOL="${TOL:-0.02}"                  # |E0(B)-E0(2B)| below this = converged
SWEEPS="${SWEEPS:-}"                # empty -> keep m-scaled adaptive anneal length

echo "======================================================================"
echo "  XBK 2x3 Hubbard r-CONVERGENCE  (n=10):  U/t=$RATIO   r in [$RLIST]"
echo "  budgets: B=${BASE}x-adaptive restarts, 2B=${BOOST}x  (sweeps: ${SWEEPS:-adaptive})"
echo "  ranks=$SLURM_NTASKS   fast SA   TOL=$TOL"
echo "  Job $SLURM_JOB_ID  start $(date '+%F %T')"
echo "======================================================================"

for f in models_mpi.py xbk_pipeline.py fieldmath.py sa_fast.py; do
    [[ ! -f "$f" ]] && echo "ERROR: missing '$f'" >&2 && exit 1
done
python3 -c "import openfermion, numpy, scipy, mpi4py" \
    || { echo "ERROR: missing python deps"; exit 1; }

# adaptive default restart count is 10 (see models_mpi._budget_for)
ADAPT_RESTARTS=10
RB=$(( ADAPT_RESTARTS * BASE ))
R2B=$(( ADAPT_RESTARTS * BOOST ))
SWARG=""
[[ -n "$SWEEPS" ]] && SWARG="--sweeps $SWEEPS"

START=$(date +%s)
run_one () {   # $1=r  $2=restarts  $3=tag(B|2B) -> writes CSV, echoes E0
    local R="$1" NR="$2" TAG="$3"
    local CSV="${MODEL}_rconv_U${RATIO}_r${R}_${TAG}_job${SLURM_JOB_ID}.csv"
    echo "" >&2
    echo ">>> r=$R  budget=$TAG (${NR} restarts)  ($(date '+%T'))  -> $CSV" >&2
    mpirun -np $SLURM_NTASKS \
        --map-by slot:PE=$SLURM_CPUS_PER_TASK \
        --bind-to core \
        python3 models_mpi.py \
            --model "$MODEL" --L 2 --ratios "$RATIO" --r "$R" \
            --restarts "$NR" $SWARG --fast \
            --csv "$CSV" \
            --png "${MODEL}_rconv_U${RATIO}_r${R}_${TAG}_job${SLURM_JOB_ID}.png" >&2
    # data row: ratio,L,E0_XBK,E0_exact,abs_err  -> print E0_XBK
    grep -E "^${RATIO}(\.0*)?," "$CSV" 2>/dev/null | tail -1 | awk -F, '{print $3}'
}

declare -A EB E2B EX
for R in $RLIST; do
    eb=$(run_one  "$R" "$RB"  "B")
    e2=$(run_one  "$R" "$R2B" "2B")
    EB[$R]="$eb"; E2B[$R]="$e2"
    # capture exact once (same in both CSVs)
    CSVB="${MODEL}_rconv_U${RATIO}_r${R}_B_job${SLURM_JOB_ID}.csv"
    EX[$R]=$(grep -E "^${RATIO}(\.0*)?," "$CSVB" 2>/dev/null | tail -1 | awk -F, '{print $4}')
done

echo ""
echo "======================================================================"
echo "  2x3 HUBBARD r-CONVERGENCE SUMMARY   (U/t=$RATIO,  n=10,  exact=${EX[2]:-?})"
echo "----------------------------------------------------------------------"
printf "  %3s  %12s  %12s  %10s  %8s  %s\n" r "E0(B)" "E0(2B)" "|dE|(2B)" "B-2B" "status"
prev=""
for R in $RLIST; do
    eb="${EB[$R]}"; e2="${E2B[$R]}"; ex="${EX[$R]}"
    awk -v r="$R" -v eb="$eb" -v e2="$e2" -v ex="$ex" -v tol="$TOL" -v prev="$prev" 'BEGIN{
        best=(e2<eb)?e2:eb;
        dE = (ex!="")? (best-ex) : 0; if(dE<0)dE=-dE;
        gap = eb-e2; if(gap<0)gap=-gap;
        conv = (gap<=tol)? "converged" : "STALL(2B<B: rerun harder)";
        mono = (prev!="" && best>prev+1e-6)? " !NON-MONOTONE" : "";
        printf "  %3d  %12.4f  %12.4f  %10.4f  %8.4f  %s%s\n", r, eb, e2, dE, gap, conv, mono;
    }'
    # update prev to the better (lower) of the two budgets
    prev=$(awk -v eb="$eb" -v e2="$e2" 'BEGIN{print (e2<eb)?e2:eb}')
done
echo "----------------------------------------------------------------------"
echo "  Reading:"
echo "   * 'converged'  -> E0(2B) is the true rank-r bound; use it in the paper."
echo "   * 'STALL'      -> SA not at the rank-r optimum; rerun that r with"
echo "                     BOOST=8 (or higher) before trusting the point."
echo "   * |dE|(2B) vs r is the convergence curve; monotone decrease = ansatz"
echo "                     recovering as r -> n=10."
echo "  total wall: $(( $(date +%s) - START ))s"
echo "======================================================================"
