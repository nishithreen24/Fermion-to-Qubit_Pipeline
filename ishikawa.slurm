#!/bin/bash
###############################################################################
#  SLURM Job Script — XBK pipeline with ISHIKAWA reduction (proof-of-concept)
#  Runs the t-V model L=2, r=2 through the pipeline using Ishikawa higher-order
#  -> quadratic reduction instead of Rosenberg, and checks the recovered
#  ground-state energy against exact diagonalization and the Rosenberg path.
#
#  If this reproduces the exact/Rosenberg energy, we roll Ishikawa into the
#  main files.  EDIT account/dir if not shassan, then:  sbatch ishikawa.slurm
###############################################################################
#SBATCH -J xbk_ishikawa
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
echo "  XBK + ISHIKAWA reduction — t-V L=2, r=2  (job $SLURM_JOB_ID)"
echo "  start $(date '+%F %T')   python $(which python3)"
echo "======================================================================"

for f in xbk_pipeline.py fieldmath.py ishikawa.py; do
    [[ ! -f "$f" ]] && echo "ERROR: missing '$f'" >&2 && exit 1
done
python3 -c "import openfermion, numpy, scipy" \
    || { echo "ERROR: missing python deps (openfermion/numpy/scipy)"; exit 1; }
echo "[CHECK] files and deps present."

# Single-process check (no MPI needed for this tiny instance): compares
# exact ED, k-local SA, Rosenberg-quad SA, and Ishikawa-quad SA. Exits non-zero
# if Ishikawa disagrees with exact/Rosenberg, so the job FAILS loudly on a bug.
python3 - <<'PY'
import math, sys
import numpy as np
import xbk_pipeline as xp
from ishikawa import quadratize_ishikawa
from collections import defaultdict

def dinkelbach_ishikawa(H_is, C_is, n_phys, seed, nr=8, ns=1500, mx=30, tol=1e-6):
    lam, best = 0.0, math.inf
    base = max((abs(c) for c in H_is.values()), default=1.0) \
         + max((abs(c) for c in C_is.values()), default=1.0)
    for it in range(mx):
        comb = defaultdict(float, H_is)
        for m, c in C_is.items(): comb[m] -= lam*c
        comb = {m:v for m,v in comb.items() if abs(v) > 1e-12}
        if not comb: break
        q2, nid = quadratize_ishikawa(xp._ising_to_binary(comb), n_phys)  # no M
        spins,_ = xp.sa_run(xp._binary_to_ising(q2), nid, nr, ns,
                            max(2.0*base,5.0), 1e-3, seed+it*1009)
        Hv, Cv = xp._eval(H_is, spins), xp._eval(C_is, spins)
        if abs(Cv) < 1e-8: continue
        new = Hv/Cv; best = min(best, new)
        if abs(new-lam) < tol: break
        lam = new
    return best

def gse_ishikawa(H, r=2, seed=3):
    best = math.inf
    for si, sec in enumerate(xp.z2_taper(xp.bk_transform(H))):
        terms, n = xp._qubitop_to_terms(sec)
        if n == 0:
            best = min(best, float(np.real(sec.terms.get((),0.0)))); continue
        for p in range(math.ceil(r/2)+1):
            H_is = xp.build_xbk_ising(terms, n, r, p)
            C_is = xp.build_counting(n, r, p)
            if not C_is: continue
            best = min(best, dinkelbach_ishikawa(H_is, C_is, n*r, seed+si*101+p*37))
    return best

H = xp.tv1d(2, 1.0, 1.0)
Ex = xp.exact_ground_energy(H)
e_k = xp.ground_state_energy(H, r=2, n_restarts=8, n_sweeps=1500, seed=3)
e_r = xp.ground_state_energy(H, r=2, n_restarts=8, n_sweeps=1500, seed=3, quadratic=True)
e_i = gse_ishikawa(H, r=2, seed=3)
print(f"  exact ED        = {Ex:.6f}")
print(f"  k-local  SA     = {e_k:.6f}")
print(f"  Rosenberg-quad  = {e_r:.6f}")
print(f"  Ishikawa-quad   = {e_i:.6f}")
ok = abs(e_i-e_r) < 1e-6 and e_i >= Ex - 1e-6
print(f"  Ishikawa == Rosenberg and >= exact : {ok}")
sys.exit(0 if ok else 1)
PY
RC=$?

echo ""
echo "======================================================================"
echo "  RESULT: exit=$RC  ($([[ $RC -eq 0 ]] && echo 'Ishikawa PASS' || echo 'Ishikawa FAIL'))"
echo "  finished $(date '+%F %T')"
echo "======================================================================"
exit $RC
