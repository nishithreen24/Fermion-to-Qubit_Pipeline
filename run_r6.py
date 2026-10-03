"""Run one case at r=6 with live, unbuffered output. Usage: run_r6.py <case>"""
import sys, time
import xbk_pipeline as xp

CASES = {
    "pbc": ("tv1d L3 V1.5 PBC", xp.tv1d(3, 1, 1.5, periodic=True), -2.3750, -2.3750),
    "l4":  ("tv1d L4 V0.5",     xp.tv1d(4, 1, 0.5),               -2.4679, -2.4107),
}

key = sys.argv[1]
name, H, Ex, opt = CASES[key]
r = 6
t0 = time.time()

H_bk = xp.bk_transform(H)
sectors = xp.z2_taper(H_bk)
print(f"[{name}] r={r}  exact={Ex:.4f}  r6_opt={opt:.4f}  "
      f"{len(sectors)} sectors", flush=True)

best = float("inf")
for si, sec in enumerate(sectors):
    ts = time.time()
    e = xp.run_xbk(sec, r=r, seed=3 + si * 101)
    best = min(best, e)
    print(f"  sector {si}: E0={e:.4f}   running_best={best:.4f}   "
          f"(+{time.time()-ts:.0f}s)", flush=True)

dt = time.time() - t0
hit = "YES" if abs(best - opt) < 1e-3 else f"{best-opt:+.4f}"
print(f"[{name}] r6_SA={best:.4f}  opt={opt:.4f}  hit={hit}  "
      f"err_vs_exact={abs(best-Ex):.4f}  total={dt:.0f}s", flush=True)
