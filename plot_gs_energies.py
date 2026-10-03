#!/usr/bin/env python3
"""
plot_gs_energies.py -- compare the ground-state ENERGIES actually found by the
three honest t-V solvers against exact diagonalisation:

  * Levy  SQA-SSE   (stochastic series expansion, no Trotter error)
  * Levy  SQA-PIMC  (discrete-time world-line; shown dtau->0 EXTRAPOLATED so the
                     O(dtau^2) Trotter bias is removed -- see panel (c))
  * Yip   SA-subspace (selected-CI: SA samples configs, diagonalise the support)

Reads CSVs written by tv_levy_sqa.py (both the single-run and --trotter_scan
outputs) and tv_yip_sa.py.  ED reference is read from the CSV headers.

Design (dataviz house style):
  * methods are CATEGORICAL -> fixed Okabe-Ito colorblind-safe hues, never cycled;
    ED is the neutral reference line (ink), not a series color.
  * L is ORDERED -> viridis light->dark for the per-L curves in (c),(d).
  * single y-axis per panel, recessive grid, legend + direct labels.
"""
import argparse, glob, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

COL = {"SSE": "#0072B2", "PIMC": "#D55E00", "Yip": "#009E73"}   # Okabe-Ito
ED_INK = "#333333"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def hdr(p):
    with open(p) as fh:
        return fh.readline()


def hfloat(h, key):
    m = re.search(rf"{key}=([-\d.]+|nan)", h)
    return float(m.group(1)) if m else np.nan


def extrapolate(dtau, E, err, cut=0.35):
    """Linear-in-dtau^2 fit to dtau=0 using only the asymptotic points
    (dtau<=cut).  Returns (E0, err0).  Coarse dtau carry dtau^4 curvature and
    would bias a pure dtau^2 intercept, so they are excluded from the fit."""
    dtau, E, err = map(np.asarray, (dtau, E, err))
    m = dtau <= cut
    if m.sum() < 2:
        m = np.isin(np.arange(len(dtau)), np.argsort(dtau)[:2])
    x = dtau[m]**2; w = 1.0 / err[m]**2
    A = np.vstack([np.ones(m.sum()), x]).T
    coef, *_ = np.linalg.lstsq(A * w[:, None], E[m] * w, rcond=None)
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/Users/nishithreen/Downloads")
    ap.add_argument("--job", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = f"_job{args.job}" if args.job else ""

    # ---- Levy SQA single-run (SSE energy; PIMC coarse -- not plotted raw) ----
    levy = {}
    for f in sorted(glob.glob(f"{args.dir}/tv_levy_sqa_L*{tag}.csv")):
        L = int(re.search(r"_L(\d+)", f).group(1))
        h = hdr(f)
        rows = {}
        with open(f) as fh:
            next(fh)
            for line in fh:
                p = line.strip().split(",")
                if len(p) >= 3:
                    rows[p[0]] = (float(p[1]), float(p[2]))
        levy[L] = {"ed": hfloat(h, "E0_ed"), **rows}

    # ---- PIMC Trotter-extrapolation CSVs ----
    trot = {}
    for f in sorted(glob.glob(f"{args.dir}/tv_levy_pimc_trotter_L*{tag}.csv")):
        L = int(re.search(r"_L(\d+)", f).group(1))
        h = hdr(f)
        d = np.atleast_2d(np.genfromtxt(f, delimiter=",", skip_header=1))
        E0, err0 = extrapolate(d[:, 1], d[:, 2], d[:, 3])   # refit from raw points
        trot[L] = {"ed": hfloat(h, "E0_ed"),
                   "E0": E0, "err": err0,
                   "dtau": d[:, 1], "E": d[:, 2], "Eerr": d[:, 3]}

    # ---- Yip SA-subspace ----
    yip = {}
    for f in sorted(glob.glob(f"{args.dir}/tv_yip_sa_L*{tag}.csv")):
        L = int(re.search(r"_L(\d+)", f).group(1))
        h = hdr(f)
        d = np.atleast_2d(np.genfromtxt(f, delimiter=",", skip_header=1))
        yip[L] = {"ed": hfloat(h, "E0_ed"), "Efinal": hfloat(h, "E_var_final"),
                  "support": d[:, 0], "Evar": d[:, 1], "absdE": d[:, 2], "frac": d[:, 3]}

    if not (levy or yip or trot):
        sys.exit(f"no GS-energy CSVs found in {args.dir} (tag='{tag}')")

    Ls = sorted(set(levy) | set(yip) | set(trot))
    fig, AX = plt.subplots(2, 2, figsize=(12.5, 9))
    ax = AX.ravel()

    # ===== (a) E0 vs L : ED line + SSE + PIMC(extrap) + Yip =====
    edLs = [L for L in Ls if L in levy and np.isfinite(levy[L]["ed"])]
    if edLs:
        ax[0].plot(edLs, [levy[L]["ed"] for L in edLs], "-", color=ED_INK, lw=1.4,
                   zorder=1, label="ED (exact)")
    xs = [L for L in Ls if L in levy and "SSE" in levy[L]]
    if xs:
        ax[0].errorbar(xs, [levy[L]["SSE"][0] for L in xs],
                       yerr=[levy[L]["SSE"][1] for L in xs], fmt="o", ms=6,
                       color=COL["SSE"], capsize=2.5, lw=1.4, mec="white", mew=0.8,
                       label="SQA-SSE", zorder=3)
    xs = [L for L in Ls if L in trot and np.isfinite(trot[L]["E0"])]
    if xs:
        ax[0].errorbar(xs, [trot[L]["E0"] for L in xs],
                       yerr=[trot[L]["err"] for L in xs], fmt="D", ms=6,
                       color=COL["PIMC"], capsize=2.5, lw=1.4, mec="white", mew=0.8,
                       label=r"SQA-PIMC ($d\tau\!\to\!0$)", zorder=3)
    xs = [L for L in Ls if L in yip and np.isfinite(yip[L]["Efinal"])]
    if xs:
        ax[0].plot(xs, [yip[L]["Efinal"] for L in xs], "s", ms=6, color=COL["Yip"],
                   mec="white", mew=0.8, label="Yip SA-subspace", zorder=3)
    ax[0].set_xlabel("system size  $L$")
    ax[0].set_ylabel(r"ground-state energy  $E_0$")
    ax[0].set_title("(a)  $E_0$ vs $L$: solvers vs ED")
    ax[0].set_xticks(Ls); ax[0].legend(frameon=False, fontsize=8.5)

    # ===== (b) |E - ED| vs L (log-y) =====
    def acc(xs, ys):
        return [x for x, y in zip(xs, ys)], [max(abs(v), 1e-6) for v in ys]
    xs = [L for L in Ls if L in levy and "SSE" in levy[L] and np.isfinite(levy[L]["ed"])]
    if xs:
        ax[1].plot(*acc(xs, [levy[L]["SSE"][0]-levy[L]["ed"] for L in xs]),
                   "-o", color=COL["SSE"], ms=5, lw=1.5, mec="white", mew=0.7, label="SQA-SSE")
    xs = [L for L in Ls if L in trot and np.isfinite(trot[L]["E0"]) and np.isfinite(trot[L]["ed"])]
    if xs:
        ax[1].plot(*acc(xs, [trot[L]["E0"]-trot[L]["ed"] for L in xs]),
                   "-D", color=COL["PIMC"], ms=5, lw=1.5, mec="white", mew=0.7,
                   label=r"SQA-PIMC ($d\tau\!\to\!0$)")
    xs = [L for L in Ls if L in yip and np.isfinite(yip[L]["Efinal"]) and np.isfinite(yip[L]["ed"])]
    if xs:
        ax[1].plot(*acc(xs, [yip[L]["Efinal"]-yip[L]["ed"] for L in xs]),
                   "-s", color=COL["Yip"], ms=5, lw=1.5, mec="white", mew=0.7, label="Yip SA-subspace")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("system size  $L$")
    ax[1].set_ylabel(r"$|E_{\mathrm{method}}-E_{\mathrm{ED}}|$")
    ax[1].set_title("(b)  accuracy vs ED"); ax[1].set_xticks(Ls)
    ax[1].legend(frameon=False, fontsize=8.5)
    # Yip pinned to the 1e-6 floor here = exact to machine precision (the
    # half-filling sector is small enough to enumerate fully); it only becomes a
    # genuine variational error once dim(sector) > max_support at larger L.
    if any(L in yip and np.isfinite(yip[L].get("Efinal", np.nan))
           and abs(yip[L]["Efinal"] - yip[L]["ed"]) < 1e-6 for L in Ls
           if L in yip and np.isfinite(yip[L].get("ed", np.nan))):
        ax[1].annotate("Yip exact here\n(full sector enumerated)",
                       xy=(0.5, 0.07), xycoords="axes fraction", ha="center",
                       fontsize=7.5, color=COL["Yip"])

    # ===== (c) PIMC Trotter extrapolation: E vs dtau^2 -> ED =====
    if trot:
        tLs = sorted(trot)
        cols = cm.viridis(np.linspace(0.15, 0.85, len(tLs)))
        for L, c in zip(tLs, cols):
            d = trot[L]
            x2 = d["dtau"]**2
            ax[2].errorbar(x2, d["E"], yerr=d["Eerr"], fmt="o", ms=5, color=c,
                           capsize=2, lw=1.2, label=f"L={L}")
            xf = np.linspace(0, x2.max()*1.05, 50)
            ax[2].plot(xf, d["E0"] + (d["E"][-1]-d["E0"])/x2[-1]*xf, "--", color=c, lw=1.2)
            ax[2].plot(0, d["E0"], "*", color=c, ms=13, mec="white", mew=0.8, zorder=5)
            if np.isfinite(d["ed"]):
                ax[2].axhline(d["ed"], color=c, lw=0.8, alpha=0.35)
        ax[2].set_xlabel(r"$d\tau^{2}$")
        ax[2].set_ylabel(r"PIMC energy  $E(d\tau)$")
        ax[2].set_title(r"(c)  PIMC Trotter extrapolation ($\star=d\tau\!\to\!0$; line=ED)")
        ax[2].legend(frameon=False, fontsize=8.5)
    else:
        ax[2].axis("off")

    # ===== (d) Yip selected-CI convergence =====
    if yip:
        yLs = sorted(yip)
        cols = cm.viridis(np.linspace(0.12, 0.88, len(yLs)))
        for L, c in zip(yLs, cols):
            d = yip[L]; m = d["absdE"] > 0
            if m.any():
                ax[3].plot(d["frac"][m]*100, np.maximum(d["absdE"][m], 1e-9),
                           "-o", color=c, ms=3.5, lw=1.5, label=f"L={L}")
        ax[3].set_yscale("log")
        ax[3].set_xlabel("support fraction  $|B|/\\dim$  (%)")
        ax[3].set_ylabel(r"$|E_{\mathrm{var}}-E_{\mathrm{ED}}|$")
        ax[3].set_title("(d)  Yip selected-CI convergence")
        ax[3].legend(frameon=False, fontsize=8, ncol=2)
    else:
        ax[3].axis("off")

    fig.suptitle("t-V 1D  --  ground-state energies actually found "
                 "(Levy SQA: SSE + PIMC,  Yip SA-subspace)  vs ED",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    out = args.out or f"{args.dir}/tv_gs_energies.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", out)

    # ---- console summary ----
    print(f"\n{'L':>3} {'ED':>11} {'SSE':>20} {'PIMC(dt->0)':>20} {'Yip E_var':>11}")
    for L in Ls:
        ed = (levy.get(L, {}).get("ed") or yip.get(L, {}).get("ed")
              or trot.get(L, {}).get("ed") or np.nan)
        sse = levy.get(L, {}).get("SSE")
        pi = trot.get(L)
        yv = yip.get(L, {}).get("Efinal", np.nan)
        sse_s = f"{sse[0]:.5f}+/-{sse[1]:.5f}" if sse else "--"
        pi_s = f"{pi['E0']:.5f}+/-{pi['err']:.5f}" if pi else "--"
        print(f"{L:>3} {ed:>11.5f} {sse_s:>20} {pi_s:>20} {yv:>11.5f}")


if __name__ == "__main__":
    main()
