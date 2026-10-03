"""
export_pubo_dict.py — write model_pubo_dict.py, a single importable dictionary
of PUBOs (one entry per model) for the XBK r=2 ground subproblem.

Usage:  python export_pubo_dict.py [r]     # default r=2

Output file defines:
    PUBOS = {
        "<model>": {
            "label":   "<instance description>",
            "n_vars":  <int>,               # binary variables 0..n_vars-1
            "sector":  <int>, "p": <int>,   # which XBK subproblem this is
            "H":       {tuple(sorted idx): coeff, ...},   # numerator  <psi|H|psi>
            "C":       {...},                             # counting   <psi|psi>
            "P":       {...},                             # H - lambda*C  (min=0 at argmin)
            "lambda":  <float>,             # = XBK E0^(r) for this instance
            "E_xbk":   <float>, "E_exact": <float>,
        }, ...
    }
Each PUBO is over x in {0,1}; key () is the constant term.  min_x H/C == E_xbk.
"""
import sys, math
from collections import defaultdict
import numpy as np
import xbk_pipeline as xp

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def ising_to_pubo(ising):
    b = xp._ising_to_binary(ising)
    return {tuple(sorted(k)): float(v) for k, v in b.items()}


def eval_pubo(poly, x):
    tot = 0.0
    for k, c in poly.items():
        p = 1.0
        for i in k:
            p *= x[i]
        tot += c * p
    return tot


def ground_subproblem(H, r):
    """Return dict for the (sector,p) subproblem with the lowest brute-forced H/C."""
    spin = not hasattr(H, "terms") or "FermionOperator" in type(H).__name__
    Hq = H if type(H).__name__ == "QubitOperator" else xp.bk_transform(H)
    best = None
    for si, sec in enumerate(xp.z2_taper(Hq)):
        terms, n = xp._qubitop_to_terms(sec)
        if n == 0:
            const = float(np.real(sec.terms.get((), 0.0)))
            cand = dict(sector=si, p=None, n_vars=0, H={(): const}, C={(): 1.0}, e=const)
            if best is None or cand["e"] < best["e"] - 1e-9:
                best = cand
            continue
        if n * r > 20:               # brute force cap (2^20); larger handled below
            continue
        for p in range(math.ceil(r / 2) + 1):
            H_is = xp.build_xbk_ising(terms, n, r, p)
            C_is = xp.build_counting(n, r, p)
            if not C_is:
                continue
            Hp, Cp = ising_to_pubo(H_is), ising_to_pubo(C_is)
            nv = n * r
            be, bx = math.inf, None
            for mask in range(1 << nv):
                x = [(mask >> b) & 1 for b in range(nv)]
                cv = eval_pubo(Cp, x)
                if abs(cv) < 1e-12:
                    continue
                v = eval_pubo(Hp, x) / cv
                if v < be:
                    be, bx = v, x
            cand = dict(sector=si, p=p, n_vars=nv, H=Hp, C=Cp, e=be)
            if best is None or be < best["e"] - 1e-9:
                best = cand
    return best


def instances():
    return [
        ("tv1d",      xp.tv1d(4, 1.0, 2.0),        "L=4  V/t=2"),
        ("tv2d",      xp.tv2d(2, 1.0, 2.0),        "2x2  V/t=2"),
        ("hubbard1d", xp.hubbard1d(2, 1.0, 4.0),   "L=2  U/t=4 (half-fill)"),
        ("hubbard2d", xp.hubbard2d(2, 1.0, 4.0),   "2x2  U/t=4"),
        ("siam",      xp.siam(2, U=4.0),           "N_bath=2  U/t=4"),
        ("pam",       xp.pam(2, 1.0, 1.0, 4.0),    "N_b=2  U/t=4"),
        ("kitaev",    xp.kitaev(4, 1.0, 0.5),      "L=4  Delta/t=0.5"),
        ("j1j2_1d",   xp.j1j2_1d(4, 1.0, 0.5),     "L=4  J2/J1=0.5"),
        ("j1j2_2d",   xp.j1j2_2d(2, 1.0, 0.5),     "2x2  J2/J1=0.5"),
    ]


if __name__ == "__main__":
    out = {}
    print(f"{'model':10s} {'instance':24s} {'n_vars':>6} {'E_xbk':>9} {'E_exact':>9}")
    for name, H, label in instances():
        rec = ground_subproblem(H, R)
        Ex = xp.exact_ground_energy(H)
        lam = rec["e"]
        P = defaultdict(float)
        for k, c in rec["H"].items():
            P[k] += c
        for k, c in rec["C"].items():
            P[k] -= lam * c
        P = {k: v for k, v in P.items() if abs(v) > 1e-12}
        out[name] = dict(label=label, n_vars=rec["n_vars"], sector=rec["sector"],
                         p=rec["p"], H=rec["H"], C=rec["C"], P=P,
                         **{"lambda": lam}, E_xbk=lam, E_exact=Ex)
        print(f"{name:10s} {label:24s} {rec['n_vars']:>6} {lam:>9.4f} {Ex:>9.4f}")

    with open("model_pubo_dict.py", "w") as f:
        f.write(f'"""Auto-generated XBK PUBO dictionary (r={R}).\n')
        f.write("Each PUBOS[model] has H, C, P dicts over x in {0,1}; key () = constant.\n")
        f.write("min_x H(x)/C(x) == E_xbk;  P = H - lambda*C has global min 0 at that argmin.\n")
        f.write('"""\n\n')
        f.write(f"R = {R}\n\n")
        f.write("PUBOS = " + repr(out) + "\n")
    print(f"\nwrote model_pubo_dict.py  (importable: from model_pubo_dict import PUBOS)")
