#!/usr/bin/env python3
"""
hubbard2d_pubos.py -- post-XBK, PRE-REDUCTION PUBOs for 2D Hubbard at
4x4 (16 sites / 32 modes) and 4x6 (24 sites / 48 modes), half filling,
U/t = 4, r = 2 -- written in the SAME two formats as the NHOIM inputs
model_pubos.py and model_pubo_dict.py (BK -> Z2 taper -> XBK -> binary PUBO,
via the local xbk_pipeline.py, so conventions match by construction).

Outputs
-------
  hubbard2d_big_pubos.py       flat  <name>_H_pubo / _C_pubo / _lambda_star /
                               _P_pubo  (model_pubos.py style)
  hubbard2d_big_pubo_dict.py   PUBOS = {name: {label, n_vars, sector, p,
                               H, C, P, lambda, E_xbk, E_exact, ...}}
                               (model_pubo_dict.py style)

THE CENTRAL SIZE FACT (exact, from build_xbk_ising's own factor rules)
----------------------------------------------------------------------
Every cross-replica (j != k) factor -- X, Y, Z and identity alike --
carries exactly 2 monomials, so each (Pauli term, cross pair) expands to
exactly 2^n monomials, and the counting PUBO alone is

    C(x) = r + sum_{j!=k} +/- prod_i (1 + Z_i^j Z_i^k)/2
         = exactly 2^n + 1 distinct monomials  (r = 2).

  hubbard2d 2x2 : n =  6 tapered  -> |C| =        65   (the NHOIM benchmark)
  hubbard2d 4x4 : n ~ 30 tapered  -> |C| ~ 1.1e9   H: T*2*2^n ~ 1e12
  hubbard2d 4x6 : n ~ 46 tapered  -> |C| ~ 7.0e13  H: T*2*2^n ~ 1e17

So the EXPANDED dicts for 4x4 / 4x6 cannot exist on any machine (PB-EB of
RAM); the exact PUBOs are delivered in FACTORED product form instead:

    H(x) = sum_j sum_k S_j S_k sum_t c_t  prod_i f_i(s),   s = 1 - 2x,

with every factor affine in 1-2 spins.  eval_H(x) / eval_C(x) below evaluate
them exactly in O(T * r^2 * n) -- the pointwise oracle an annealer-style
solver (SA / NHOIM / PT) actually requires.  The flat output file embeds
self-contained evaluators (no imports), keyed like the dict entries.

Validation: the same code path reproduces the hubbard2d 2x2 benchmark entry
(sector 0, p = 0, 12 vars) EXACTLY against model_pubo_dict.py before the big
instances are built.
"""
import argparse
import math
import sys
import time

import numpy as np

import xbk_pipeline as xp


R = 2
MATERIALIZE_CAP = 5_000_000        # monomial budget for explicit expansion


# ═══════════════════════════════════════════════════════════════════════════
# Factored form + exact evaluators (layout: phys spin = i + n*j, s = 1-2x)
# ═══════════════════════════════════════════════════════════════════════════

def factored_xbk(terms, n, r, p):
    """[(pref, j, k, [(site, char), ...])], pref = S_j S_k c_t.  Same-copy
    blocks of X/Y-containing terms vanish and are omitted."""
    Sp = [-1.0 if j < p else 1.0 for j in range(r)]
    fac = []
    for j in range(r):
        for k in range(r):
            sign = Sp[j] * Sp[k]
            for coeff, pd in terms:
                if j == k and any(c in ("X", "Y") for c in pd.values()):
                    continue
                fac.append((sign * coeff, j, k,
                            sorted((int(i), c) for i, c in pd.items())))
    return fac


def eval_H(fac, n, x):
    """Exact H(x) from the factored form (complex-safe; Y pairs -> real)."""
    s = 1.0 - 2.0 * np.asarray(x, dtype=float)
    tot = 0.0 + 0.0j
    for pref, j, k, sites in fac:
        acc = complex(pref)
        if j == k:
            for i, _c in sites:                    # only Z survives same-copy
                acc *= s[i + n * j]
        else:
            touched = set()
            for i, c in sites:
                a, b = s[i + n * j], s[i + n * k]
                touched.add(i)
                if c == "X":
                    acc *= 0.5 * (1.0 - a * b)
                elif c == "Y":
                    acc *= 0.5j * (b - a)
                else:
                    acc *= 0.5 * (a + b)
                if acc == 0.0:
                    break
            if acc != 0.0:
                for i in range(n):                 # identity sites
                    if i not in touched:
                        acc *= 0.5 * (1.0 + s[i + n * j] * s[i + n * k])
                        if acc == 0.0:
                            break
        tot += acc
    return tot.real


def eval_C(n, r, p, x):
    """Exact C(x): signed overlaps of the replica bitstrings."""
    Sp = [-1.0 if j < p else 1.0 for j in range(r)]
    xs = np.asarray(x, dtype=int)
    tot = float(r)
    for j in range(r):
        for k in range(r):
            if j != k and np.array_equal(xs[n*j:n*(j+1)], xs[n*k:n*(k+1)]):
                tot += Sp[j] * Sp[k]
    return tot


# ═══════════════════════════════════════════════════════════════════════════
# Vectorised PUBO evaluation over all bitstrings (small n_vars only)
# ═══════════════════════════════════════════════════════════════════════════

def eval_pubo_all(pubo, nv):
    """Vector of PUBO values over all 2^nv bitstrings (bit b of mask = x_b)."""
    masks = np.arange(1 << nv, dtype=np.int64)
    X = ((masks[:, None] >> np.arange(nv)) & 1).astype(bool)
    out = np.zeros(1 << nv)
    for k, c in pubo.items():
        if not k:
            out += c
        else:
            out += c * X[:, list(k)].all(axis=1)
    return out


def brute_lambda(H, C, nv):
    hv, cv = eval_pubo_all(H, nv), eval_pubo_all(C, nv)
    ok = np.abs(cv) > 1e-12
    ratio = hv[ok] / cv[ok]
    i = int(np.argmin(ratio))
    return float(ratio[i]), int(np.flatnonzero(ok)[i])


def exact_lambda_r2(terms, n):
    """Exact lambda* = E_xbk(r=2) without touching the PUBO:
    the r=2 XBK manifold is psi = s1|b1> + s2|b2>, s in {+-1}, so

        lambda* = min( min_i H_ii,
                       min_{i<j} (H_ii + H_jj)/2 - |Re H_ij| )

    and the H_ij != 0 pairs come straight from the sparse sector matrix.
    Feasible whenever 2^n fits in memory (n <= ~16)."""
    import openfermion as of
    from openfermion.ops import QubitOperator
    op = QubitOperator()
    for c, pd in terms:
        s = " ".join(f"{ch}{i}" for i, ch in sorted(pd.items()))
        op += QubitOperator(s, c)
    Hs = of.get_sparse_operator(op, n_qubits=n).tocoo()
    d = np.real(Hs.diagonal() if hasattr(Hs, "diagonal")
                else Hs.tocsr().diagonal())
    lam = float(d.min())
    two = np.partition(d, 1)[:2]
    lam = min(lam, float(two.mean()))              # Re H_ij = 0 pairs
    off = Hs.row != Hs.col
    if off.any():
        i, j = Hs.row[off], Hs.col[off]
        v = np.abs(np.real(Hs.data[off]))
        lam = min(lam, float(((d[i] + d[j]) / 2.0 - v).min()))
    return lam


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline stages (identical calls to export_pubo_dict.py)
# ═══════════════════════════════════════════════════════════════════════════

def fast_ising_to_binary(ising, nv):
    """Spin->binary change of basis as a per-variable butterfly over the full
    coefficient vector (s_v = 1-2x_v  =>  A + s_v B = (A+B) - 2 x_v B).
    Exact same result as xp._ising_to_binary, but O(nv * 2^nv) instead of
    O(|monomials| * 2^degree).  Needs nv <= ~24 (2^nv float64 array)."""
    arr = np.zeros(1 << nv)
    for S, c in ising.items():
        m = 0
        for i in S:
            m |= (1 << i)
        arr[m] += c
    for v in range(nv):
        a3 = arr.reshape(-1, 2, 1 << v)            # axis 1 = bit v
        B = a3[:, 1, :].copy()
        a3[:, 0, :] += B
        a3[:, 1, :] = -2.0 * B
    out = {}
    for m in np.flatnonzero(np.abs(arr) > 1e-12):
        m = int(m)
        out[tuple(b for b in range(nv) if (m >> b) & 1)] = float(arr[m])
    return out


def build_sectors(fermion_H, taper):
    Hq = xp.bk_transform(fermion_H)
    sectors = xp.z2_taper(Hq) if taper else [Hq]
    out = []
    for si, sec in enumerate(sectors):
        terms, n = xp._qubitop_to_terms(sec)
        if n:
            out.append((si, terms, n))
    return out


def materialize(terms, n, r, p, fast=True):
    nv = n * r
    if fast and nv <= 24:
        Hb = fast_ising_to_binary(xp.build_xbk_ising(terms, n, r, p), nv)
        Cb = fast_ising_to_binary(xp.build_counting(n, r, p), nv)
        return Hb, Cb
    Hb = {tuple(sorted(k)): float(v) for k, v in
          xp._ising_to_binary(xp.build_xbk_ising(terms, n, r, p)).items()}
    Cb = {tuple(sorted(k)): float(v) for k, v in
          xp._ising_to_binary(xp.build_counting(n, r, p)).items()}
    return Hb, Cb


def exact_counts(T, n, r):
    """(C monomials exact for r=2, H cross-monomial upper bound)."""
    return (2 ** n) + 1, T * r * (r - 1) * (2 ** n)


# ═══════════════════════════════════════════════════════════════════════════
# Validation vs the NHOIM benchmark inputs
# ═══════════════════════════════════════════════════════════════════════════

def load_reference():
    import importlib.util
    for path in ("/Users/nishithreen/Downloads/model_pubo_dict.py",
                 "model_pubo_dict.py"):
        try:
            spec = importlib.util.spec_from_file_location("mpd_ref", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.PUBOS, path
        except Exception:
            continue
    return None, None


def dicts_equal(A, B, tol=1e-9):
    keys = set(A) | set(B)
    return all(abs(A.get(k, 0.0) - B.get(k, 0.0)) < tol for k in keys)


def validate_2x2():
    print("== validation: hubbard2d 2x2 (sector 0, p 0) vs NHOIM inputs ==")
    ref, path = load_reference()
    secs = build_sectors(xp.hubbard2d(2, 1.0, 4.0), taper=True)
    si, terms, n = secs[0]
    Hb, Cb = materialize(terms, n, R, 0)
    Hs, Cs = materialize(terms, n, R, 0, fast=False)
    assert dicts_equal(Hb, Hs) and dicts_equal(Cb, Cs), \
        "fast butterfly spin->binary transform disagrees with xp._ising_to_binary"
    nv = n * R
    lam, argmask = brute_lambda(Hb, Cb, nv)
    lam2 = exact_lambda_r2(terms, n)
    assert abs(lam - lam2) < 1e-9, f"pair-formula lambda* {lam2} != brute {lam}"
    print(f"   sector {si} p=0: n_vars={nv}  |H|={len(Hb)} |C|={len(Cb)}  "
          f"brute-force lambda* = {lam:.6f} (pair formula agrees: {lam2:.6f})")
    # spot-check factored evaluators against the expanded dicts
    fac = factored_xbk(terms, n, R, 0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.integers(0, 2, nv)
        hd = sum(c * (np.prod([x[i] for i in k]) if k else 1.0)
                 for k, c in Hb.items())
        cd = sum(c * (np.prod([x[i] for i in k]) if k else 1.0)
                 for k, c in Cb.items())
        assert abs(eval_H(fac, n, x) - hd) < 1e-8, "factored H mismatch"
        assert abs(eval_C(n, R, 0, x) - cd) < 1e-8, "factored C mismatch"
    print("   factored evaluators == expanded dicts on 200 random bitstrings")
    if ref is None:
        print("   (model_pubo_dict.py not found -- skipped dict comparison)")
        return abs(lam - (-2.0)) < 1e-6
    e = ref["hubbard2d"]
    okH = dicts_equal(Hb, e["H"]); okC = dicts_equal(Cb, e["C"])
    okl = abs(lam - e["lambda"]) < 1e-6
    print(f"   vs {path}: H match={okH}  C match={okC}  "
          f"lambda match={okl} (ref {e['lambda']})")
    return okH and okC and okl


# ═══════════════════════════════════════════════════════════════════════════
# Output writers (mirroring the two NHOIM input formats)
# ═══════════════════════════════════════════════════════════════════════════

EVAL_SNIPPET = '''
# --- self-contained exact evaluators for the factored entries ---------------
# x: list/array of n_vars ints in {0,1}; layout: variable (i + n*j) is logical
# qubit i of replica j; s = 1-2x.  P(x; lam) = H(x) - lam*C(x).
def eval_H_factored(fac, n, x):
    s = [1.0 - 2.0*int(b) for b in x]
    tot_re = tot_im = 0.0
    for pref, j, k, sites in fac:
        re, im = float(pref), 0.0
        if j == k:
            for i, _c in sites:
                re, im = re * s[i + n*j], im * s[i + n*j]
        else:
            touched = set()
            for i, c in sites:
                a, b = s[i + n*j], s[i + n*k]
                touched.add(i)
                if c == "X":
                    f = 0.5*(1.0 - a*b); re, im = re*f, im*f
                elif c == "Y":
                    g = 0.5*(b - a); re, im = -im*g, re*g
                else:
                    f = 0.5*(a + b); re, im = re*f, im*f
                if re == 0.0 and im == 0.0:
                    break
            if re != 0.0 or im != 0.0:
                for i in range(n):
                    if i not in touched:
                        f = 0.5*(1.0 + s[i + n*j]*s[i + n*k])
                        re, im = re*f, im*f
                        if re == 0.0 and im == 0.0:
                            break
        tot_re += re; tot_im += im
    return tot_re

def eval_C_factored(n, r, p, x):
    S = [-1.0 if j < p else 1.0 for j in range(r)]
    tot = float(r)
    for j in range(r):
        for k in range(r):
            if j != k and list(x[n*j:n*(j+1)]) == list(x[n*k:n*(k+1)]):
                tot += S[j]*S[k]
    return tot
'''


def write_outputs(entries, out_flat, out_dict):
    with open(out_flat, "w") as f:
        f.write("# Auto-generated XBK PUBO dicts (r=2), hubbard2d large "
                "instances.  x_i in {0,1}; key ()=const.\n"
                "# Format identical to model_pubos.py.  Where the expanded "
                "dict is impossible (2^n scaling,\n# see counts), the exact "
                "PUBO is given in FACTORED form + evaluators below.\n\n")
        for e in entries:
            nm = e["name"]
            f.write(f"\n# === {nm}  ({e['label']}) ===\n")
            f.write(f"# subproblem: sector={e['sector']} p={e['p']} "
                    f"n_vars={e['n_vars']}  n_qubits_tapered={e['n']}  "
                    f"pauli_terms={e['n_pauli']}\n")
            f.write(f"# exact monomial counts: |C| = 2^{e['n']}+1 = "
                    f"{e['cnt_C']:.4g};  |H| cross <= {e['cnt_H']:.4g}\n")
            if e.get("E_exact") is not None:
                f.write(f"# exact (Lanczos ED) E0 = {e['E_exact']:.8f}\n")
            if e["H"] is not None:
                f.write(f"{nm}_H_pubo = {e['H']!r}\n")
                f.write(f"{nm}_C_pubo = {e['C']!r}\n")
                f.write(f"{nm}_lambda_star = {e['lambda']!r}\n")
                if e["lambda"] is not None:
                    P = dict(e["H"])
                    for k, c in e["C"].items():
                        P[k] = P.get(k, 0.0) - e["lambda"] * c
                    P = {k: v for k, v in P.items() if abs(v) > 1e-12}
                    f.write(f"{nm}_P_pubo = {P!r}"
                            "  # = H - lambda_star*C, min=0\n")
            else:
                f.write(f"{nm}_H_pubo = None  # NOT MATERIALIZABLE: "
                        f"~{e['cnt_H']:.3g} monomials\n")
                f.write(f"{nm}_C_pubo = None  # NOT MATERIALIZABLE: "
                        f"{e['cnt_C']:.3g} monomials\n")
                f.write(f"{nm}_lambda_star = None  # unknown at this size\n")
                f.write(f"{nm}_n = {e['n']}\n{nm}_r = {R}\n"
                        f"{nm}_p = {e['p']}\n")
                f.write(f"{nm}_H_factored = {e['factored']!r}\n")
        f.write(EVAL_SNIPPET)
    print(f"wrote {out_flat}")

    with open(out_dict, "w") as f:
        f.write('"""Auto-generated XBK PUBO dictionary (r=2), hubbard2d '
                'large instances.\nFormat identical to model_pubo_dict.py; '
                'H/C/P are None where the expansion cannot exist\n(exact '
                'counts in the entry); the factored form is exact -- see '
                'hubbard2d_pubos.py\nfor evaluators.\n"""\n\n')
        f.write(f"R = {R}\n\nPUBOS = ")
        out = {}
        for e in entries:
            P = None
            if e["H"] is not None and e["lambda"] is not None:
                P = dict(e["H"])
                for k, c in e["C"].items():
                    P[k] = P.get(k, 0.0) - e["lambda"] * c
                P = {k: v for k, v in P.items() if abs(v) > 1e-12}
            out[e["name"]] = dict(
                label=e["label"], n_vars=e["n_vars"], sector=e["sector"],
                p=e["p"], H=e["H"], C=e["C"], P=P,
                n_qubits=e["n"], n_pauli_terms=e["n_pauli"],
                monomials_C_exact=e["cnt_C"], monomials_H_upper=e["cnt_H"],
                factored=None if e["H"] is not None else e["factored"],
                E_exact=e.get("E_exact"),
                **{"lambda": e["lambda"]}, E_xbk=e["lambda"])
        f.write(repr(out) + "\n")
    print(f"wrote {out_dict}")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-taper", action="store_true")
    ap.add_argument("--U", type=float, default=4.0)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--E4x4", type=float, default=None,
                    help="exact (Lanczos) E0 for the 4x4 OBC instance")
    ap.add_argument("--include-ladder", action="store_true",
                    help="also emit materializable 2x3 / 2x4 rungs")
    ap.add_argument("--out-flat", default="hubbard2d_big_pubos.py")
    ap.add_argument("--out-dict", default="hubbard2d_big_pubo_dict.py")
    args = ap.parse_args()

    if not validate_2x2():
        sys.exit("VALIDATION FAILED -- not generating large instances")

    jobs = [("hubbard2d_4x4", "4x4  U/t=4 half-fill (OBC)",
             xp.hubbard2d(4, args.t, args.U), args.E4x4, (4, 4))]
    jobs.append(("hubbard2d_4x6", "4x6  U/t=4 half-fill (OBC)",
                 xp.hubbard2d_rect(4, 6, args.t, args.U, periodic=False),
                 None, (4, 6)))
    if args.include_ladder:
        jobs.insert(0, ("hubbard2d_2x4", "2x4  U/t=4 half-fill (OBC)",
                        xp.hubbard2d_rect(2, 4, args.t, args.U,
                                          periodic=False), None, (2, 4)))
        jobs.insert(0, ("hubbard2d_2x3", "2x3  U/t=4 half-fill (OBC)",
                        xp.hubbard2d_rect(2, 3, args.t, args.U,
                                          periodic=False), None, (2, 3)))

    entries = []
    for name, label, Hf, e_exact, (Lx, Ly) in jobs:
        if e_exact is None and Lx * Ly <= 12:
            try:                       # exact Lanczos E0 for the small rungs
                from hubbard2d_ed import hubbard_E0
                N = Lx * Ly
                e_exact = hubbard_E0(Lx, Ly, args.t, args.U, "obc",
                                     N // 2, N // 2, verbose=False)
            except Exception:
                pass
        print(f"\n== {name}  ({label}) ==")
        t0 = time.time()
        try:
            secs = build_sectors(Hf, taper=not args.no_taper)
        except Exception as exc:
            print(f"   taper failed ({exc!r}) -> untapered")
            secs = build_sectors(Hf, taper=False)
        print(f"   built {len(secs)} sector(s) in {time.time()-t0:.1f}s")
        # sector 0 mirrors the benchmark convention (ground sector was 0 for
        # every model in model_pubo_dict.py); include p = 0 and 1.
        si, terms, n = secs[0]
        T = len(terms)
        cnt_C, cnt_H = exact_counts(T, n, R)
        for p in range(math.ceil(R / 2)):          # p = 0 (p=1 = global sign)
            ent = dict(name=name if p == 0 else f"{name}_p{p}", label=label,
                       sector=si, p=p, n=n, n_vars=n * R, n_pauli=T,
                       cnt_C=float(cnt_C), cnt_H=float(cnt_H),
                       E_exact=e_exact, H=None, C=None, factored=None,
                       **{"lambda": None})
            if cnt_C + cnt_H <= MATERIALIZE_CAP and n * R <= 24:
                Hb, Cb = materialize(terms, n, R, p)
                lam = exact_lambda_r2(terms, n) if n <= 16 else None
                ent.update(H=Hb, C=Cb, **{"lambda": lam})
                print(f"   p={p}: materialized |H|={len(Hb)} |C|={len(Cb)} "
                      f"n_vars={n*R}  lambda*={lam}")
            else:
                fac = factored_xbk(terms, n, R, p)
                ent["factored"] = [(float(c), j, k, sites)
                                   for c, j, k, sites in fac]
                print(f"   p={p}: FACTORED ONLY  n_vars={n*R}  "
                      f"|C|=2^{n}+1={cnt_C:.4g}  |H|<={cnt_H:.4g}  "
                      f"factored terms={len(fac)}")
            entries.append(ent)

    write_outputs(entries, args.out_flat, args.out_dict)
    print("\nSummary: expanded PUBOs exist only where 2^n fits in RAM; "
          "4x4/4x6 entries are exact\nfactored PUBOs + evaluators "
          "(pointwise oracle), with the impossibility counts recorded.")


if __name__ == "__main__":
    main()
