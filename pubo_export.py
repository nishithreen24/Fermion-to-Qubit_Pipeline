"""
pubo_export.py — export the XBK mapping of every model as PUBO python dicts.

A PUBO is  { tuple(sorted binary-variable indices) : float coefficient }  over
x_i in {0,1}, with the constant term under the empty-tuple key ().

XBK maps a Hamiltonian to a FRACTIONAL binary program
        E0^(r) = min_x  H'(x) / C(x)
split over (symmetry-sector, p-projection) subproblems; the model energy is the
min over subproblems.  We therefore emit, per subproblem:
    H_pubo : numerator <psi|H|psi>          (higher-order PUBO)
    C_pubo : counting  <psi|psi>            (higher-order PUBO)
and, for the ground subproblem, the single Dinkelbach-combined PUBO
    P_pubo = H_pubo - lam* * C_pubo   (lam* = E0^(r); global min = 0 at argmin).

Variables are indexed 0..n*r-1, where copy j, tapered-qubit i -> index i + n*j.
"""
import math
from collections import defaultdict
import xbk_pipeline as xp
from openfermion import QubitOperator


def ising_to_pubo(ising):
    """Ising {frozenset:float} over s=±1  ->  PUBO {tuple:float} over x in {0,1}."""
    binary = xp._ising_to_binary(ising)
    return {tuple(sorted(k)): float(v) for k, v in binary.items()}


def eval_pubo(poly, x):
    """Evaluate a PUBO dict at bit-list x."""
    tot = 0.0
    for k, c in poly.items():
        p = 1.0
        for i in k:
            p *= x[i]
        tot += c * p
    return tot


def xbk_pubo_family(H, r=2, taper=True):
    """Full XBK PUBO mapping: list of subproblem records
       {sector, p, n_vars, H_pubo, C_pubo}."""
    spin = isinstance(H, QubitOperator)
    H_q = H if spin else xp.bk_transform(H)
    sectors = xp.z2_taper(H_q) if taper else [H_q]
    recs = []
    for si, sec in enumerate(sectors):
        terms, n = xp._qubitop_to_terms(sec)
        if n == 0:
            const = float(next((c for c, pd in terms if not pd), 0.0)) if terms else 0.0
            recs.append(dict(sector=si, p=None, n_vars=0,
                             H_pubo={(): const}, C_pubo={(): 1.0}))
            continue
        for p in range(math.ceil(r / 2) + 1):
            H_is = xp.build_xbk_ising(terms, n, r, p)
            C_is = xp.build_counting(n, r, p)
            if not C_is:
                continue
            recs.append(dict(sector=si, p=p, n_vars=n * r,
                             H_pubo=ising_to_pubo(H_is),
                             C_pubo=ising_to_pubo(C_is)))
    return recs


def brute_ratio(rec):
    """Exact min_x H/C over x in {0,1}^n_vars (n_vars<=~22). Returns (E, argmin)."""
    nv = rec['n_vars']
    if nv == 0:
        return rec['H_pubo'][()], ()
    Hp, Cp = rec['H_pubo'], rec['C_pubo']
    best, bx = math.inf, None
    for mask in range(1 << nv):
        x = [(mask >> b) & 1 for b in range(nv)]
        Cv = eval_pubo(Cp, x)
        if abs(Cv) < 1e-12:
            continue
        v = eval_pubo(Hp, x) / Cv
        if v < best:
            best, bx = v, tuple(x)
    return best, bx


def combined_pubo(rec, lam, tol=1e-12):
    """Dinkelbach single PUBO  P = H - lam*C  (min = 0 at the XBK ground config)."""
    P = defaultdict(float)
    for k, c in rec['H_pubo'].items():
        P[k] += c
    for k, c in rec['C_pubo'].items():
        P[k] -= lam * c
    return {k: v for k, v in P.items() if abs(v) > tol}


def ground_subproblem(recs):
    """Brute-force the ground (sector,p) subproblem; return (rec, energy, argmin)."""
    best = None
    for rec in recs:
        e, x = brute_ratio(rec)
        if best is None or e < best[1] - 1e-9:
            best = (rec, e, x)
    return best


def max_degree(poly):
    return max((len(k) for k in poly), default=0)


# representative small instance per model (n*r <= 12 so brute force is exact)
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
    import pprint
    R = 2
    out_py = ["# Auto-generated XBK PUBO dicts (r=%d).  x_i in {0,1}; key ()=const.\n" % R]
    print(f"{'model':10s} {'instance':24s} {'n_vars':>6} {'degH':>4} "
          f"{'#H':>6} {'#C':>5} {'XBK E':>10} {'exactED':>10}")
    for name, H, label in instances():
        recs = xbk_pubo_family(H, r=R)
        rec, e, x = ground_subproblem(recs)
        Ex = xp.exact_ground_energy(H)
        nH = len(rec['H_pubo'])
        P = combined_pubo(rec, e)
        print(f"{name:10s} {label:24s} {rec['n_vars']:>6} "
              f"{max_degree(rec['H_pubo']):>4} {nH:>6} {len(rec['C_pubo']):>5} "
              f"{e:>10.5f} {Ex:>10.5f}")
        # write full python-literal dicts for this model's ground subproblem
        out_py.append(f"\n# === {name}  ({label}) ===")
        out_py.append(f"# ground subproblem: sector={rec['sector']} p={rec['p']} "
                      f"n_vars={rec['n_vars']}  XBK E0^(2)={e:.6f} (exact ED {Ex:.6f})")
        out_py.append(f"{name}_H_pubo = {rec['H_pubo']!r}")
        out_py.append(f"{name}_C_pubo = {rec['C_pubo']!r}")
        out_py.append(f"{name}_lambda_star = {e!r}")
        out_py.append(f"{name}_P_pubo = {P!r}  # = H - lambda_star*C, min=0")
    with open("model_pubos.py", "w") as f:
        f.write("\n".join(out_py) + "\n")
    print("\nfull python-literal dicts written to model_pubos.py")
