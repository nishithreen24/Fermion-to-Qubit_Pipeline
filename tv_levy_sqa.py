#!/usr/bin/env python3
"""
tv_levy_sqa.py -- Levy-style ground-state ENERGY of the 1D t-V chain by
Simulated Quantum Annealing, via two independent quantum Monte Carlo estimators:

  * SSE  : Stochastic Series Expansion (Sandvik), heat-bath directed loops.
           No Trotter error.  E0 = -<n>/beta + N_bonds * C.
  * PIMC : discrete-time (Suzuki-Trotter) world-line path integral, checkerboard
           decomposition, local world-line updates, run as an anneal in the
           number-conserving sector.  Has controllable Trotter error O(dtau^2).

Both simulate the SAME sign-problem-free XXZ image of the t-V chain (see
tv_gs_common.py) and must agree with each other and with ED as beta -> inf.
This replaces the ED "cheat" in tv_levy.py, which only diagonalised H to read
off E0 and then measured the adiabatic gap as a cost proxy.

Spin (S=1/2) form used by both:
    H = V sum_<ij> S^z_i S^z_j  -  t sum_<ij> (S^+_i S^-_j + S^-_i S^+_j)
(up == occupied).  Off-diagonal matrix element = -t <= 0  => stoquastic.

Usage:
    python3 tv_levy_sqa.py --L 8 --t 1.0 --V 2.0 --beta 32 --sweeps 20000 \
                           --method both --csv tv_levy_sqa_L8.csv
"""
import argparse
import numpy as np

import tv_gs_common as C

try:                                # C kernels (tv_kernels.c via ctypes); the
    import tv_kernels as K          # pure-Python below stays as the validated
    HAVE_C = K.HAVE_C               # reference and fallback.
except Exception:
    K = None
    HAVE_C = False


def _binned_err(x, nb=20):
    """Mean and binned standard error of a sample series."""
    x = np.asarray(x, dtype=float)
    binsz = max(1, len(x) // nb)
    binned = np.array([x[k*binsz:(k+1)*binsz].mean()
                       for k in range(len(x) // binsz)])
    return float(x.mean()), float(binned.std(ddof=1) / np.sqrt(len(binned)))


def _use_c(engine):
    if engine == "py":
        return False
    if engine == "c" and not HAVE_C:
        raise RuntimeError("C kernel requested but unavailable: "
                           f"{getattr(K, 'BUILD_ERROR', 'import failed')}")
    return HAVE_C



# ============================================================================
# SSE  --  Stochastic Series Expansion with heat-bath directed loops
# ============================================================================
# Operator encoding in the string:  0 = identity.
# else op = ((b+1) << 1) | typ ,  typ: 0 diagonal, 1 off-diagonal, b = bond idx.
def _op_bond(op):  return (op >> 1) - 1
def _op_type(op):  return op & 1
def _mk_op(b, typ): return ((b + 1) << 1) | typ


def sse_run(L, t, V, beta, warmup, measures, seed, Cconst=None, engine="auto"):
    """Return (E_mean, E_err) for the physical t-V energy at inverse temp beta."""
    rng = np.random.default_rng(seed)
    bonds = C.bonds_obc(L)
    Nb = len(bonds)
    # diagonal bond weights (constant C shifts H by -Nb*C; any C >= V/4 is exact)
    if Cconst is None:
        Cconst = V / 4.0 + t                      # W_par = t > 0, good mixing
    W_apar = Cconst + V / 4.0                      # antiparallel diagonal weight
    W_par  = Cconst - V / 4.0                      # parallel diagonal weight
    W_off  = t                                     # off-diagonal weight
    if min(W_apar, W_par, W_off) < 0:
        raise ValueError("negative SSE weight; need C >= V/4")

    # ---- C fast path: same update rules compiled in tv_kernels.c ----
    if _use_c(engine):
        spins = rng.integers(0, 2, L).astype(np.int64) * 2 - 1
        Mcut = max(20, 4 * L)
        opstr = np.zeros(Mcut, dtype=np.int64)
        bonds_arr = np.asarray(bonds, dtype=np.int64)
        n, call, done = 0, 0, 0
        chunk = 200                     # grow Mcut between chunks, as the
        while done < warmup:            # Python version does between steps
            m = min(chunk, warmup - done)
            n, ns = K.sse_sweep(L, bonds_arr, beta, W_par, W_apar, W_off,
                                Mcut, opstr, spins, n, 0, m,
                                np.uint64(seed) * np.uint64(1000003) + np.uint64(call))
            call += 1; done += m
            nmax = int(ns.max()) if len(ns) else n
            need = nmax + nmax // 3 + 2
            if need > Mcut:
                tmp = np.zeros(need, dtype=np.int64)
                tmp[:Mcut] = opstr
                opstr, Mcut = tmp, need
        n, ns = K.sse_sweep(L, bonds_arr, beta, W_par, W_apar, W_off,
                            Mcut, opstr, spins, n, 0, measures,
                            np.uint64(seed) * np.uint64(1000003) + np.uint64(call))
        return _binned_err(-ns / beta + Nb * Cconst)

    spins = rng.integers(0, 2, L).astype(np.int64) * 2 - 1   # +/-1
    Mcut = max(20, 4 * L)
    opstring = np.zeros(Mcut, dtype=np.int64)
    n = 0

    def Wdiag_state(b, st):
        i, j = bonds[b]
        return W_par if st[i] == st[j] else W_apar

    def diagonal_update():
        nonlocal n
        st = spins.copy()
        for p in range(Mcut):
            op = opstring[p]
            if op == 0:                                   # try insert diagonal
                b = rng.integers(Nb)
                w = Wdiag_state(b, st)
                if w > 0 and rng.random() < (beta * Nb * w) / (Mcut - n):
                    opstring[p] = _mk_op(b, 0); n += 1
            elif _op_type(op) == 0:                       # try remove diagonal
                b = _op_bond(op)
                w = Wdiag_state(b, st)
                if w <= 0 or rng.random() < (Mcut - n + 1) / (beta * Nb * w):
                    opstring[p] = 0; n -= 1
            else:                                         # off-diag: propagate
                b = _op_bond(op); i, j = bonds[b]
                st[i] = -st[i]; st[j] = -st[j]

    def vertex_weight(si, sj, typ):
        if typ == 0:
            return W_par if si == sj else W_apar
        return W_off if si != sj else 0.0

    def directed_loops():
        # gather operator positions in order
        pos = np.nonzero(opstring)[0]
        if len(pos) == 0:
            # no operators: free spins, randomise
            for q in range(L):
                if rng.random() < 0.5:
                    spins[q] = -spins[q]
            return
        nv = len(pos)
        # leg spins: 4 per vertex  (0=bot_i,1=bot_j,2=top_i,3=top_j)
        legs = np.empty(4 * nv, dtype=np.int64)
        vbond = np.empty(nv, dtype=np.int64)
        st = spins.copy()
        # first pass: assign bottom/top leg spins as the string propagates
        last_leg = -np.ones(L, dtype=np.int64)      # last (top) leg touching a site
        first_leg = -np.ones(L, dtype=np.int64)     # first (bottom) leg touching a site
        link = -np.ones(4 * nv, dtype=np.int64)
        for v, p in enumerate(pos):
            op = opstring[p]; b = _op_bond(op); typ = _op_type(op)
            i, j = bonds[b]; vbond[v] = b
            bi, bj = st[i], st[j]
            # link bottom legs to previous top legs on same site
            for site, leg in ((i, 4 * v + 0), (j, 4 * v + 1)):
                if last_leg[site] >= 0:
                    link[leg] = last_leg[site]; link[last_leg[site]] = leg
                else:
                    first_leg[site] = leg
            if typ == 1:                              # off-diagonal flips both
                st[i] = -st[i]; st[j] = -st[j]
            ti, tj = st[i], st[j]
            legs[4 * v + 0], legs[4 * v + 1] = bi, bj
            legs[4 * v + 2], legs[4 * v + 3] = ti, tj
            last_leg[i] = 4 * v + 2
            last_leg[j] = 4 * v + 3
        # close the periodic boundary: connect last top leg to first bottom leg
        for site in range(L):
            if last_leg[site] >= 0:
                link[last_leg[site]] = first_leg[site]
                link[first_leg[site]] = last_leg[site]

        legsite = [bonds[vbond[v]][k % 2] for v in range(nv)
                   for k in range(4)]  # site of each leg (i,j,i,j)

        # Directed-loop (heat-bath exit) updates.  Correctness note: each loop is
        # run to completion (returns to its start leg); we only bound the NUMBER
        # of complete loops per MC step via a cumulative-length target ~ 2*<n>,
        # which keeps cost O(n) instead of O(n^2) without truncating any loop.
        visited_any = np.zeros(L, dtype=bool)
        target = max(4, 2 * nv)
        done = 0
        maxsteps = 2000 * (nv + 1)     # safety only; valid loops close on their own             # per-loop safety cap only
        while done < target:
            e0 = int(rng.integers(4 * nv))
            e = e0
            steps = 0
            while True:
                v4 = 4 * (e >> 2)
                el = e - v4
                # entrance leg flipped; evaluate the 4 candidate exits inline
                c0 = legs[v4]; c1 = legs[v4 + 1]; c2 = legs[v4 + 2]; c3 = legs[v4 + 3]
                if el == 0:   c0 = -c0
                elif el == 1: c1 = -c1
                elif el == 2: c2 = -c2
                else:         c3 = -c3
                w0 = w1 = w2 = w3 = 0.0
                # exit x flips leg x; vertex legal if (bot==top diag) or (both flipped off-diag)
                # x=0
                a0, a1, a2, a3 = -c0, c1, c2, c3
                if a0 == a2 and a1 == a3:      w0 = W_par if a0 == a1 else W_apar
                elif a2 == -a0 and a3 == -a1:  w0 = W_off if a0 != a1 else 0.0
                # x=1
                a0, a1, a2, a3 = c0, -c1, c2, c3
                if a0 == a2 and a1 == a3:      w1 = W_par if a0 == a1 else W_apar
                elif a2 == -a0 and a3 == -a1:  w1 = W_off if a0 != a1 else 0.0
                # x=2
                a0, a1, a2, a3 = c0, c1, -c2, c3
                if a0 == a2 and a1 == a3:      w2 = W_par if a0 == a1 else W_apar
                elif a2 == -a0 and a3 == -a1:  w2 = W_off if a0 != a1 else 0.0
                # x=3
                a0, a1, a2, a3 = c0, c1, c2, -c3
                if a0 == a2 and a1 == a3:      w3 = W_par if a0 == a1 else W_apar
                elif a2 == -a0 and a3 == -a1:  w3 = W_off if a0 != a1 else 0.0
                tot = w0 + w1 + w2 + w3
                if tot <= 0:
                    break
                r = rng.random() * tot
                if r < w0:      x = 0
                elif r < w0+w1: x = 1
                elif r < w0+w1+w2: x = 2
                else:           x = 3
                # commit: entrance flip already in c*, plus exit flip on leg x
                legs[v4] = c0; legs[v4+1] = c1; legs[v4+2] = c2; legs[v4+3] = c3
                if x == 0:   legs[v4] = -legs[v4]
                elif x == 1: legs[v4+1] = -legs[v4+1]
                elif x == 2: legs[v4+2] = -legs[v4+2]
                else:        legs[v4+3] = -legs[v4+3]
                visited_any[legsite[v4 + el]] = True
                visited_any[legsite[v4 + x]] = True
                steps += 1
                exit_leg = v4 + x
                if exit_leg == e0:
                    break
                e = link[exit_leg]
                if e == e0 or steps > maxsteps:
                    break
            done += steps

        # rebuild opstring types and |alpha> from updated legs
        for v, p in enumerate(pos):
            b = vbond[v]
            bi_, bj_, ti_, tj_ = legs[4*v:4*v+4]
            typ = 0 if (bi_ == ti_ and bj_ == tj_) else 1
            opstring[p] = _mk_op(b, typ)
        # update the stored state |alpha> from the bottom legs of first ops / free spins
        for site in range(L):
            fl = first_leg[site]
            if fl >= 0:
                spins[site] = legs[fl]
            elif visited_any[site]:
                pass
        # free (untouched) spins already randomised only when no ops; here flip 50%
        touched = set()
        for v in range(nv):
            b = vbond[v]; touched.add(bonds[b][0]); touched.add(bonds[b][1])
        for site in range(L):
            if site not in touched and rng.random() < 0.5:
                spins[site] = -spins[site]

    # ---- warmup with adaptive Mcut ----
    for it in range(warmup):
        diagonal_update()
        directed_loops()
        if n + n // 3 + 2 > Mcut:                        # grow cutoff
            new = n + n // 3 + 2
            tmp = np.zeros(new, dtype=np.int64)
            tmp[:Mcut] = opstring
            opstring = tmp; Mcut = new

    # ---- measurement ----
    ns = np.empty(measures)
    for it in range(measures):
        diagonal_update()
        directed_loops()
        ns[it] = n
    E = -ns / beta + Nb * Cconst
    # binned standard error
    nb_ = 20
    binsz = max(1, measures // nb_)
    binned = np.array([E[k*binsz:(k+1)*binsz].mean()
                       for k in range(measures // binsz)])
    return float(E.mean()), float(binned.std(ddof=1) / np.sqrt(len(binned)))


# ============================================================================
# PIMC  --  discrete-time (Suzuki-Trotter) checkerboard world-line
# ============================================================================
# Trotter:  e^{-b H} ~ [e^{-dt H_A} e^{-dt H_B}]^M,  dt = b/M,  H_A/H_B = even/odd
# bonds.  2M half-step rows; every shaded plaquette is a FIXED 4-leg vertex whose
# leg spins are sampled by the SAME heat-bath directed loops as SSE, only the
# weights differ (Trotter values below).  No diagonal insert/remove update -- the
# expansion order (#plaquettes = M*N_bonds) is fixed by the space-time grid.
#
# 2-site propagator weights (z = 1-2*occ, "parallel" = equal occupations):
#   parallel  (00->00, 11->11) : e^{-dt V/4}
#   apar diag (10->10, 01->01) : e^{+dt V/4} cosh(dt t)
#   apar off  (10->01, 01->10) : e^{+dt V/4} sinh(dt t)      (all >= 0 => sign-free)
#
# Energy estimator (exact at finite M, O(dt^2) Trotter error):
#   E = -(1/M) * sum_plaq  d ln w_plaq / d(dt)
#     = -(1/M)[ N_par(-V/4) + N_ad(V/4 + t*tanh(dt t)) + N_ao(V/4 + t*coth(dt t)) ]
def pimc_run(L, t, V, beta, M, warmup, measures, seed, engine="auto"):
    """Return (E_mean, E_err) at inverse temp beta with M Trotter steps."""
    rng = np.random.default_rng(seed)
    bonds = C.bonds_obc(L)
    Nb = len(bonds)
    dt = beta / M
    W_par = np.exp(-dt * V / 4.0)
    W_ad  = np.exp(dt * V / 4.0) * np.cosh(dt * t)
    W_ao  = np.exp(dt * V / 4.0) * np.sinh(dt * t)
    g_par = -V / 4.0
    g_ad  = V / 4.0 + t * np.tanh(dt * t)
    g_ao  = V / 4.0 + t / np.tanh(dt * t)          # coth

    # ---- build the fixed vertex list over 2M checkerboard rows ----
    rows = 2 * M
    vs, vb, vi, vj = [], [], [], []
    for s in range(rows):
        for b, (i, j) in enumerate(bonds):
            if b % 2 == s % 2:                     # bond active on this row
                vs.append(s); vb.append(b); vi.append(i); vj.append(j)
    nv = len(vs)
    legsite = [ (vi[v] if k % 2 == 0 else vj[v]) for v in range(nv) for k in range(4) ]

    # ---- vertical links: for each site, chain plaquettes in imaginary time ----
    link = -np.ones(4 * nv, dtype=np.int64)
    per_site = {x: [] for x in range(L)}           # (row, top_leg, bot_leg)
    for v in range(nv):
        i, j, s = vi[v], vj[v], vs[v]
        per_site[i].append((s, 4 * v + 2, 4 * v + 0))   # top_i, bot_i
        per_site[j].append((s, 4 * v + 3, 4 * v + 1))   # top_j, bot_j
    for x in range(L):
        seq = sorted(per_site[x])                  # by row
        m = len(seq)
        for k in range(m):
            top_leg = seq[k][1]
            bot_next = seq[(k + 1) % m][2]          # periodic in imaginary time
            link[top_leg] = bot_next
            link[bot_next] = top_leg

    # ---- initial world line: constant half-filling config on every slice ----
    occ0 = np.zeros(L, dtype=np.int64)
    occ0[rng.choice(L, L // 2, replace=False)] = 1
    z0 = 1 - 2 * occ0                              # +/-1
    legs = np.empty(4 * nv, dtype=np.int64)
    for v in range(nv):
        zi, zj = z0[vi[v]], z0[vj[v]]
        legs[4*v+0] = zi; legs[4*v+1] = zj
        legs[4*v+2] = zi; legs[4*v+3] = zj          # all-diagonal start

    def directed_loops():
        if nv == 0:
            return
        target = max(4, 2 * nv)
        done = 0
        maxsteps = 2000 * (nv + 1)     # safety only; valid loops close on their own
        while done < target:
            e0 = int(rng.integers(4 * nv))
            e = e0
            steps = 0
            while True:
                v4 = 4 * (e >> 2)
                el = e - v4
                c0 = legs[v4]; c1 = legs[v4 + 1]; c2 = legs[v4 + 2]; c3 = legs[v4 + 3]
                if el == 0:   c0 = -c0
                elif el == 1: c1 = -c1
                elif el == 2: c2 = -c2
                else:         c3 = -c3
                w0 = w1 = w2 = w3 = 0.0
                a0, a1, a2, a3 = -c0, c1, c2, c3
                if a0 == a2 and a1 == a3:      w0 = W_par if a0 == a1 else W_ad
                elif a2 == -a0 and a3 == -a1:  w0 = W_ao if a0 != a1 else 0.0
                a0, a1, a2, a3 = c0, -c1, c2, c3
                if a0 == a2 and a1 == a3:      w1 = W_par if a0 == a1 else W_ad
                elif a2 == -a0 and a3 == -a1:  w1 = W_ao if a0 != a1 else 0.0
                a0, a1, a2, a3 = c0, c1, -c2, c3
                if a0 == a2 and a1 == a3:      w2 = W_par if a0 == a1 else W_ad
                elif a2 == -a0 and a3 == -a1:  w2 = W_ao if a0 != a1 else 0.0
                a0, a1, a2, a3 = c0, c1, c2, -c3
                if a0 == a2 and a1 == a3:      w3 = W_par if a0 == a1 else W_ad
                elif a2 == -a0 and a3 == -a1:  w3 = W_ao if a0 != a1 else 0.0
                tot = w0 + w1 + w2 + w3
                if tot <= 0:
                    break
                r = rng.random() * tot
                if r < w0:          x = 0
                elif r < w0+w1:     x = 1
                elif r < w0+w1+w2:  x = 2
                else:               x = 3
                legs[v4] = c0; legs[v4+1] = c1; legs[v4+2] = c2; legs[v4+3] = c3
                if x == 0:   legs[v4] = -legs[v4]
                elif x == 1: legs[v4+1] = -legs[v4+1]
                elif x == 2: legs[v4+2] = -legs[v4+2]
                else:        legs[v4+3] = -legs[v4+3]
                steps += 1
                exit_leg = v4 + x
                if exit_leg == e0:
                    break
                e = link[exit_leg]
                if e == e0 or steps > maxsteps:
                    break
            done += steps

    def measure():
        Npar = Nad = Nao = 0
        for v in range(nv):
            bi = legs[4*v]; bj = legs[4*v+1]; ti = legs[4*v+2]; tj = legs[4*v+3]
            if bi == ti and bj == tj:
                if bi == bj: Npar += 1
                else:        Nad += 1
            else:
                Nao += 1
        return -(g_par * Npar + g_ad * Nad + g_ao * Nao) / M

    for _ in range(warmup):
        directed_loops()
    Es = np.empty(measures)
    for it in range(measures):
        directed_loops()
        Es[it] = measure()
    nb_ = 20
    binsz = max(1, measures // nb_)
    binned = np.array([Es[k*binsz:(k+1)*binsz].mean()
                       for k in range(measures // binsz)])
    return float(Es.mean()), float(binned.std(ddof=1) / np.sqrt(len(binned)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--V", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=32.0)
    ap.add_argument("--M", type=int, default=64, help="PIMC Trotter steps")
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--measures", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--method", choices=["sse", "pimc", "both"], default="both")
    ap.add_argument("--trotter_scan", type=str, default=None,
                    help="comma list of M values; runs PIMC at each, fits E vs "
                         "dtau^2 -> dtau=0, writes tv_levy_pimc_trotter_L{L}.csv")
    ap.add_argument("--trotter_cut", type=float, default=0.35,
                    help="only dtau<=cut enter the linear-in-dtau^2 fit "
                         "(coarse dtau carry dtau^4 curvature and bias the intercept)")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    E0 = C.ed_reference(args.L, args.t, args.V) if args.L <= 16 else None
    print(f"# t-V SQA:  L={args.L} t={args.t} V={args.V} beta={args.beta}")
    if E0 is not None:
        print(f"# ED E0 = {E0:.6f}")

    # ---- PIMC Trotter extrapolation mode ----
    if args.trotter_scan:
        Ms = [int(x) for x in args.trotter_scan.split(",")]
        dt2, Es, ers = [], [], []
        for M in Ms:
            e, er = pimc_run(args.L, args.t, args.V, args.beta, M,
                             args.warmup, args.measures, args.seed)
            dtau = args.beta / M
            dt2.append(dtau**2); Es.append(e); ers.append(er)
            print(f"#   M={M:>4} dtau={dtau:.4f}  E={e:.6f} +/- {er:.6f}")
        # weighted linear fit E = E0 + a*dtau^2, ASYMPTOTIC regime only (dtau<=cut):
        # coarse dtau have small MC error but large dtau^4 curvature, so including
        # them in a pure dtau^2 fit biases the intercept.
        dt2 = np.array(dt2); Es = np.array(Es); ers = np.array(ers)
        dtau_arr = np.sqrt(dt2)
        mask = dtau_arr <= args.trotter_cut
        if mask.sum() < 2:                       # fall back to the two finest
            mask = np.argsort(dt2)[:max(2, mask.sum())]
            mask = np.isin(np.arange(len(dt2)), mask)
        w = 1.0 / ers[mask]**2
        A = np.vstack([np.ones(mask.sum()), dt2[mask]]).T
        coef, *_ = np.linalg.lstsq(A * w[:, None], Es[mask] * w, rcond=None)
        E_extrap = coef[0]
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        e_extrap = float(np.sqrt(cov[0, 0]))
        de = abs(E_extrap - E0) if E0 is not None else float('nan')
        print(f"# PIMC dtau->0 extrapolation (dtau<={args.trotter_cut}, "
              f"n={int(mask.sum())}): E0 = {E_extrap:.6f} +/- {e_extrap:.6f}"
              f"   |dE_ED| = {de:.6f}")
        if args.csv:
            meta = (f"L={args.L} t={args.t} V={args.V} beta={args.beta} "
                    f"E0_ed={E0 if E0 is not None else float('nan'):.6f} "
                    f"E0_extrap={E_extrap:.6f} err_extrap={e_extrap:.6f}")
            path = args.csv
            rows = [(m, args.beta / m, e, er) for m, e, er in zip(Ms, Es, ers)]
            C.write_csv(path, ["M", "dtau", "E", "err"], rows, meta)
            print(f"# wrote {path}")
        return

    Esse = errsse = Epi = errpi = None
    if args.method in ("sse", "both"):
        Esse, errsse = sse_run(args.L, args.t, args.V, args.beta,
                               args.warmup, args.measures, args.seed)
        de = abs(Esse - E0) if E0 is not None else float('nan')
        print(f"# SSE    E0 = {Esse:.6f} +/- {errsse:.6f}   |dE_ED| = {de:.6f}")
    if args.method in ("pimc", "both"):
        Epi, errpi = pimc_run(args.L, args.t, args.V, args.beta, args.M,
                              args.warmup, args.measures, args.seed)
        de = abs(Epi - E0) if E0 is not None else float('nan')
        print(f"# PIMC   E0 = {Epi:.6f} +/- {errpi:.6f}   |dE_ED| = {de:.6f}"
              f"   (M={args.M}, dtau={args.beta/args.M:.4f})")

    if args.csv:
        meta = (f"L={args.L} t={args.t} V={args.V} beta={args.beta} M={args.M} "
                f"E0_ed={E0 if E0 is not None else float('nan'):.6f}")
        rows = [("SSE", Esse if Esse is not None else float('nan'),
                 errsse if errsse is not None else float('nan')),
                ("PIMC", Epi if Epi is not None else float('nan'),
                 errpi if errpi is not None else float('nan'))]
        C.write_csv(args.csv, ["method", "E0", "err"], rows, meta)
        print(f"# wrote {args.csv}")


if __name__ == "__main__":
    main()
