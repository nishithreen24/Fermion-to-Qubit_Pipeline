/* tv_kernels.c -- C kernels for the 1D t-V SQA solvers (tv_levy_sqa.py).
 *
 * Ports the two hot inner loops to C, called via ctypes:
 *   sse_sweep()  : SSE diagonal update + heat-bath directed loops
 *   pimc_sweep() : fixed-vertex Trotter world-line directed loops
 *
 * The physics is a line-by-line port of the validated pure-Python version;
 * the pure-Python path remains in tv_levy_sqa.py as the reference fallback.
 *
 * Leg convention (4 legs per vertex): 0=bot_i 1=bot_j 2=top_i 3=top_j.
 * Spins are +/-1.  Operator encoding: 0 = identity, else ((b+1)<<1)|typ.
 *
 * RNG: xoshiro256** seeded via splitmix64, so runs are reproducible and
 * independent of NumPy's stream (results agree statistically, not bitwise).
 *
 * Build (done automatically by tv_kernels.py):
 *   cc -O3 -shared -fPIC -o libtvkernels.so tv_kernels.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- RNG: xoshiro256** ---------------- */
typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(rng_t *r, uint64_t seed) {
    uint64_t x = seed ? seed : 0x123456789ABCDEFULL;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&x);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

/* uniform double in [0,1) */
static inline double rng_uni(rng_t *r) {
    return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* uniform integer in [0,n) */
static inline int64_t rng_int(rng_t *r, int64_t n) {
    return (int64_t)(rng_uni(r) * (double)n);
}

/* ============================================================
 * Shared directed-loop core.
 *
 * Walks heat-bath directed loops over a fixed vertex list, given the three
 * vertex weights (parallel-diagonal, antiparallel-diagonal, off-diagonal).
 * SSE and PIMC differ ONLY in those three numbers, so one routine serves both.
 *
 * legs  : 4*nv leg spins (updated in place)
 * link  : 4*nv leg -> partner leg
 * nv    : number of vertices
 * Wpar/Wad/Wao : vertex weights
 * ============================================================ */
static void directed_loops(int64_t *legs, const int64_t *link, int64_t nv,
                           double Wpar, double Wad, double Wao, rng_t *rng) {
    if (nv <= 0) return;
    int64_t target = 2 * nv; if (target < 4) target = 4;
    int64_t maxsteps = 2000 * (nv + 1);   /* safety only; valid loops self-close */
    int64_t done = 0;

    while (done < target) {
        int64_t e0 = rng_int(rng, 4 * nv);
        int64_t e = e0;
        int64_t steps = 0;
        for (;;) {
            int64_t v4 = 4 * (e >> 2);
            int64_t el = e - v4;
            int64_t c0 = legs[v4], c1 = legs[v4+1], c2 = legs[v4+2], c3 = legs[v4+3];
            /* flip the entrance leg */
            if      (el == 0) c0 = -c0;
            else if (el == 1) c1 = -c1;
            else if (el == 2) c2 = -c2;
            else              c3 = -c3;

            double w[4]; int64_t a0, a1, a2, a3;
            /* exit x flips leg x; vertex legal if all-diagonal or both-flipped */
            for (int x = 0; x < 4; x++) {
                a0 = c0; a1 = c1; a2 = c2; a3 = c3;
                if      (x == 0) a0 = -a0;
                else if (x == 1) a1 = -a1;
                else if (x == 2) a2 = -a2;
                else             a3 = -a3;
                double wx = 0.0;
                if (a0 == a2 && a1 == a3)              wx = (a0 == a1) ? Wpar : Wad;
                else if (a2 == -a0 && a3 == -a1)       wx = (a0 != a1) ? Wao : 0.0;
                w[x] = wx;
            }
            double tot = w[0] + w[1] + w[2] + w[3];
            if (tot <= 0.0) break;

            double r = rng_uni(rng) * tot;
            int x;
            if      (r < w[0])               x = 0;
            else if (r < w[0]+w[1])          x = 1;
            else if (r < w[0]+w[1]+w[2])     x = 2;
            else                             x = 3;

            /* commit entrance flip + exit flip */
            legs[v4] = c0; legs[v4+1] = c1; legs[v4+2] = c2; legs[v4+3] = c3;
            legs[v4+x] = -legs[v4+x];

            steps++;
            int64_t exit_leg = v4 + x;
            if (exit_leg == e0) break;
            e = link[exit_leg];
            if (e == e0 || steps > maxsteps) break;
        }
        done += steps;
        if (steps == 0) done++;    /* guarantee progress on dead-end starts */
    }
}
/* ============================================================
 * SSE driver.
 *
 * Runs (warmup + measures) full MC steps.  Each step = diagonal update
 * (insert/remove diagonal operators, adaptive-free: Mcut fixed by caller)
 * followed by directed loops.  Returns the per-measurement operator count n
 * in nout[], from which Python forms E = -<n>/beta + Nb*C.
 *
 * bonds : 2*Nb  flattened (i,j) pairs
 * spins : L     +/-1, updated in place (carries state across calls)
 * opstr : Mcut  operator string, updated in place
 * Returns the final n (expansion order).
 * ============================================================ */
int64_t sse_sweep(int64_t L, const int64_t *bonds, int64_t Nb,
                  double beta, double Wpar, double Wad, double Woff,
                  int64_t Mcut, int64_t *opstr, int64_t *spins, int64_t n_in,
                  int64_t warmup, int64_t measures, double *nout,
                  uint64_t seed) {
    rng_t rng; rng_seed(&rng, seed);
    int64_t n = n_in;
    int64_t total = warmup + measures;

    int64_t *st        = (int64_t*)malloc(sizeof(int64_t) * (size_t)L);
    int64_t *last_leg  = (int64_t*)malloc(sizeof(int64_t) * (size_t)L);
    int64_t *first_leg = (int64_t*)malloc(sizeof(int64_t) * (size_t)L);
    int64_t *touched   = (int64_t*)malloc(sizeof(int64_t) * (size_t)L);
    /* worst case: every slot holds an operator */
    int64_t *pos   = (int64_t*)malloc(sizeof(int64_t) * (size_t)Mcut);
    int64_t *vbond = (int64_t*)malloc(sizeof(int64_t) * (size_t)Mcut);
    int64_t *legs  = (int64_t*)malloc(sizeof(int64_t) * (size_t)(4 * Mcut));
    int64_t *link  = (int64_t*)malloc(sizeof(int64_t) * (size_t)(4 * Mcut));
    if (!st || !last_leg || !first_leg || !touched || !pos || !vbond || !legs || !link) {
        free(st); free(last_leg); free(first_leg); free(touched);
        free(pos); free(vbond); free(legs); free(link);
        return -1;
    }

    for (int64_t it = 0; it < total; it++) {
        /* ---------- diagonal update ---------- */
        memcpy(st, spins, sizeof(int64_t) * (size_t)L);
        for (int64_t p = 0; p < Mcut; p++) {
            int64_t op = opstr[p];
            if (op == 0) {
                int64_t b = rng_int(&rng, Nb);
                int64_t i = bonds[2*b], j = bonds[2*b+1];
                double w = (st[i] == st[j]) ? Wpar : Wad;
                if (w > 0.0 && rng_uni(&rng) < (beta * (double)Nb * w) / (double)(Mcut - n)) {
                    opstr[p] = ((b + 1) << 1); n++;
                }
            } else if ((op & 1) == 0) {
                int64_t b = (op >> 1) - 1;
                int64_t i = bonds[2*b], j = bonds[2*b+1];
                double w = (st[i] == st[j]) ? Wpar : Wad;
                if (w <= 0.0 ||
                    rng_uni(&rng) < (double)(Mcut - n + 1) / (beta * (double)Nb * w)) {
                    opstr[p] = 0; n--;
                }
            } else {
                int64_t b = (op >> 1) - 1;
                int64_t i = bonds[2*b], j = bonds[2*b+1];
                st[i] = -st[i]; st[j] = -st[j];
            }
        }

        /* ---------- build the vertex list / linked legs ---------- */
        int64_t nv = 0;
        for (int64_t p = 0; p < Mcut; p++) if (opstr[p] != 0) pos[nv++] = p;

        if (nv == 0) {
            for (int64_t q = 0; q < L; q++)
                if (rng_uni(&rng) < 0.5) spins[q] = -spins[q];
        } else {
            for (int64_t x = 0; x < L; x++) { last_leg[x] = -1; first_leg[x] = -1; touched[x] = 0; }
            for (int64_t k = 0; k < 4 * nv; k++) link[k] = -1;
            memcpy(st, spins, sizeof(int64_t) * (size_t)L);

            for (int64_t v = 0; v < nv; v++) {
                int64_t op = opstr[pos[v]];
                int64_t b = (op >> 1) - 1, typ = op & 1;
                int64_t i = bonds[2*b], j = bonds[2*b+1];
                vbond[v] = b;
                touched[i] = 1; touched[j] = 1;
                int64_t bi = st[i], bj = st[j];
                /* link bottom legs to the previous top leg on the same site */
                int64_t site2[2] = {i, j};
                for (int k = 0; k < 2; k++) {
                    int64_t site = site2[k], leg = 4 * v + k;
                    if (last_leg[site] >= 0) {
                        link[leg] = last_leg[site];
                        link[last_leg[site]] = leg;
                    } else {
                        first_leg[site] = leg;
                    }
                }
                if (typ == 1) { st[i] = -st[i]; st[j] = -st[j]; }
                legs[4*v+0] = bi;    legs[4*v+1] = bj;
                legs[4*v+2] = st[i]; legs[4*v+3] = st[j];
                last_leg[i] = 4 * v + 2;
                last_leg[j] = 4 * v + 3;
            }
            /* close the periodic imaginary-time boundary */
            for (int64_t x = 0; x < L; x++) {
                if (last_leg[x] >= 0) {
                    link[last_leg[x]] = first_leg[x];
                    link[first_leg[x]] = last_leg[x];
                }
            }

            directed_loops(legs, link, nv, Wpar, Wad, Woff, &rng);

            /* rebuild operator types from the updated legs */
            for (int64_t v = 0; v < nv; v++) {
                int64_t typ = (legs[4*v+0] == legs[4*v+2] && legs[4*v+1] == legs[4*v+3]) ? 0 : 1;
                opstr[pos[v]] = ((vbond[v] + 1) << 1) | typ;
            }
            /* propagate |alpha> from the first bottom leg of each site */
            for (int64_t x = 0; x < L; x++) {
                if (first_leg[x] >= 0)      spins[x] = legs[first_leg[x]];
                else if (!touched[x] && rng_uni(&rng) < 0.5) spins[x] = -spins[x];
            }
        }

        if (it >= warmup) nout[it - warmup] = (double)n;
    }

    free(st); free(last_leg); free(first_leg); free(touched);
    free(pos); free(vbond); free(legs); free(link);
    return n;
}

/* ============================================================
 * PIMC driver.
 *
 * The vertex list and vertical links are FIXED by the space-time grid, so the
 * caller builds them once (in Python) and this routine only runs loops and
 * measures.  Per-measurement energy is written to Eout[].
 *
 * E = -(1/M)[ N_par*g_par + N_ad*g_ad + N_ao*g_ao ]
 * ============================================================ */
void pimc_sweep(int64_t nv, int64_t *legs, const int64_t *link,
                double Wpar, double Wad, double Wao,
                double g_par, double g_ad, double g_ao, double M,
                int64_t warmup, int64_t measures, double *Eout,
                uint64_t seed) {
    rng_t rng; rng_seed(&rng, seed);
    int64_t total = warmup + measures;
    for (int64_t it = 0; it < total; it++) {
        directed_loops(legs, link, nv, Wpar, Wad, Wao, &rng);
        if (it >= warmup) {
            int64_t Npar = 0, Nad = 0, Nao = 0;
            for (int64_t v = 0; v < nv; v++) {
                int64_t bi = legs[4*v], bj = legs[4*v+1];
                int64_t ti = legs[4*v+2], tj = legs[4*v+3];
                if (bi == ti && bj == tj) { if (bi == bj) Npar++; else Nad++; }
                else Nao++;
            }
            Eout[it - warmup] =
                -(g_par * (double)Npar + g_ad * (double)Nad + g_ao * (double)Nao) / M;
        }
    }
}

