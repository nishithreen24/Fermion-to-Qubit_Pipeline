#!/usr/bin/env python3
"""
tv_yip.py  --  Yip-style Variational Quantum Annealing (VarQA) for the spinless
1D t-V chain, run classically to MEASURE the cost that sets the required number
of samples.

Method (Yip, Yeter-Aydeniz, Dong, arXiv:2503.15473):
  * # logical qubits = # spin-orbitals (here L for the spinless chain),
  * the annealer is used as a SAMPLER: anneal a parameterised fully-connected
    Ising model, draw S bitstring samples {m_s}, and build a trial state
        |psi(theta)> = sum_m eps_m * sqrt(freq(m)/S) |m>
    from the empirical bitstring frequencies (eps_m are sign/phase parameters
    refined by the outer optimiser),
  * evaluate <psi|H|psi> classically from the samples and optimise over theta.

The key difference from XBK: <H> is ESTIMATED FROM SAMPLES, not built as an
Ising polynomial -- so there is no 2^n normalisation blow-up.  The exponential
instead relocates into the SAMPLE COUNT S needed to resolve the amplitudes of a
ground state spread over ~PR (participation ratio) configurations.

WHAT THIS SCRIPT REPORTS (the honest cost quantity):
  * PR = 1/sum|c|^4 of the exact ground state (how many configs carry weight),
  * energy error |E_var(S) - E0| as a function of sample count S, so the
    S-needed-for-target-accuracy can be read off directly,
  * a fit of S_required vs system size L (via tv_scaling.slurm) -- the actual
    answer to "does Yip scale past ED".

Honest scope: this is an IDEALISED VarQA upper bound.  We give the optimiser the
exact-GS amplitude MAGNITUDES (so we isolate the SAMPLING cost, not the outer-
loop optimisation difficulty), draw S multinomial samples from |c_m|^2, rebuild
the state from empirical frequencies with exact signs, and measure the resulting
energy error.  Real VarQA also pays an outer-optimisation cost on top; this
lower-bounds the sample requirement, which is exactly the quantity asked about.

Usage:
    python3 tv_yip.py --L 3 --t 1.0 --V 2.0 --seeds 40 --csv tv_yip_L3.csv
"""
import argparse
import numpy as np
from openfermion import FermionOperator, jordan_wigner, get_sparse_operator


def tv1d(L, t=1.0, V=1.0, periodic=False):
    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic and L >= 3:
        bonds.append((L - 1, 0))
    H = FermionOperator()
    for (i, j) in bonds:
        H += FermionOperator(((i, 1), (j, 0)), -t)
        H += FermionOperator(((j, 1), (i, 0)), -t)
        H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)
        H += FermionOperator(((i, 1), (i, 0)), -V / 2)
        H += FermionOperator(((j, 1), (j, 0)), -V / 2)
        H += FermionOperator((), V / 4)
    return H


def energy_from_state(psi, Hmat):
    psi = psi / np.linalg.norm(psi)
    return float(np.real(psi.conj() @ (Hmat @ psi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--V", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=40,
                    help="independent sampling repeats per S (for mean/std)")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    Hf = tv1d(args.L, args.t, args.V)
    Hmat = get_sparse_operator(jordan_wigner(Hf), n_qubits=args.L).astype(complex)
    dim = 2 ** args.L

    w, v = np.linalg.eigh(Hmat.toarray())
    E0 = w[0].real
    c = v[:, 0].astype(complex)
    c /= np.linalg.norm(c)
    p = np.abs(c) ** 2                      # target sampling distribution
    signs = np.sign(c.real + 1e-30)         # exact signs (t-V GS is real)
    PR = 1.0 / np.sum(p ** 2)

    print(f"# t-V Yip VarQA (sampling cost):  L={args.L} t={args.t} V={args.V}"
          f"  dim={dim}")
    print(f"# ED ground energy E0 = {E0:.6f}   participation ratio PR = {PR:.3f}"
          f"   (PR/dim = {PR/dim:.4f})")
    print(f"#   {'S':>10} {'mean|dE|':>12} {'std|dE|':>12} {'S/PR':>8}")

    # sample-count ladder (geometric); report energy error vs S
    S_list = [int(x) for x in np.unique(np.round(
        np.geomspace(max(8, dim // 4), 4000 * dim, 12)).astype(int))]
    rng = np.random.default_rng(0)
    rows = []
    for S in S_list:
        errs = []
        for _ in range(args.seeds):
            counts = rng.multinomial(S, p)
            freq = counts / S
            amp = signs * np.sqrt(freq)          # rebuild state from samples
            if np.linalg.norm(amp) < 1e-12:
                continue
            Evar = energy_from_state(amp, Hmat)
            errs.append(abs(Evar - E0))
        errs = np.array(errs)
        rows.append((S, errs.mean(), errs.std(), S / PR))
        print(f"#   {S:>10} {errs.mean():>12.5f} {errs.std():>12.5f} {S/PR:>8.1f}",
              flush=True)

    # S required to reach |dE| <= eps, for a few eps
    print("# sample count to reach target accuracy (interp):")
    S_arr = np.array([r[0] for r in rows], float)
    e_arr = np.array([r[1] for r in rows], float)
    for eps in (0.1, 0.05, 0.01):
        below = np.where(e_arr <= eps)[0]
        Sneed = int(S_arr[below[0]]) if len(below) else None
        tag = f"{Sneed:,}" if Sneed else f">{int(S_arr[-1]):,}"
        print(f"#   eps={eps:<5}  S_required = {tag}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("S,mean_absdE,std_absdE,S_over_PR"
                    f"  # L={args.L} t={args.t} V={args.V} E0={E0:.6f} PR={PR:.4f}"
                    f" dim={dim}\n")
            for (S, m, s, sp) in rows:
                f.write(f"{S},{m:.6f},{s:.6f},{sp:.4f}\n")
        print(f"# wrote {args.csv}")


if __name__ == "__main__":
    main()
