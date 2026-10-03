"""
taper_report.py  — deterministic BK -> Z2 tapering dump (no XBK, no SA).

For each instance prints the quadruple that settles the tapered-qubit counts:
  1. untapered qubit count (right after BK encoding)
  2. number of independent stabiliser generators actually used by z2_taper
     (the _valid_stabilizers subset of the GF(2) nullspace)
  3. each generator as a dense Pauli string + the +/-1 eigenvalue fixed in the
     GROUND-state sector
  4. tapered qubit count n  (= untapered - #generators)

Run at the exact table points (half filling / eps_f=-U/2 / Delta=t).  Generic-
point controls at the bottom show the extra generator disappears off the
symmetric point.  Everything here is deterministic (ED per sector, no SA).
"""
from itertools import product as iproduct
import numpy as np
import xbk_pipeline as xp
from openfermion import taper_off_qubits, count_qubits, get_sparse_operator


def dense_pauli(qop, n):
    """Single-Pauli QubitOperator -> dense string over n qubits, e.g. 'ZIZI'."""
    term = next(t for t in qop.terms if t != ())
    chars = ['I'] * n
    for (idx, ch) in term:
        chars[idx] = ch
    return ''.join(chars)


def ground_sector_signs(qubit_H, generators):
    """Diagonalise every sign sector; return (signs, energy) of the ground one."""
    best_e, best_signs = np.inf, None
    for signs in iproduct([1, -1], repeat=len(generators)):
        signed = [signs[i] * generators[i] for i in range(len(generators))]
        sec = taper_off_qubits(qubit_H, signed)
        sec.compress()
        nq = count_qubits(sec)
        if nq == 0:
            e = float(np.real(sec.terms.get((), 0.0)))
        else:
            e = float(np.linalg.eigvalsh(
                get_sparse_operator(sec, nq).toarray())[0].real)
        if e < best_e - 1e-9:
            best_e, best_signs = e, signs
    return best_signs, best_e


def report(label, H_fermion, spin=False):
    H_q = H_fermion if spin else xp.bk_transform(H_fermion)
    n_bk = count_qubits(H_q)
    gens = xp._valid_stabilizers(H_q, xp._symmetry_generators(H_q))
    k = len(gens)
    print(f"\n### {label}")
    print(f"    untapered qubits (post-BK) : {n_bk}")
    print(f"    # stabiliser generators    : {k}")
    print(f"    tapered qubits n           : {n_bk - k}")
    if k == 0:
        print("    (no valid stabilisers; n unchanged)")
        return
    signs, ge = ground_sector_signs(H_q, gens)
    for i, g in enumerate(gens):
        print(f"      g{i+1}: {dense_pauli(g, n_bk):<{max(n_bk,4)}}"
              f"   sector eigenvalue = {signs[i]:+d}")
    print(f"    ground-sector energy (ED)  : {ge:.6f}")


if __name__ == "__main__":
    print("=" * 70)
    print("  BK -> Z2 tapering generator dump  (symmetric / table points)")
    print("=" * 70)

    report("tv1d  L=2  V/t=2",                    xp.tv1d(2, 1.0, 2.0))
    report("tv1d  L=3  V/t=2",                    xp.tv1d(3, 1.0, 2.0))
    report("tv1d  L=4  V/t=2",                    xp.tv1d(4, 1.0, 2.0))
    report("tv2d  2x2  V/t=2",                    xp.tv2d(2, 1.0, 2.0))
    report("hubbard1d L=2 U/t=4 (half-fill)",     xp.hubbard1d(2, 1.0, 4.0))
    report("hubbard1d L=3 U/t=4 (half-fill)",     xp.hubbard1d(3, 1.0, 4.0))
    report("hubbard1d L=4 U/t=4 (half-fill)",     xp.hubbard1d(4, 1.0, 4.0))
    report("hubbard2d 2x2 U/t=4 (NO -U/2 shift)", xp.hubbard2d(2, 1.0, 4.0))
    report("siam  N_bath=2  U/t=4 (eps_d=-U/2)",  xp.siam(2, U=4.0))
    report("siam  N_bath=3  U/t=4 (eps_d=-U/2)",  xp.siam(3, U=4.0))
    report("siam  N_bath=4  U/t=4 (eps_d=-U/2)",  xp.siam(4, U=4.0))
    report("pam   N_b=2  U/t=4 (eps_f=-U/2)",     xp.pam(2, 1.0, 1.0, 4.0))
    report("kitaev L=4  Delta/t=1 (self-dual)",   xp.kitaev(4, 1.0, 1.0))
    report("kitaev L=6  Delta/t=1 (self-dual)",   xp.kitaev(6, 1.0, 1.0))
    report("kitaev L=4  Delta/t=0.5 (generic)",   xp.kitaev(4, 1.0, 0.5))

    print("\n" + "-" * 70)
    print("  CONTROLS (off the symmetric point -> particle-hole should vanish)")
    print("-" * 70)
    report("hubbard1d L=2 U/t=4 mu=0 (NOT half-fill)",
           xp.hubbard1d(2, 1.0, 4.0, mu=0.0))
    report("hubbard1d L=4 U/t=4 mu=0 (NOT half-fill)",
           xp.hubbard1d(4, 1.0, 4.0, mu=0.0))
