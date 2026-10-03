"""
Dinkelbach bug: direct algebraic proof.

Strategy:
  1. Build H_ising and C_ising (pre-quad) on n_phys=6 physical spins.
  2. Build H_quad/C_quad (post-quad) on n_total ~136 spins.
  3. Find the minimising physical spin configuration s* by brute force
     (only 2^6 = 64 configurations).
  4. Extend s* to n_total spins by setting all auxiliary spins to +1
     (an arbitrary, penalty-violating choice).
  5. Compare:
       (a) _eval(H_ising, s_phys) / _eval(C_ising, s_phys)  <- correct E0
       (b) _eval(H_qi, s_full)    / _eval(C_qi, s_full)     <- buggy E0
  If the bug exists, (a) and (b) will differ — the quadratised polys
  include penalty terms (of order M ~hundreds) that are non-zero when
  auxiliary spins are not at their Rosenberg-prescribed values.
"""
import sys
from itertools import product as iproduct
sys.path.insert(0, '/mnt/project')
sys.path.insert(0, '/home/claude/deliverable')
import numpy as np
from openfermion import get_sparse_operator
import pipeline_base as pb
from tv_model import tV_hamiltonian_1d


def _eval(poly, spins):
    E = 0.0
    for mono, c in poly.items():
        p = 1.0
        for i in mono: p *= spins[i]
        E += c * p
    return E


def build_C_serial(n_qubits):
    from collections import defaultdict
    partial = defaultdict(complex)
    for basis in iproduct([0,1], repeat=n_qubits):
        inner = defaultdict(complex)
        for copy_j in range(1, 3):
            p = {frozenset(): 1.0+0j}
            for k, b_k in enumerate(basis):
                ph = 2*k + (copy_j-1)
                p = pb._poly_mul(p, {frozenset(): 0.5+0j,
                                     frozenset([ph]): float((-1)**b_k)*0.5+0j})
            for m, v in p.items(): inner[m] += v
        for m, v in pb._poly_mul(dict(inner), dict(inner)).items(): partial[m] += v
    return pb._poly_clean(dict(partial))


# ── build the system ──────────────────────────────────────────────────────────
Hf         = tV_hamiltonian_1d(3, 1.0, 2.0, 0.3, False)
E_exact    = np.linalg.eigvalsh(get_sparse_operator(Hf, n_qubits=3).toarray())[0]
bk_terms, n_qubits = pb.bk_transform(Hf)
C_ising    = build_C_serial(n_qubits)
n_phys     = n_qubits * 2   # = 6
M          = 12.0 * max(abs(c) for c, _ in bk_terms)
sector_signs = [+1]*n_qubits

H_ising    = pb.build_Hprime(bk_terms, n_qubits, sector_signs)

from collections import defaultdict
next_id = n_phys
H_quad, next_id = pb.quadratize(pb._ising_to_binary(H_ising), M, next_id)
C_quad, next_id = pb.quadratize(pb._ising_to_binary(C_ising),  M, next_id)
H_qi = pb._binary_to_ising(H_quad)
C_qi = pb._binary_to_ising(C_quad)

print(f"n_phys={n_phys}, n_total={next_id}, M={M:.1f}")
print(f"Physical: H has {len(H_ising)} terms, C has {len(C_ising)} terms")
print(f"Quad:     H has {len(H_qi)} terms, C has {len(C_qi)} terms (incl. {next_id-n_phys} aux spins)")
print(f"E_exact = {E_exact:.6f}")
print()

# ── brute-force the optimal physical spin config ───────────────────────────────
best_ratio = float('inf')
best_config = None
for bits in iproduct([-1,1], repeat=n_phys):
    s = {i: bits[i] for i in range(n_phys)}
    h = _eval(H_ising, s)
    c = _eval(C_ising, s)
    if abs(c) < 1e-9: continue
    r = h/c
    if r < best_ratio:
        best_ratio = r
        best_config = dict(s)

print(f"Best physical ratio (brute force, sector 0): {best_ratio:.6f}")
print(f"Physical spins: {[best_config[i] for i in range(n_phys)]}")
print()

# ── extend to full spin vector with aux spins = +1 (violates constraints) ─────
# This simulates what happens when SA returns a configuration where auxiliary
# spins are not at their Rosenberg-prescribed values.
s_full_aux_plus1 = dict(best_config)
for i in range(n_phys, next_id):
    s_full_aux_plus1[i] = +1   # all aux = +1: wrong for most terms

s_full_aux_minus1 = dict(best_config)
for i in range(n_phys, next_id):
    s_full_aux_minus1[i] = -1  # all aux = -1: also wrong

# ── evaluate both ways ─────────────────────────────────────────────────────────
# (a) FIXED: physical polys, physical spins only
H_phys_val = _eval(H_ising, best_config)
C_phys_val = _eval(C_ising, best_config)
E_fixed    = H_phys_val / C_phys_val if abs(C_phys_val)>1e-9 else float('inf')

# (b) BUGGY: quad polys, full spin vector including auxiliaries
H_bug_p1   = _eval(H_qi, s_full_aux_plus1)
C_bug_p1   = _eval(C_qi, s_full_aux_plus1)
E_buggy_p1 = H_bug_p1/C_bug_p1 if abs(C_bug_p1)>1e-9 else float('inf')

H_bug_m1   = _eval(H_qi, s_full_aux_minus1)
C_bug_m1   = _eval(C_qi, s_full_aux_minus1)
E_buggy_m1 = H_bug_m1/C_bug_m1 if abs(C_bug_m1)>1e-9 else float('inf')

print("="*60)
print("Direct evaluation comparison (same physical spins, aux varied):")
print(f"  E_exact          = {E_exact:.6f}")
print(f"  E_fixed          = {E_fixed:.6f}   <- H_phys/C_phys (correct)")
print(f"  E_buggy(aux=+1)  = {E_buggy_p1:.6f}   <- H_quad/C_quad (penalty contaminated)")
print(f"  E_buggy(aux=-1)  = {E_buggy_m1:.6f}   <- H_quad/C_quad (penalty contaminated)")
print()
print(f"  H_phys  = {H_phys_val:.6f}   C_phys  = {C_phys_val:.6f}")
print(f"  H_quad(+1) = {H_bug_p1:.2f},   C_quad(+1) = {C_bug_p1:.2f}   (penalty terms O(M={M:.0f}) visible)")
print(f"  H_quad(-1) = {H_bug_m1:.2f},   C_quad(-1) = {C_bug_m1:.2f}")
print()

bug_visible = abs(E_buggy_p1 - E_fixed) > 0.01 or abs(E_buggy_m1 - E_fixed) > 0.01
print(f"  Bug is visible: {bug_visible}")
print(f"  Fix correct:    {abs(E_fixed - E_exact) < 0.5}")   # within 50% for sector 0 only