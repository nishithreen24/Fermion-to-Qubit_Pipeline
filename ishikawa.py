"""
ishikawa.py  (NEW FILE — does not modify xbk_pipeline.py)
========================================================
Ishikawa higher-order -> quadratic reduction of a binary polynomial, as a
drop-in alternative to xbk_pipeline.quadratize (Rosenberg).

Ishikawa, "Transformation of General Binary MRF Minimization to the
First-Order Case," IEEE TPAMI 33(6):1234 (2011).

Input/output match xbk_pipeline.quadratize:
    binary  : dict { frozenset(var indices) -> float }   over x in {0,1}
    returns : (poly, next_id) with every monomial of degree <= 2.

Key differences from Rosenberg:
  * Single pass (no cascade): each degree-d monomial is reduced directly.
  * NO penalty parameter M: the reductions are min-selection identities,
    f(x) = min_w g(x,w), so minimizing g over all variables equals
    minimizing f.  (Rosenberg needs a large M to enforce w = x_a x_b.)
  * Auxiliaries per monomial: 1 for a negative coeff (any degree),
    floor((d-1)/2) for a positive coeff.

Formulas (x_i in {0,1}, S = sum_{i in mono} x_i):
  Negative   -|c| prod_i x_i = min_w |c| * w (d-1-S)
                              = |c|(d-1) w - |c| sum_i w x_i.
  Positive   +c  prod_i x_i = c [ sum_{j=1..n_d} w_j ( c_{d,j}(2j - S) - 1 )
                                  + sum_{i<k} x_i x_k ],
             n_d = floor((d-1)/2),  c_{d,j} = 1 if (d odd and j=n_d) else 2.
"""

from collections import defaultdict


def quadratize_ishikawa(binary, next_id, tol=1e-14):
    """Reduce a binary polynomial to degree <= 2 (Ishikawa). No penalty needed.

    Returns (poly, next_id).  poly is dict{frozenset->float} over x in {0,1}
    plus the auxiliary variables [original next_id .. returned next_id).
    """
    out = defaultdict(float)
    for mono, c in binary.items():
        if abs(c) < tol:
            continue
        d = len(mono)
        if d <= 2:
            out[mono] += c
            continue

        sl = sorted(mono)
        if c < 0.0:
            # one auxiliary; |c|(d-1) w - |c| sum_i w x_i
            ac = -c
            w = next_id
            next_id += 1
            out[frozenset([w])] += ac * (d - 1)
            for i in sl:
                out[frozenset([w, i])] += -ac
        else:
            # positive: S_2 pairwise + floor((d-1)/2) auxiliaries
            nd = (d - 1) // 2
            for a in range(d):
                for b in range(a + 1, d):
                    out[frozenset([sl[a], sl[b]])] += c
            for j in range(1, nd + 1):
                cdj = 1.0 if (d % 2 == 1 and j == nd) else 2.0
                w = next_id
                next_id += 1
                out[frozenset([w])] += c * (cdj * (2 * j) - 1.0)
                for i in sl:
                    out[frozenset([w, i])] += -c * cdj
    # drop numerical zeros
    return {m: v for m, v in out.items() if abs(v) > tol}, next_id
