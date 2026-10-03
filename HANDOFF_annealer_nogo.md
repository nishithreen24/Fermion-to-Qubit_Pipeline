# Handoff: t-V/Hubbard annealing project → no-go paper(s)
*Self-contained context file for other Claude chats. State as of 2026-07-29.*
*Owner: Nishith. Local code: `/Users/nishithreen/Desktop/Pipeline/` (Python 3.14 venv). Cluster: `/home/shassan/XBKReal` (SLURM, kshort).*

## 1. Project state (done and validated)

The project replaced ED "cheats" in three quantum-annealing-pipeline simulations of the 1D spinless t-V model (OBC, PH-symmetric; JW image = XXZ chain, stoquastic in 1D → sign-free QMC):

| File | Method | Status |
|---|---|---|
| `tv_levy_sqa.py` | Levy → SQA via **SSE** (no Trotter error) + **PIMC** (world-line, dτ→0 extrapolated, fit only dτ≤0.35) | Validated vs ED: SSE within <0.008 at L=4,6,8; PIMC extrap within ~0.004 |
| `tv_yip_sa.py` | Yip → SA-sampled subspace diagonalization (selected-CI; hop-closure + Lanczos) | Exact for L≤14 (sector < max_support, fully enumerated) |
| `tv_gs_common.py` | shared: ED reference, sector enumeration, sparse H | self-test passes L=3–8 |
| `tv_gs_scaling.slurm`, `plot_gs_energies.py` | cluster driver + 2×2 comparison figure (Okabe-Ito/viridis) | ready |
| `tv_kernels.c` / `tv_kernels.py` | C/ctypes acceleration (numba impossible: Py3.14 no llvmlite wheel). C kernel compiles; SSE fast path wired; PIMC C path NOT wired; NOT validated | **ABANDONED by user decision — do not resume unless asked** |

Key numbers: ED L=4 E0=-3.2404 (V=2,t=1,β=32); 1D is sign-free ONLY because JW strings are local in 1D.

## 2. Decisions locked by the user

1. **qanneal library (user's PDF manual) rejected**: all five engines (sa/sqa/sqapt/ctpimc/sqa_chi) target *diagonal* Ising H_cl with a single-site transverse-field driver ramped to zero. Cannot express XX+YY hopping in the FINAL Hamiltonian → wrong problem class for t-V/Hubbard ground states.
2. **2D Hubbard**: JW trick does NOT carry to 2D (sign problem; strings non-local). Compact encodings (Derby-Klassen etc.) restore *locality*, not *stoquasticity* — still no sign-free QMC, still not annealer-embeddable. Correct classical methods: ED/Lanczos (exact, ≤~20 sites) or DQMC (sign-free at half filling on bipartite lattices). Target chosen: **4×4 square, half filling, U/t=4**.
3. **Current focus = paper(s), not more code.**

## 3. The paper thesis and where the gap is

Proposed thesis: *strongly correlated fermions (2D Hubbard) cannot be efficiently treated on quantum annealers / higher-order Ising solvers regardless of pipeline (BK-XBK, Levy, Yip); resource needs exceed hardware by many orders of magnitude.*

**Cannot be an impossibility proof** (XBK is a working exact mapping at tiny sizes). Correct form: quantitative no-go / resource accounting.

### Verified literature anchors (fetched 2026-07-29)
- **XBK**: Xia, Bian, Kais, *J. Phys. Chem. B* 122, 3384 (2018), [arXiv:1706.00271](https://arxiv.org/abs/1706.00271). r replicas per qubit map σx/σy to diagonal ops; Rosenberg quadratization to 2-local. Follow-ups note r must grow to express multi-determinant CI states; symmetry-adapted variants (Desroches & Dong) reduce qubits but not scaling.
- **QAE on hardware**: Teplukhin et al., *Sci. Rep.* 10, 20753 (2020) ([nature.com](https://www.nature.com/articles/s41598-020-77315-4)); follow-up work states current annealing tech gives exponential scaling for "inherently quantum" problems, new couplers needed (verify exact source in full text).
- **Closest prior resource accounting**: Copenhaver, Wasserman, Wehefritz-Kaufmann, *J. Chem. Phys.* 153, 214103 (2020), [arXiv:2009.10779](https://arxiv.org/abs/2009.10779) — XBK & QAE on H₃⁺/H₂O; "outperformed by modern classical algorithms", "scaling of the resource requirements remains a challenge". Molecular only, no lattice accounting, no pipeline taxonomy. **Does not scoop the lattice/taxonomy paper.**
- **Complexity anchor**: Schuch & Verstraete, *Nat. Phys.* 5, 732 (2009), [arXiv:0712.0483](https://arxiv.org/abs/0712.0483): **Hubbard model with site-dependent magnetic field is QMA-hard**. Also O'Gorman, Irani, Whitfield, Fefferman, *PRX Quantum* 3, 020322 (2022), [arXiv:2103.08215](https://arxiv.org/abs/2103.08215): electronic structure in a fixed basis is QMA-complete. Bose-Hubbard QMA-complete: Childs, Gosset, Webb.
- **Opposite camp is active (timeliness)**: QAE-for-strong-correlation, [arXiv:2412.20464](https://arxiv.org/abs/2412.20464) (Zade et al., H4 on D-Wave Advantage 4.1, ~1.2% of FCI, self-described "preliminary" scaling analysis, pro-QAE); VarQA [arXiv:2503.15473](https://arxiv.org/pdf/2503.15473).

### Still to verify (searches were rate-limited/flagged; from knowledge, NOT yet confirmed):
- D-Wave ICE / effective coupler precision (~4–6 bits, h/J errors ~1–2%) — get from D-Wave docs + Zaborniak & de Sousa (2021); Hauke et al., *Rep. Prog. Phys.* 83, 054401 (2020) review.
- Largest embeddable clique: Advantage/Pegasus ~K177; Advantage2/Zephyr ~K230 (verify Boothby et al. + D-Wave whitepapers).
- Non-stoquastic hardware: Ozfidan et al., *PRApplied* (2020) 2-qubit XX demo only; nothing programmable at scale.
- Troyer & Wiese, *PRL* 94, 170201 (2005) (sign-problem NP-hard); Barahona 1982 (Ising NP-hard) — standard, cite-safe.
- 4×4 Hubbard U=4 half-filling ED benchmark: E0/site ≈ −0.851 (PBC) — verify against LeBlanc et al., *PRX* 5, 041041 (2015) benchmark tables.
- ED practical ceiling for 2D Hubbard ~20–22 sites (verify recent Lanczos records).

### The gap (what is genuinely novel)
1. **Three-way failure taxonomy** — the three pipelines fail for *different, complementary* reasons (variables/precision vs coupler type vs information content). Nobody has published this side-by-side.
2. **Lattice-scale accounting at just-beyond-ED size (4×6 Hubbard)** with exact sector combinatorics (existing negatives are molecular, ≤ H₂O).
3. **Precision/dynamic-range as the binding constraint** (analog ~4–6 bit couplers vs growing coefficient range) — kills the approach before qubit count does.
4. 1D t-V as the **positive control**: the one case where annealing-style classical QMC *does* work, and exactly why (stoquasticity).

## 4. Why each pipeline fails on Ising-class hardware (2D Hubbard)

- **BK-XBK**: NOT merely "too many terms". Two separate blowups: (a) **variables** — r replicas must scale with the number of significant determinants (~10⁷+ at U/t=4, 24 sites) → ~5×10⁸ logical spins; (b) **precision** — coefficient dynamic range after quadratization exceeds analog coupler resolution at sizes far below (a). Terms ≳10¹⁶ 2-local couplers after quadratization.
- **Levy** (direct adiabatic prep, problem = final H): term COUNT is tiny (~152 hopping strings). The blocker is coupler **TYPE**: needs programmable multi-qubit XX/YY (weight ≤12 under JW, ≤3–4 under compact encoding) in the final Hamiltonian. Every Ising machine has exactly 0 such couplers (only global single-qubit σx ramped to zero). Even given them: T ≳ 1/Δ²_min with gap generically collapsing, and QMA-hardness forbids a generic efficient path.
- **Yip** (annealer samples diagonal part → classical subspace diag): the annealer-side QUBO is trivially small (fits today's hardware) — that's the tell. The diagonal Hubbard energy U·Σn↑n↓ is **uninformative**: at half filling its ground manifold is ALL 2²⁴ ≈ 1.7×10⁷ singly-occupied configs, exactly degenerate (zero ranking), and doublon-holon configs that carry the kinetic correlations are penalized, not selected. Required support |B| ~ 10⁷–10⁹ determinants (selected-CI scale, estimate) → the classical Lanczos on B IS ED-class work. The annealer contributes nothing; contrast 1D t-V where V·ΣZZ has real structure (why tv_yip worked).

## 5. Resource table (4×6 = 24 sites, half filling, U/t=4 — smallest standard lattice beyond ED ceiling ~20 sites)

Exact combinatorics: sector dim = C(24,12)² = 2,704,156² ≈ 7.3×10¹²; bonds = 38; JW qubits = 48; hopping Pauli strings = 152 (weight 2–12); singly-occupied manifold = 2²⁴ ≈ 1.7×10⁷.

| | BK-XBK (replica r) | BK-XBK / QAE (amplitude, b=8 bits) | Levy (direct AQC) | Yip (sampler + subspace) |
|---|---|---|---|---|
| Logical binary variables | 49r ≈ **5×10⁸** (r~10⁷ est.) | D·b ≈ **6×10¹³** | 48 qubits (JW) / ~72 (compact) | 24–48 (trivial) |
| 2-local couplers/terms needed | ≳**10¹⁶** (r²/2 × 176 terms × quadratization) | ≳**2×10¹⁶** (NNZ×b²; normalization penalty → up to ~10²⁷) | ~176 total, BUT **152 must be XX/YY-type multi-qubit** | ≤ ~10² |
| Available (Advantage2: ~4.4×10³ qubits, ~4×10⁴ couplers, ZZ only) | short ~10⁵× vars, ~10¹²× couplers (before embedding; ~10¹²⁻¹³ physical after) | short ~10¹⁰× vars | **0 of the required coupler type exist** | fits — but output carries no signal |
| Digital Ising (10⁵–10⁶ vars, full precision) | short ~10³× vars, ~10¹⁰× terms | short ~10⁸× vars | N/A (no off-diagonal at all) | fits — same information failure |
| Binding constraint | variables + analog precision | variables (exponential by construction) | coupler type; then gap/QMA | information content of diagonal spectrum |
| Failure class | exponential encoding cost | exponential encoding cost | wrong Hamiltonian class for hardware | annealer solves the easy part only |

(r~10⁷ and |B|~10⁷–10⁹ are literature-informed estimates — pin down via SHCI/FCIQMC Hubbard convergence studies before publication. All other numbers are exact combinatorics.)

## 6. Two-paper plan (user's proposal, refined)

- **Paper 1 — the no-go**: unified resource accounting + three-way taxonomy on Ising-class hardware, anchored by QMA-hardness + stoquasticity + precision; 1D t-V (existing validated pipeline!) as positive control. Gap confirmed; timely (2024–25 pro-QAE papers claim the opposite on tiny systems).
- **Paper 2 — constructive counterpart**: NOT "a mapping paper" (gate-based Hubbard mapping space is crowded: Cade-Mineh-Montanaro-Stanisic PRB 2020; Kivlichan et al. swap networks; Google Fermi-Hubbard experiments). Defensible version: port the SAME three cost metrics (XBK variable count / Levy gap-time / Yip sample complexity) to gate-based hardware on the SAME benchmark instances → cross-architecture comparison. Analog simulators (cold atoms) need no mapping — they enter as the native-hardware baseline column, not a mapped target.

## 6b. NHOIM / PUBO thread (added 2026-08-02)

- User ran the r=2 XBK PUBOs (`~/Downloads/model_pubo_dict.py`, `model_pubos.py`, generated by
  `export_pubo_dict.py`) on a neuromorphic higher-order Ising machine (NHOIM): 9/9 toy models
  solved exactly, incl. hubbard2d 2x2 (12 vars, 1087-term P, lambda*=-2; note E_xbk=-2 vs
  E_exact=-7.254 — the r=2 accuracy gap is itself a paper datum). Results: `~/Downloads/results.txt`.
- `hubbard2d_pubos.py` (validated: regenerates the 2x2 benchmark H/C dicts EXACTLY) now emits
  `hubbard2d_big_pubos.py` + `hubbard2d_big_pubo_dict.py` (both in Pipeline/ and ~/Downloads/),
  same formats as the NHOIM inputs, for 4x4 and 4x6 OBC U/t=4 half filling, r=2, sector 0, p=0:
  - 4x4: 32 modes -> BK -> taper -> n=30, n_vars=60, 177 Pauli terms, 452 factored terms.
    Expanded PUBO CANNOT exist: |C| = 2^30+1 ≈ 1.07e9 exactly; |H| cross-monomials ≤ 3.8e11.
  - 4x6: 48 modes -> n=46, n_vars=92, ~300 Pauli terms, 596 factored terms.
    |C| = 2^46+1 ≈ 7.0e13; |H| ≤ 3.2e16.
  - Key theorem-let for Paper 1: every cross-replica XBK factor (X,Y,Z,I alike) carries exactly
    2 monomials, so each (Pauli term, j≠k pair) expands to exactly 2^n monomials; C(x) = 2^n+1
    monomials exactly at r=2. Pre-reduction blow-up measured on the project's own toolchain.
  - The files carry the exact PUBOs in FACTORED product form + self-contained pointwise
    evaluators (eval_H_factored / eval_C_factored) — usable by any sampling solver as an oracle.
- `hubbard2d_ed.py` (spin-factorized 3-vector Lanczos, selftest: dense/eigsh/free-fermion all
  pass; one bug found+fixed: beta_prev never updated in the recurrence): 4x4 OBC U/t=4 half
  filling E0 = -11.24603020 (E0/site = -0.702877), 84 iters, ~21 min, ~7 GB on the Mac.
  CSV: ~/Downloads/hubbard2d_4x4_obc_U4_ed.csv. PBC run (literature benchmark ≈ -0.851/site,
  LeBlanc PRX 2015) not yet run — same command with --bc pbc.

## 7. Open items

- Verify the "still to verify" citations above (searches were rate-limited this session).
- Task #12 (pending): 4×4 Hubbard ED/Lanczos on cluster — now doubles as generating Paper 1's benchmark row + positive-control data.
- Optional: identify the exact "Levy" and "Yip" source papers the pipeline names came from (user knows; record them here when confirmed).
