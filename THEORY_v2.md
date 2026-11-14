# Theory notes for the three-layer extension

Status: **draft written alongside the code; every statement below must be checked by the authors before it goes into the paper.**
Statements of the current draft (Propositions 7, 8, 12, 13 and Lemma 11) are not repeated here; `tests/test_propositions.py` checks them numerically.

Notation. `x^F` is the individual, `a = (δ, b)` a plan (changes and blocks), `Φ_ω(x^F, a)` the counterfactual under the SCM version `ω = (G, g)` (graph and mechanisms), `m` the margin of the black box, and `ω*` the true SCM. A plan *works* if `m(Φ_{ω*}(x^F, a)) ≥ 0`.

---

## 1. Error budget across layers

**Proposition T1 (union bound).** Let `Ĝ` be a data-dependent set of graphs and, for each graph `G`, `Ĉ(G)` a data-dependent set of mechanisms, such that

- `P(G* ∈ Ĝ) ≥ 1 − α₁`, and
- `P(g* ∈ Ĉ(G*)) ≥ 1 − α₂`.

Let `A` be the event that the assumptions about the recourse itself hold for this individual (the changes are achieved, the blocks hold, the used edges operate), with `P(A) ≥ 1 − α₃`. If the plan satisfies `m(Φ_{(G,g)}(x^F, a)) ≥ 0` for every `G ∈ Ĝ` and every `g ∈ Ĉ(G)`, then it works with probability at least `1 − α₁ − α₂ − α₃`.

*Proof.* On the intersection of the three events the true SCM is one of the versions the plan was checked against and the operator applied is the one that was evaluated, so the plan works. The complement has probability at most `α₁ + α₂ + α₃`. ∎

Remarks.

1. The probability is over the data used to build the sets (and over `A`). It is a **marginal** statement: it does not say that, for the dataset at hand, the plan works. Section 4 comes back to this.
2. The proposition needs sets that are valid confidence sets. What the code uses by default is weaker:
   - The bootstrap of graphs (`extensions.bootstrap_graphs`) is **not** a confidence set for the graph. No coverage is claimed for it. Valid alternatives exist for restricted model classes (for example confidence sets of causal orderings); they are not implemented.
   - The bootstrap of mechanisms (`scm.Ensemble`) with the `(1 − ε)` rule of Definition 5 is a percentile-type heuristic, not a simultaneous confidence set.
   - `α₃` is an expert input. The 3 % used in the experiments is illustrative.
3. The bound is conservative. In notebook 01 the budget was 0.20 and the measured failure rate 0.05.

## 2. A case with a formal guarantee: linear mechanisms, known order, affine margin

**Assumptions.** (a) The DAG is known. (b) Every mechanism is linear with additive noise, `x_j = w_j · x_{pa(j)} + c_j + u_j`. (c) The noise is Gaussian and homoscedastic (the intervals are then exact; without Gaussianity they are asymptotic). (d) The margin is affine in the inputs, `m(x) = β · x + β₀`. (e) Changes are additive (soft) or hard, and blocks are as in Definition 3.

Let `E(a)` be the set of edges on directed paths from a changed variable to the model that the plan does not cut. For each `e ∈ E(a)` take the usual `t` interval of its least-squares coefficient at level `1 − α₂ / |E(a)|`, and let `B` be the product of these intervals (a box).

**Proposition T2.** Under (a)–(e), with probability at least `1 − α₂` over the data, if the margin of the plan is non-negative at every vertex of `B`, the plan works.

*Proof.* By Bonferroni, the true coefficients of the edges in `E(a)` lie in `B` with probability at least `1 − α₂`. Under (b) and (e), with abduction, the counterfactual value of a node is its factual value plus the sum over uncut directed paths from changed variables of `δ_source × (product of the coefficients on the path)` (Proposition 8); intercepts and noise cancel. In a DAG an edge appears at most once on a path, so each product is of degree at most one in each coefficient, and by (d) the margin is a multi-affine function of the coefficients of `E(a)`. A multi-affine function on a box attains its minimum at a vertex (fix all coordinates but one: the function is affine in it, so it is minimised at an endpoint; repeat). Hence a non-negative minimum over the vertices implies a non-negative margin at the true coefficients. ∎

What breaks when an assumption is dropped:

- Without (d) (a neural network or a tree ensemble as black box) the margin is not multi-affine, and checking vertices, or sampling the box, is a heuristic. A sufficient replacement is monotonicity: if the margin is monotone in every input along the direction each coefficient moves it, vertices still suffice. This is not checked by the code.
- Without (a) the box has to be built for every graph in the set of Proposition T1.
- Without (b) there is no finite-dimensional box; Section 3 is the non-parametric route.
- The number of vertices is `2^{|E(a)|}`: fine for the running example, not for a dense graph.

## 3. Conformal intervals for mechanisms, under intervention

**What a conformal interval covers.** Fit `ĝ_j` on one half of the data and compute residual scores `s_i = |x_{ij} − ĝ_j(x_{i,pa(j)})|` on the other half. For a new exchangeable unit, `ĝ_j(X_{pa}) ± q` with `q` the usual quantile covers `X_j` with probability at least `1 − α`. This is marginal over the parents.

**Why intervention is a covariate shift.** Intervening upstream changes the distribution of the parents of `j`, not the mechanism of `j` (modularity). The outcome of a unit whose parents are set to `z` is distributed as `X_j | X_{pa} = z`. So predicting node `j` after a plan is prediction under covariate shift, for which weighted conformal prediction applies ([Tibshirani, Barber, Candès and Ramdas, 2019](https://arxiv.org/abs/1904.06019)) when the likelihood ratio between the shifted and the original parent distributions exists and is known.

**The gap.** A plan sends the parents to a single point `z*`. A point mass has no likelihood ratio with respect to the observational distribution. `extensions.conformal_plan` uses kernel weights centred at `z*` (localized conformal prediction: [Guan, 2023](https://doi.org/10.1093/biomet/asac040); [Hore and Barber, 2023](https://arxiv.org/abs/2310.07850)). The guarantee that transfers is coverage for the kernel-smoothed target, not at the point `z*`. Exact coverage conditional on a point is impossible without assumptions; what we have at `z*` is approximate and depends on the bandwidth and on the effective number of calibration points, which the function reports. *(These three references are not yet in `paper_clear2027/references_v2.bib`. Their existence, titles and authors were checked on 2026-11-13, see the list at the end; whether they state what this paragraph attributes to them has not been checked. Verify before citing.)*

**Proposition T3 (number of mechanisms touched).** Let `M(a)` be the mechanisms through which the plan propagates (nodes with a changed parent through an uncut edge, not themselves set by a hard intervention) and `L = |M(a)|`. Suppose that for each `j ∈ M(a)` there is an interval `C_j(z)` such that, for every value `z` of the parents, `P(Y_j(z) ∈ C_j(z)) ≥ 1 − α`, where `Y_j(z)` is the outcome of node `j` when its parents are set to `z`. Then, with probability at least `1 − Lα`, every node of `M(a)` lies in its interval along the propagation, where the interval of each node is evaluated at the realised values of its parents.

*Proof.* Order `M(a)` topologically. Conditionally on the realised parents `z`, node `j` falls outside `C_j(z)` with probability at most `α` by the pointwise assumption; integrate over `z` and add over the `L` nodes. ∎

Remarks.

1. The assumption is **pointwise (conditional) coverage**. Split conformal gives marginal coverage only, so the bound is proved for the observational case (where marginal coverage is the right notion; notebook 02 measured 0.953 at `L = 1` down to 0.665 at `L = 8`, above `1 − Lα`) and holds under intervention only to the extent that localization delivers conditional coverage. Notebook 04 checked one heteroscedastic mechanism in simulation: the standard interval dropped to 0.43–0.73 coverage away from the centre of the data and the localized one stayed at 0.90–0.95. That is evidence, not a proof.
2. Along a single directed path `L` is the length of the path. In general `L` is at most the number of descendants of the changed variables. Blocking an edge or intervening hard on a node removes mechanisms from `M(a)` and so tightens the bound: under the conformal criterion blocks become more valuable (seen in experiment X).
3. **Abduction.** The interval covers the outcome of a *new* unit, noise included. The plan concerns *this* individual, whose noise `u_j` is fixed. Under additive noise independent of the parents, `g_j(z*) + u_j` for a randomly drawn individual has the distribution of `Y_j(z*)`, so coverage holds marginally over individuals. It does not hold conditionally on the individual's factual values (which reveal `u_j`), and individuals selected because the model rejected them are not a random draw. The interval is also much wider than the uncertainty that matters for the individual, which is the error of `ĝ_j(z*) − ĝ_j(z^F)`: the noise cancels in the counterfactual but not in the interval. In experiment X a plan at the boundary of bootstrap-robust validity almost never passes the conformal check for this reason. A conformal procedure whose score targets the *difference* `g_j(z*) − g_j(z^F)` would be the right tool; we do not know of one and have not built it.

## 4. Marginal and conditional validity

All guarantees above average over datasets. For one dataset the estimates can be off in the same direction for every individual. Two observations from the code:

- Notebook 01: calibration error of the success probability 0.164 with one dataset, 0.026 averaged over 30 datasets.
- Package, perioperative case, data seed 0: the three individuals tried all had a robust plan that failed under the true SCM by a margin of about 0.01, because the bootstrap versions of that dataset did not reach the true coefficient of one edge.

Consequences for the paper: report validity across data seeds, not across individuals of one dataset; state guarantees as marginal; and treat "this plan works with probability p̂" as a statement about the procedure.

## 5. Not proven

- Any coverage for the bootstrap of graphs.
- Coverage of the localized conformal interval at a point.
- A box guarantee for non-linear black boxes or non-linear mechanisms.
- Optimality of the plan found: Proposition 12 is local, and the search has no regret bound.

## References mentioned in this file

Links checked on 2026-11-13 (title and authors against the arXiv API or Crossref). The check covers that each link points to the stated paper, not the claims this file makes about its content.

- Tibshirani, Barber, Candès and Ramdas. *Conformal Prediction Under Covariate Shift*. NeurIPS 2019. <https://arxiv.org/abs/1904.06019>
- Guan. *Localized conformal prediction: a generalized inference framework for conformal prediction*. Biometrika (published online July 2022; cited above as 2023, to confirm against the journal issue). <https://doi.org/10.1093/biomet/asac040>, preprint <https://arxiv.org/abs/2106.08460>
- Hore and Barber. *Conformal prediction with local weights: randomization enables local guarantees*. arXiv preprint, 2023. <https://arxiv.org/abs/2310.07850>
