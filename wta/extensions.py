"""Extension beyond the current draft: three layers of error.
Layer 1, the graph: bootstrap over graphs, and an SCM whose versions do not share a graph.
Layer 2, the mechanisms: localized conformal intervals at the point the plan moves the parents to.
Layer 3, the recourse itself: partial changes, failing edges and blocks that do not hold (see explain.answer).
Plus the baseline of Karimi et al. (2020), re-implemented by us from their paper (GP-SCM).
"""
import copy, itertools
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Const
from .scm import Spec, Ensemble, MultiGraph
from .search import robust


# ---------------------------------------------------------------- layer 1: the graph
def _bic(X, j, pa):
    n = len(X); y = X[:, j]; Z = np.c_[X[:, pa], np.ones(n)] if len(pa) else np.ones((n, 1))
    r = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]; return n * np.log(r.var() + 1e-12) + (len(pa) + 1) * np.log(n)


def _parents_given_order(X, order, forbidden, max_parents):
    """Best parent set of each node among its predecessors, by BIC (linear-Gaussian score).
    Exhaustive when there are at most 8 predecessors, greedy forward selection otherwise."""
    out = {}
    for q, j in enumerate(order):
        pred = [k for k in order[:q] if (k, j) not in forbidden]
        if len(pred) <= 8:
            sets = [list(c) for r in range(min(max_parents, len(pred)) + 1) for c in itertools.combinations(pred, r)]
            out[j] = sorted(min(sets, key=lambda s: _bic(X, j, s)))
        else:
            cur, best = [], _bic(X, j, [])
            while len(cur) < max_parents:
                cand = [(k, _bic(X, j, cur + [k])) for k in pred if k not in cur]; k, b = min(cand, key=lambda t: t[1])
                if b >= best: break
                cur.append(k); best = b
            out[j] = sorted(cur)
    return out


def _hill_climb(X, forbidden, max_parents, rng, restarts=2):
    """Order-free search: greedy hill climbing over orders (adjacent swaps), scoring each order by its best parent
    sets. APPROXIMATE: it can stop at a local optimum, and with Gaussian data graphs in the same Markov equivalence
    class score the same. For five or six variables, notebook 04 has the exact search over all orders."""
    d = X.shape[1]; best, bs = None, np.inf
    def score(order):
        pa = _parents_given_order(X, order, forbidden, max_parents); return sum(_bic(X, j, pa[j]) for j in order), pa
    for _ in range(restarts):
        order = list(rng.permutation(d)); s, pa = score(order); improved = True
        while improved:
            improved = False
            for q in range(d - 1):
                o2 = order.copy(); o2[q], o2[q + 1] = o2[q + 1], o2[q]; s2, pa2 = score(o2)
                if s2 < s - 1e-9: order, s, pa, improved = o2, s2, pa2, True
        if s < bs: best, bs = pa, s
    return best


def bootstrap_graphs(X, B=100, order=None, forbidden=(), max_parents=3, seed=0):
    """Layer 1. Re-estimates the graph on B resamples. order: list of column indices (expert knowledge) or None.
    forbidden: set of (parent, child) index pairs. Returns [(parents dict {child: [parents]}, weight)], heaviest first."""
    rng = np.random.default_rng(seed); n = len(X); count = {}; forbidden = set(forbidden)
    for b in range(B):
        Xb = X[rng.integers(0, n, n)]
        pa = _parents_given_order(Xb, order, forbidden, max_parents) if order is not None else _hill_climb(Xb, forbidden, max_parents, rng)
        key = tuple(sorted((j, tuple(p)) for j, p in pa.items() if p)); count[key] = count.get(key, 0) + 1
    return [({j: list(p) for j, p in k}, c / B) for k, c in sorted(count.items(), key=lambda t: -t[1])]


def graph_ensemble(spec, X, graphs, S=40, kind="linear", seed=0, top=8, **ens_kw):
    """An SCM whose versions do not share a graph: each of the `top` heaviest graphs gets a number of bootstrap
    re-fits proportional to its weight. The action space (changes and blockable edges) is that of `spec`;
    blocking an edge that a graph does not have does nothing in that graph."""
    graphs = graphs[:top]; w = np.array([g[1] for g in graphs]); w = w / w.sum(); members = []
    for i, ((pa, _), wi) in enumerate(zip(graphs, w)):
        par = {spec.names[j]: [spec.names[k] for k in p] for j, p in pa.items()}
        blk = [(spec.names[k], spec.names[j]) for (k, j) in spec.blk if k in pa.get(j, [])]
        sp = Spec(spec.names, par, spec.actionable, {v: (spec.lo[q], spec.hi[q]) for q, v in enumerate(spec.actionable)}, None, blk, tuple(spec.hard), [spec.names[t] for t in spec.inp])
        sp.kap = spec.kap; e = Ensemble(sp, X, S=max(1, int(round(wi * S))), kind=kind, seed=seed + i, **ens_kw)
        e.cols = [spec.nA + spec.blk.index((sp.idx[k], sp.idx[j])) for (k, j) in blk]; members.append(e)
    mg = MultiGraphAligned(members, spec); mg.graph_id = np.concatenate([[i] * e.S for i, e in enumerate(members)]); mg.graph_weights = w; return mg


class MultiGraphAligned(MultiGraph):
    """MultiGraph whose members may lack some blockable edges: the block columns are routed by name."""
    def __init__(self, members, spec): self.members, self.spec, self.S = members, spec, sum(e.S for e in members)

    def counterfactual(self, x, A, extra_cut=(), lifted=(), gamma=None, block_ok=None, **kw):
        A = np.atleast_2d(np.asarray(A, float)); outs, s0 = [], 0
        for e in self.members:
            A2 = np.c_[A[:, :self.spec.nA], A[:, e.cols]] if e.cols else A[:, :self.spec.nA]
            bo = None if block_ok is None else np.asarray(block_ok)[s0:s0 + e.S]
            outs.append(e.counterfactual(x, A2, [c for c in extra_cut if c in e.spec.eidx], [l for l in lifted if l in e.spec.eidx], gamma, bo)); s0 += e.S
        return np.concatenate(outs, axis=1)


def layered_success(scm_mech, scm_graph, x, a, margin_fn, block_fail=0.03, edge_fail=0.0, used=(), seed=0):
    """Success probability by layers: (1) mechanisms only, (2) + graph, (3) + assumptions about the recourse itself:
    each block fails with probability `block_fail` and each used edge with probability `edge_fail` (EXPERT INPUTS,
    not estimated from data). Returns a dict of (p-hat, number of versions)."""
    rng = np.random.default_rng(seed); out = {}
    for name, scm in [("mechanisms", scm_mech), ("mechanisms + graph", scm_graph)]:
        if scm is None: continue
        M = margin_fn(scm.counterfactual(x, a)[0]); out[name] = (float((M >= 0).mean()), len(M))
    scm = scm_graph or scm_mech; ok = (rng.random(scm.S) >= block_fail).astype(float); M = np.empty(scm.S)
    fail = rng.random((scm.S, len(used))) < edge_fail; Mb = margin_fn(scm.counterfactual(x, a, block_ok=ok)[0])
    for s in range(scm.S):
        M[s] = Mb[s] if not fail[s].any() else margin_fn(scm.counterfactual(x, a, extra_cut=[e for e, f in zip(used, fail[s]) if f], block_ok=ok)[0])[s]
    out["+ recourse assumptions"] = (float((M >= 0).mean()), scm.S); return out


# ---------------------------------------------------------------- layer 2: localized conformal inside the plan
def conformal_plan(ens, X_cal, x, a, margin_fn, alpha=0.10, bw=0.5, n_draws=200, seed=0):
    """For every mechanism the plan propagates through, a localized conformal half-width at the parent values the
    plan produces (weights = Gaussian kernel on the standardised distance to those values), at level alpha / L
    (union bound over the L mechanisms). Then the plan is re-evaluated with every mechanism perturbed inside its
    interval (corners when L <= 6, plus random draws) and the worst margin is reported.
    What is guaranteed and what is not is spelled out in THEORY.md, section 3."""
    sp = ens.spec; rng = np.random.default_rng(seed); e0 = ens.subset([0]); xc = e0.counterfactual(x, a)[0, 0]; cut = e0.cuts(a)[0]
    apos = {j: i for i, j in enumerate(sp.act)}; touched = []
    for j in sp.order:
        pa = sp.pa[j]
        if not pa or (j in sp.hard_idx and j in apos and a[apos[j]] != 0): continue
        if any((not cut[sp.eidx[(k, j)]]) and xc[k] != x[k] for k in pa): touched.append(j)
    L = len(touched); q, n_eff = {}, {}
    for j in touched:
        pa = sp.pa[j]; Z = X_cal[:, pa]; s = np.abs(e0._lat(j, X_cal[:, j]) - e0._g(j, Z[:, None, :])[:, 0]); sd = Z.std(0) + 1e-9
        w = np.exp(-0.5 * (((Z - xc[pa]) / (bw * sd)) ** 2).sum(1)); o = np.argsort(s); cw = np.cumsum(np.r_[w[o], 1.0]) / (w.sum() + 1.0)
        k = np.searchsorted(cw, 1 - alpha / max(L, 1)); q[j] = s[o][k] if k < len(s) else np.inf; n_eff[j] = w.sum() ** 2 / (w ** 2).sum()
    if L == 0 or not np.all(np.isfinite(list(q.values()))):
        return dict(L=L, level=1 - alpha, half_width={sp.names[j]: q[j] for j in touched}, n_eff={sp.names[j]: n_eff[j] for j in touched}, worst_margin=float(margin_fn(xc[None])[0]) if L == 0 else -np.inf, frac_valid=np.nan)
    U = rng.uniform(-1, 1, (n_draws, L))
    if L <= 6: U = np.vstack([U, np.array(list(itertools.product([-1.0, 1.0], repeat=L)))])
    eK = ens.subset([0] * len(U)); M = margin_fn(eK.counterfactual(x, a, shift={j: U[:, i] * q[j] for i, j in enumerate(touched)})[0])
    return dict(L=L, level=1 - alpha, half_width={sp.names[j]: float(q[j]) for j in touched}, n_eff={sp.names[j]: float(n_eff[j]) for j in touched},
                worst_margin=float(M.min()), frac_valid=float((M >= 0).mean()))


class ConformalBoxSCM:
    """Versions = the point estimate with every mechanism shifted inside its split-conformal interval (level
    alpha / number of mechanisms, NOT localized, so that it does not depend on the action). Searching with eps = 0
    over these versions gives a plan that survives the whole box; conformal_plan then re-checks it with localized
    intervals. Intervals are for the outcome of a NEW unit (noise included), hence wider than needed for the
    individual: see THEORY.md, section 3."""
    def __init__(self, ens, X_cal, alpha=0.10, K=40, seed=0):
        sp = ens.spec; rng = np.random.default_rng(seed); self.spec, self.S = sp, K; self.base = ens.subset([0] * K); e0 = ens.subset([0])
        nodes = [j for j in sp.order if sp.pa[j]]; n = len(X_cal); self.q = {}
        for j in nodes:
            s = np.sort(np.abs(e0._lat(j, X_cal[:, j]) - e0._g(j, X_cal[:, sp.pa[j]][:, None, :])[:, 0]))
            self.q[j] = s[min(n - 1, int(np.ceil((n + 1) * (1 - alpha / len(nodes)))) - 1)]
        U = rng.uniform(-1, 1, (K, len(nodes))); U[: K // 2] = np.sign(U[: K // 2]); U[0] = 0.0
        self.shift = {j: U[:, i] * self.q[j] for i, j in enumerate(nodes)}

    def counterfactual(self, x, A, **kw): return self.base.counterfactual(x, A, shift=self.shift, **kw)


# ---------------------------------------------------------------- Karimi et al. (2020): GP-SCM, re-implemented
class GPSCM:
    """One Gaussian process with additive Gaussian noise per mechanism; counterfactuals are sampled with the
    posterior of the individual's noise (their Propositions 5 and 6). OUR re-implementation from the paper, not
    their code. No edge blocking (their actions are interventions on variables)."""
    def __init__(self, spec, X, seed=0):
        self.spec, self.X, self.h = spec, X, {}
        for j in spec.order:
            p = spec.pa[j]
            if not p: continue
            mu, sd = X[:, p].mean(0), X[:, p].std(0) + 1e-9; y = X[:, j]; my, sy = y.mean(), y.std() + 1e-9
            gp = GaussianProcessRegressor(Const(1.0) * RBF(np.ones(len(p))) + WhiteKernel(0.5), random_state=seed).fit((X[:, p] - mu) / sd, (y - my) / sy)
            k = gp.kernel_; self.h[j] = (p, mu, sd, my, sy, k.k1.k1.constant_value, np.atleast_1d(k.k1.k2.length_scale), k.k2.noise_level)

    @staticmethod
    def _k(A, B, s2, ls):
        A, B = A / ls, B / ls; return s2 * np.exp(-0.5 * np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))

    def prepare(self, x):
        self.x = np.asarray(x, float); self.c = {}; Xa = np.vstack([self.X, self.x[None]])
        for j, (p, mu, sd, my, sy, s2, ls, n2) in self.h.items():
            Z = (Xa[:, p] - mu) / sd; y = (Xa[:, j] - my) / sy; cf = cho_factor(self._k(Z, Z, s2, ls) + n2 * np.eye(len(Z)), lower=True)
            al = cho_solve(cf, y); self.c[j] = (Z, cf, al, n2 * al[-1], n2 - n2 ** 2 * cho_solve(cf, np.eye(len(Z))[:, -1])[-1])

    def sample(self, A, M, rng):
        sp, x = self.spec, self.x; A = np.atleast_2d(A); C = len(A); X = np.tile(x, (C, M, 1)); apos = {j: i for i, j in enumerate(sp.act)}
        for j in sp.order:
            d = A[:, apos[j]][:, None] if j in apos else None
            if j not in self.h:
                if d is not None: X[:, :, j] = x[j] + d
                continue
            p, mu, sd, my, sy, s2, ls, n2 = self.h[j]; Z, cf, al, muF, sF = self.c[j]; Zt = (X[:, :, p].reshape(-1, len(p)) - mu) / sd
            mean, var = np.empty(len(Zt)), np.empty(len(Zt))
            for i in range(0, len(Zt), 20000):
                kt = self._k(Zt[i:i + 20000], Z, s2, ls); mean[i:i + 20000] = muF + kt @ al
                var[i:i + 20000] = sF + s2 - (solve_triangular(cf[0], kt.T, lower=True) ** 2).sum(0)
            draw = (my + sy * (mean + np.sqrt(np.maximum(var, 1e-10)) * rng.standard_normal(len(Zt)))).reshape(C, M)
            draw = np.where((X[:, :, p] == x[p]).all(-1), x[j], draw)
            if d is None: X[:, :, j] = draw
            elif j in sp.hard_idx: X[:, :, j] = np.where(d != 0, x[j] + d, draw)
            else: X[:, :, j] = draw + d
        return X

    def criterion(self, bb, gamma=1.0, M=50, seed=0):
        """valid_fn for search.grid_search: mean of h minus gamma standard deviations >= 0.5 (their equation for the
        individualised recourse under a GP-SCM). Every sampled counterfactual is one counted query."""
        rng = np.random.default_rng(seed)
        def valid(A):
            ok = np.empty(len(A), bool)
            for i in range(0, len(A), 500):
                Xs = self.sample(A[i:i + 500], M, rng); p = 1 / (1 + np.exp(-bb.margin(Xs.reshape(-1, self.spec.d), "grid (GP-SCM)").reshape(Xs.shape[:2])))
                ok[i:i + 500] = p.mean(1) - gamma * p.std(1) >= 0.5
            return ok
        return valid
