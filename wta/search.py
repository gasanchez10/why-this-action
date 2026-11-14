"""Robust validity (Definition 5), Algorithm 1 (composite Bayesian optimisation with in-loop certification and
pruning) and the search baselines and ablations of Experiment C."""
from dataclasses import dataclass, field
import itertools, warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, ConstantKernel as Const
warnings.filterwarnings("ignore")


def robust(M, eps):
    """Robust margin of each row of M (C, S): the margin of the binding version, i.e. the (floor(eps*S)+1)-th smallest.
    The action is robustly valid with tolerance eps iff this value is >= 0. Also returns the binding version."""
    M = np.atleast_2d(M); k = min(int(np.floor(eps * M.shape[1] + 1e-9)), M.shape[1] - 1)
    b = np.argpartition(M, k, axis=1)[:, k]; return M[np.arange(len(M)), b], b


def sample_actions(spec, n, rng, around=None, scale=0.08, blocks=True):
    """Random candidate actions: uniform changes (some coordinates set to zero), and, if `around` is given,
    half of them near that action (including shrunk copies of it)."""
    D = rng.uniform(spec.lo, spec.hi, (n, spec.nA)); D[rng.random(D.shape) < 0.25] = 0.0
    B = (rng.random((n, spec.nB)) < 0.5).astype(float)
    if around is not None:
        h = n // 2; a = np.asarray(around)
        D[:h] = np.clip(a[:spec.nA] + scale * (spec.hi - spec.lo) * rng.standard_normal((h, spec.nA)), spec.lo, spec.hi)
        q = h // 3; D[:q] = a[:spec.nA] * rng.uniform(0.7, 1.0, (q, spec.nA)); D[:h][rng.random((h, spec.nA)) < 0.1] = 0.0
        B[:h] = np.where(rng.random((h, spec.nB)) < 0.15, 1 - a[spec.nA:], a[spec.nA:])
    if not blocks: B[:] = 0.0
    return np.c_[D, B]


class Surrogate:
    """Gaussian process on the MODEL INPUTS: linear part + Matern-3/2 with one length-scale per input (Section 5)."""
    def __init__(self, scale_X, seed=0, cap=250):
        self.mu, self.sd = scale_X.mean(0), scale_X.std(0) + 1e-9; self.X, self.y, self.gp, self.seed, self.cap, self.k = [], [], None, seed, cap, None

    def add(self, X, y): self.X += list(np.atleast_2d(X)); self.y += list(np.atleast_1d(y))

    def fit(self, optimise=True):
        X, y = np.array(self.X), np.array(self.y)
        if len(X) > self.cap:                                   # keep the points closest to the decision boundary
            keep = np.argsort(np.abs(y))[:self.cap]; X, y = X[keep], y[keep]
        d = X.shape[1]; self.ym, self.ys = y.mean(), y.std() + 1e-9
        if optimise or self.k is None:
            k = Const(1.0, (1e-2, 1e2)) * Matern(np.ones(d) * 2.0, (0.2, 50.0), nu=1.5) + Const(1.0, (1e-3, 1e2)) * DotProduct(1.0, "fixed") + WhiteKernel(1e-3, (1e-6, 1e-1))
            self.gp = GaussianProcessRegressor(k, random_state=self.seed).fit((X - self.mu) / self.sd, (y - self.ym) / self.ys); self.k = self.gp.kernel_
        else: self.gp = GaussianProcessRegressor(self.k, optimizer=None).fit((X - self.mu) / self.sd, (y - self.ym) / self.ys)

    def predict(self, P):
        sh = P.shape[:-1]; P = (P.reshape(-1, P.shape[-1]) - self.mu) / self.sd; mu, sd = np.empty(len(P)), np.empty(len(P))
        for i in range(0, len(P), 4000): mu[i:i + 4000], sd[i:i + 4000] = self.gp.predict(P[i:i + 4000], return_std=True)
        return (self.ym + self.ys * mu).reshape(sh), (self.ys * sd).reshape(sh)


@dataclass
class Result:
    action: np.ndarray = None           # the explanation a* (None if nothing was certified)
    cost: float = np.inf
    queries: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)       # one record per evaluation or batch, with its purpose
    incumbents: list = field(default_factory=list)  # successive certified incumbents (action, cost)
    near_misses: list = field(default_factory=list) # actions that failed certification (action, cost, robust margin)
    surrogate: object = None
    n_total: int = 0


def _cf(scm, spec, x, A, propagate, kw):
    if propagate: return scm.counterfactual(x, A, **kw)
    X = np.tile(np.asarray(x, float), (len(A), 1, 1)); X[:, 0, spec.act] += A[:, :spec.nA]; return X      # no causal propagation


def certify(scm, spec, x, a, bb, eps, purpose="certification", propagate=True, kw=None):
    """Evaluate the true model under every version. Returns (robust margin, margins, inputs)."""
    Xc = _cf(scm, spec, x, np.atleast_2d(a), propagate, kw or {})[0]; M = bb.margin(Xc, purpose); return robust(M[None], eps)[0][0], M, Xc


def solve(spec, scm, x, bb, eps=0.10, budget=60, n_cand=1000, n_init=6, seed=0, scale_X=None, blocks=True, propagate=True,
          do_certify=True, prune=True, polish=False, max_cert=25, kw=None, verbose=False):
    """Algorithm 1. `budget` counts search evaluations (initial design + acquisitions); certification, pruning and
    polish evaluations are counted separately in Result.queries.
    Ablations: blocks=False (no edge blocking), propagate=False (no causal propagation), do_certify=False.
    polish=True adds a shrinking step after pruning (an extension, not in the paper's Algorithm 1)."""
    rng = np.random.default_rng(seed); kw = kw or {}; n0 = bb.n; x = np.asarray(x, float); inp = spec.inp
    gp = Surrogate(scale_X[:, inp] if scale_X is not None else np.vstack([x[inp] - 1, x[inp] + 1]), seed); res = Result(surrogate=gp)
    rec = lambda **r: res.trace.append(dict(step=len(res.trace), inc_cost=res.cost, **r))
    m0 = bb.margin(x, "initial design")[0]; gp.add(x[inp], m0); rec(purpose="initial design", action=np.zeros(spec.nA + spec.nB), version=0, margin=m0, cost=0.0)
    A0 = sample_actions(spec, n_init, rng, blocks=blocks); X0 = _cf(scm, spec, x, A0, propagate, kw)[:, 0]; M0 = bb.margin(X0, "initial design"); gp.add(X0[:, inp], M0)
    for a, m in zip(A0, M0): rec(purpose="initial design", action=a, version=0, margin=m, cost=spec.cost(a)[0])
    used, n_cert, it, cmax = 1 + n_init, 0, 0, spec.cost(np.r_[np.maximum(np.abs(spec.lo), np.abs(spec.hi)), np.zeros(spec.nB)])[0]
    failed = []
    while used < budget and it < budget + max_cert:
        it += 1; gp.fit(optimise=(it % 6 == 1))
        A = sample_actions(spec, n_cand, rng, around=res.action, blocks=blocks)
        if res.action is None and failed: A[: n_cand // 4] = sample_actions(spec, n_cand // 4, rng, around=failed[-1], scale=0.15, blocks=blocks)[: n_cand // 4]
        c = spec.cost(A); Xc = _cf(scm, spec, x, A, propagate, kw); mu, sd = gp.predict(Xc[..., inp]); r, b = robust(mu, eps); sb = sd[np.arange(len(A)), b]
        cstar = res.cost if res.action is not None else cmax * 1.01
        ok = (r - sb >= 0) & (c < cstar * 0.99 - 1e-9)
        if do_certify and ok.any() and n_cert < max_cert:                      # in-loop certification of the cheapest promising action
            i = np.where(ok)[0][np.argmin(c[ok])]; M = bb.margin(Xc[i], "certification"); rr = robust(M[None], eps)[0][0]; n_cert += 1
            low = np.argsort(M)[:3]; gp.add(Xc[i][low][:, inp], M[low])
            rec(purpose="certification", action=A[i], version=-1, margin=rr, cost=c[i], predicted=r[i], predicted_sd=sb[i])
            if rr >= 0: res.action, res.cost = A[i].copy(), c[i]; res.incumbents.append((A[i].copy(), c[i]))
            else: res.near_misses.append((A[i].copy(), c[i], rr)); failed.append(A[i].copy())
            continue
        alpha = np.maximum(cstar - c, 0) * norm.cdf(r / np.maximum(sb, 1e-9)) + 1e-12 * sb                 # expected feasible improvement
        i = int(np.argmax(alpha)); m = bb.margin(Xc[i, b[i]], "acquisition")[0]; gp.add(Xc[i, b[i], inp], m); used += 1
        rec(purpose="acquisition", action=A[i], version=int(b[i]), margin=m, cost=c[i], predicted=r[i], predicted_sd=sb[i], pruned_candidates=int((c >= cstar).sum()))
    if not do_certify:                                                         # ablation: trust the surrogate
        gp.fit(); A = sample_actions(spec, 4 * n_cand, rng, blocks=blocks); c = spec.cost(A); r, _ = robust(gp.predict(_cf(scm, spec, x, A, propagate, kw)[..., inp])[0], eps)
        if (r >= 0).any(): i = np.where(r >= 0)[0][np.argmin(c[r >= 0])]; res.action, res.cost = A[i], c[i]
    elif res.action is not None:
        if prune: _prune(spec, scm, x, bb, eps, res, propagate, kw, rec)
        if polish: _polish(spec, scm, x, bb, eps, res, propagate, kw, rec)
    res.n_total = bb.n - n0; q = {}
    for p, k in bb.log: q[p] = q.get(p, 0) + k
    res.queries = q
    if verbose: print(spec.describe(res.action), f"cost {res.cost:.3f}", q)
    return res


def _prune(spec, scm, x, bb, eps, res, propagate, kw, rec):
    """Proposition 12: try to set each single change to zero and to lift each single block; accept a removal only
    after evaluating the true model under every version; stop when no single removal passes."""
    while True:
        best = None
        for i in range(spec.nA + spec.nB):
            if res.action[i] == 0: continue
            a = res.action.copy(); a[i] = 0.0; rr = certify(scm, spec, x, a, bb, eps, "pruning", propagate, kw)[0]
            rec(purpose="pruning", action=a, version=-1, margin=rr, cost=spec.cost(a)[0])
            if rr >= 0 and (best is None or spec.cost(a)[0] < spec.cost(best)[0]): best = a
        if best is None: return
        res.action, res.cost = best, spec.cost(best)[0]; res.incumbents.append((best.copy(), res.cost))


def _polish(spec, scm, x, bb, eps, res, propagate, kw, rec, steps=(0.2, 0.1, 0.05, 0.02)):
    """Extension: shrink each change by a fraction while robust validity (checked on the true model) holds."""
    for h in steps:
        moved = True
        while moved:
            moved = False
            for i in np.argsort(-np.abs(res.action[:spec.nA]) * spec.kap):
                if res.action[i] == 0: continue
                a = res.action.copy(); a[i] *= (1 - h); rr = certify(scm, spec, x, a, bb, eps, "polish", propagate, kw)[0]
                rec(purpose="polish", action=a, version=-1, margin=rr, cost=spec.cost(a)[0])
                if rr >= 0: res.action, res.cost, moved = a, spec.cost(a)[0], True; res.incumbents.append((a.copy(), res.cost))


# ---------------------------------------------------------------- references and baselines
def reference(spec, scm, x, proba_margin, eps, n=20000, rounds=3, seed=0, blocks=True, kw=None):
    """Reference optimum of the robust problem by dense random search with local refinement.
    Uses the model freely (no query counting): it is a yardstick, not a method."""
    rng = np.random.default_rng(seed); best, bc = None, np.inf
    for r in range(rounds + 1):
        A = sample_actions(spec, n if r == 0 else n // 2, rng, around=best, scale=0.08 / (2 ** max(r - 1, 0)), blocks=blocks)
        if spec.nB and r == 0 and spec.nA <= 3:                                # add a regular grid when the action space is small
            g = [np.linspace(spec.lo[i], spec.hi[i], 25) for i in range(spec.nA)] + [[0.0, 1.0] if blocks else [0.0]] * spec.nB
            A = np.vstack([A, np.array(list(itertools.product(*g)))])
        c = spec.cost(A); keep = c < bc; A, c = A[keep], c[keep]; R = np.empty(len(A))
        for i in range(0, len(A), 2000):
            Xc = scm.counterfactual(x, A[i:i + 2000], **(kw or {})); R[i:i + 2000] = robust(proba_margin(Xc.reshape(-1, spec.d)).reshape(Xc.shape[:2]), eps)[0]
        if (R >= 0).any(): j = np.where(R >= 0)[0][np.argmin(c[R >= 0])]; best, bc = A[j].copy(), c[j]
    return best, bc


def random_search(spec, scm, x, bb, eps, budget=2000, seed=0, blocks=True):
    """Baseline: random actions cheaper than the incumbent, each certified on the true model with early stopping."""
    rng = np.random.default_rng(seed); res = Result(); n0 = bb.n; kmax = int(np.floor(eps * scm.S + 1e-9))
    while bb.n - n0 < budget:
        A = sample_actions(spec, 64, rng, around=res.action, blocks=blocks); c = spec.cost(A)
        for i in np.argsort(c):
            if c[i] >= res.cost or bb.n - n0 >= budget: continue
            Xc = scm.counterfactual(x, A[i:i + 1])[0]; fails = 0
            for s in rng.permutation(scm.S):
                fails += bb.margin(Xc[s], "random search")[0] < 0
                if fails > kmax: break
            if fails <= kmax: res.action, res.cost = A[i].copy(), c[i]; res.incumbents.append((A[i].copy(), c[i]))
    res.n_total = bb.n - n0; return res


def action_space_bo(spec, scm, x, bb, eps, n_eval=40, n_cand=1000, seed=0, blocks=True):
    """Ablation 'surrogate on actions': a GP maps the action directly to its robust margin, so every evaluation
    costs one query per version."""
    rng = np.random.default_rng(seed); res = Result(); n0 = bb.n; A = sample_actions(spec, 6, rng, blocks=blocks)
    y = np.array([certify(scm, spec, x, a, bb, eps, "action-space BO")[0] for a in A]); sc = np.r_[spec.hi - spec.lo, np.ones(spec.nB)]
    def upd():
        ok = y >= 0
        if ok.any(): i = np.where(ok)[0][np.argmin(spec.cost(A[ok]))]; res.action, res.cost = A[i].copy(), spec.cost(A[i:i + 1])[0]
    upd()
    for t in range(n_eval - 6):
        gp = GaussianProcessRegressor(Const(1.0) * Matern(np.ones(A.shape[1]), nu=1.5) + Const(1.0) * DotProduct(1.0) + WhiteKernel(1e-3), normalize_y=True, random_state=seed).fit(A / sc, y)
        Cn = sample_actions(spec, n_cand, rng, around=res.action, blocks=blocks); mu, sd = gp.predict(Cn / sc, return_std=True); c = spec.cost(Cn)
        cstar = res.cost if res.action is not None else c.max() * 1.01
        i = int(np.argmax(np.maximum(cstar - c, 0) * norm.cdf(mu / np.maximum(sd, 1e-9)) + 1e-12 * sd))
        A = np.vstack([A, Cn[i]]); y = np.r_[y, certify(scm, spec, x, Cn[i], bb, eps, "action-space BO")[0]]; upd()
    res.n_total = bb.n - n0; return res


def grid_search(spec, scm, x, bb, eps, max_actions=15000, blocks=True, valid_fn=None):
    """Brute force over a regular grid (the option of Karimi et al. 2020 for non-differentiable models), with our
    robust-validity criterion or with `valid_fn(A) -> bool array` (e.g. their GP-SCM criterion)."""
    nb = 2 ** spec.nB if blocks else 1; bins = max(3, int(np.floor((max_actions / nb) ** (1 / spec.nA))))
    g = [np.unique(np.r_[0.0, np.linspace(spec.lo[i], spec.hi[i], bins)]) for i in range(spec.nA)] + [[0.0, 1.0] if blocks else [0.0]] * spec.nB
    A = np.array(list(itertools.product(*g))); c = spec.cost(A); res = Result(); n0 = bb.n
    if valid_fn is not None: ok = valid_fn(A)
    else:
        ok = np.empty(len(A), bool)
        for i in range(0, len(A), 2000):
            Xc = scm.counterfactual(x, A[i:i + 2000]); ok[i:i + 2000] = robust(bb.margin(Xc.reshape(-1, spec.d), "grid").reshape(Xc.shape[:2]), eps)[0] >= 0
    if ok.any(): i = np.where(ok)[0][np.argmin(c[ok])]; res.action, res.cost = A[i].copy(), c[i]
    res.n_total = bb.n - n0; res.grid = A; return res
