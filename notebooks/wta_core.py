"""Why This Action? - núcleo reutilizable (mismo código del cuaderno 01, empaquetado como funciones).

Ejemplo perioperatorio lineal: Age, HbA1c, eGFR, Hb, SBP.
Un plan es [cambio en HbA1c, cambio en Hb, bloquear Hb -> SBP (0/1)].
"""
import numpy as np, warnings
from collections import Counter
from scipy import stats
from scipy.stats import norm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel as Const
warnings.filterwarnings("ignore")

VARS = ["Age", "HbA1c", "eGFR", "Hb", "SBP"]
IDX = {v: i for i, v in enumerate(VARS)}
TRUE_EDGES = {("Age", "eGFR"): -0.5, ("HbA1c", "eGFR"): -6.0, ("eGFR", "Hb"): 0.06,
              ("Age", "Hb"): -0.02, ("Hb", "SBP"): 6.0}
INTERCEPT = {"Age": 62.0, "HbA1c": 7.5, "eGFR": 140.0, "Hb": 8.56, "SBP": 87.0}
NOISE_SD = {"Age": 10.0, "HbA1c": 1.2, "eGFR": 8.0, "Hb": 0.8, "SBP": 10.0}
BETA = np.array([-0.03, -1.1, 0.02, 0.8, -0.08]); B0 = 11.89
H_RANGE, B_RANGE = (-1.5, 0.0), (0.0, 2.0)
BLOCK_EDGE = ("Hb", "SBP"); JB, KB = IDX["SBP"], IDX["Hb"]
X0 = np.array([68.0, 8.5, 55.0, 10.5, 150.0])
sigmoid = lambda z: 1 / (1 + np.exp(-z))


def W_from_edges(edges):
    W = np.zeros((5, 5))
    for (k, j), w in edges.items(): W[IDX[j], IDX[k]] = w
    return W
W_TRUE = W_from_edges(TRUE_EDGES)


def sample_world(n, rng):
    X = np.zeros((n, 5))
    for j, v in enumerate(VARS):
        X[:, j] = INTERCEPT[v] + X @ W_TRUE[j] + NOISE_SD[v] * rng.standard_normal(n)
    return X


def true_logit(X): return B0 + X @ BETA


class Problem:
    """La caja negra del hospital. Solo se accede con .blackbox(), que cuenta las consultas."""
    def __init__(self, seed=7, n=6000):
        rng = np.random.default_rng(seed)
        Xb = sample_world(n, rng); yb = rng.random(n) < sigmoid(true_logit(Xb))
        self.scaler = StandardScaler().fit(Xb)
        self.clf = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=3000, random_state=0).fit(self.scaler.transform(Xb), yb)
    def blackbox(self): return BlackBox(self)


class BlackBox:
    def __init__(self, prob, tau=0.5): self.prob, self.n, self.tau = prob, 0, tau
    def margin(self, X):
        X = np.atleast_2d(X); self.n += len(X)
        p = np.clip(self.prob.clf.predict_proba(self.prob.scaler.transform(X))[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)) - np.log(self.tau / (1 - self.tau))


def plan_cost(plans): return np.abs(plans[:, 0]) + 0.5 * np.abs(plans[:, 1])


def make_worlds(W, block_ok=None, gid=None):
    W = W if W.ndim == 3 else W[None]
    return dict(W=W, block_ok=np.ones(len(W)) if block_ok is None else np.asarray(block_ok, float), gid=gid)
TRUE_WORLD = make_worlds(W_TRUE)


def counterfactual(x0, worlds, plans, failed_edges=()):
    """(C, nW, 5): el paciente x0 después de cada plan en cada versión del modelo causal."""
    W, C, nW = worlds["W"], len(plans), len(worlds["W"])
    delta = np.zeros((C, 5)); delta[:, IDX["HbA1c"]] = plans[:, 0]; delta[:, IDX["Hb"]] = plans[:, 1]
    keep = np.ones((5, 5))
    for (k, j) in failed_edges: keep[IDX[j], IDX[k]] = 0
    blocked = plans[:, 2][:, None] * worlds["block_ok"][None, :]
    dev = np.zeros((C, nW, 5))
    for _ in range(worlds.get("sweeps", 1)):              # 1 pasada si el orden causal es el de VARS; 5 para cualquier DAG
        for j in range(5):
            contrib = dev * (W[:, j, :] * keep[j])[None]
            if j == JB: contrib[:, :, KB] *= (1 - blocked)
            dev[:, :, j] = contrib.sum(-1) + delta[:, j, None]
    return x0 + dev


# ---------------- Capa 1: grafo ----------------
def ols(X, j, pa):
    """Coeficientes, errores estándar y grados de libertad de X_j sobre sus padres pa."""
    n = len(X); Z = np.column_stack([np.ones(n), X[:, pa]])
    beta, *_ = np.linalg.lstsq(Z, X[:, j], rcond=None)
    res = X[:, j] - Z @ beta; df = n - Z.shape[1]
    se = np.sqrt(res @ res / df * np.diag(np.linalg.inv(Z.T @ Z)))
    return beta[1:], se[1:], df, np.sqrt(res @ res / df)


def select_graph(X, alpha=0.01):
    edges = []
    for j in range(1, 5):
        b, se, df, _ = ols(X, j, list(range(j)))
        p = 2 * stats.t.sf(np.abs(b / se), df)
        edges += [(VARS[k], VARS[j]) for k in range(j) if p[k] < alpha]
    return frozenset(edges)


def bootstrap_graphs(X, B=400, rng=None):
    rng = rng or np.random.default_rng(0)
    cnt = Counter(select_graph(X[rng.integers(0, len(X), len(X))]) for _ in range(B))
    return [(g, c / B) for g, c in cnt.most_common()]


# ---------------- Capa 2: mecanismos ----------------
def fit_mechanisms(X, graph):
    W = np.zeros((5, 5))
    for j in range(5):
        pa = sorted(IDX[k] for (k, jj) in graph if IDX[jj] == j)
        if pa: W[j, pa] = ols(X, j, pa)[0]
    return W


def sample_worlds(X, graphs, n_worlds, rng, fail_prob=None, block_fail=0.0):
    probs = np.array([w for _, w in graphs]); gid = rng.choice(len(graphs), n_worlds, p=probs / probs.sum())
    W = np.stack([fit_mechanisms(X[rng.integers(0, len(X), len(X))], graphs[g][0]) for g in gid])
    for (k, j), pr in (fail_prob or {}).items():
        W[rng.random(n_worlds) < pr, IDX[j], IDX[k]] = 0.0
    return make_worlds(W, rng.random(n_worlds) >= block_fail, gid)


def confidence_box_worlds(X, alpha=0.10):
    """Garantía formal (caso lineal, orden causal conocido).
    Intervalos de confianza SIMULTÁNEOS (Bonferroni) para los 6 coeficientes por los que puede viajar
    el efecto del plan. Con probabilidad >= 1 - alpha el modelo verdadero está dentro de la caja.
    Devuelve los 64 vértices de la caja como versiones del modelo causal."""
    rel = [(k, j) for j in range(2, 5) for k in range(1, j)]          # flechas entre HbA1c, eGFR, Hb, SBP
    W0, lo, hi = np.zeros((5, 5)), {}, {}
    for j in range(1, 5):
        pa = list(range(j)); b, se, df, _ = ols(X, j, pa)
        t = stats.t.ppf(1 - alpha / (2 * len(rel)), df)
        for i, k in enumerate(pa):
            W0[j, k] = b[i]
            if (k, j) in rel: lo[(k, j)], hi[(k, j)] = b[i] - t * se[i], b[i] + t * se[i]
    Ws = []
    for bits in range(2 ** len(rel)):
        W = W0.copy()
        for i, (k, j) in enumerate(rel): W[j, k] = hi[(k, j)] if (bits >> i) & 1 else lo[(k, j)]
        Ws.append(W)
    return make_worlds(np.stack(Ws)), (lo, hi)


# ---------------- Validez ----------------
def robust_margin(M, eps):
    k = int(np.floor(eps * M.shape[1])); idx = np.argpartition(M, k, axis=1)[:, k]
    return M[np.arange(len(M)), idx], idx


def success_prob(m):
    n, p = len(m), float(np.mean(m >= 0)); z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0, c - h), min(1, c + h)


# ---------------- Optimización bayesiana compuesta ----------------
def make_gp():
    kernel = Const(1.0) * DotProduct(sigma_0=1.0) + Const(0.5) * RBF(length_scale=2.0) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)


def thin(X, y, cap=220, seed=0):
    if len(X) <= cap: return X, y
    idx = np.r_[np.random.default_rng(seed).choice(len(X) - 80, cap - 80, replace=False), np.arange(len(X) - 80, len(X))]
    return X[idx], y[idx]


def sample_plans(n, rng, around=None, scale=0.12):
    if around is None:
        return np.column_stack([rng.uniform(*H_RANGE, n), rng.uniform(*B_RANGE, n), rng.integers(0, 2, n)])
    return np.column_stack([np.clip(around[0] + scale * rng.standard_normal(n), *H_RANGE),
                            np.clip(around[1] + scale * rng.standard_normal(n), *B_RANGE), np.full(n, around[2])])


def solve_bo(prob, x0, worlds, eps, bb, n_iter=40, n_cand=1200, failed_edges=(), data=None, seed=0, refine=1):
    """Devuelve (plan certificado o None, info). info: q_search, q_cert, hist, data, gp, cands, pf."""
    sc = prob.scaler; rng = np.random.default_rng(seed); nW = len(worlds["W"]); q0 = bb.n
    cf = lambda P: counterfactual(x0, worlds, P, failed_edges)
    if data is None:
        P0 = sample_plans(8, rng); X0_ = np.vstack([x0[None], cf(P0)[np.arange(8), rng.integers(0, nW, 8)]])
        data = [X0_, bb.margin(X0_)]
    Xq, yq = data
    corners = np.array([[h, b, k] for h in H_RANGE for b in B_RANGE for k in (0, 1)])
    cands = np.vstack([corners, sample_plans(n_cand, rng)]); costs = plan_cost(cands)

    def assess():
        Xt, yt = thin(Xq, yq); gp = make_gp().fit(sc.transform(Xt), yt)
        Xc = cf(cands); flat = Xc.reshape(-1, 5)
        mu = np.concatenate([gp.predict(sc.transform(flat[i:i + 40000])) for i in range(0, len(flat), 40000)]).reshape(len(cands), nW)
        mu_b, w_b = robust_margin(mu, eps); Xb = Xc[np.arange(len(cands)), w_b]
        sd_b = gp.predict(sc.transform(Xb), return_std=True)[1]
        return gp, Xc, Xb, norm.cdf(mu_b / np.maximum(sd_b, 1e-6))

    hist = []
    for it in range(n_iter):
        gp, Xc, Xb, pf = assess(); feas = pf >= 0.9
        c_star = costs[feas].min() if feas.any() else np.inf
        acq = pf if not np.isfinite(c_star) else np.maximum(c_star - costs, 0) * pf
        if acq.max() <= 1e-9: acq = pf * (1 - pf)
        i = int(np.argmax(acq))
        Xq = np.vstack([Xq, Xb[i][None]]); yq = np.append(yq, bb.margin(Xb[i]))
        hist.append((bb.n - q0, c_star))
        if it in [n_iter * (r + 1) // (refine + 1) for r in range(refine)] and feas.any():   # refinar cerca del mejor
            inc = cands[feas][np.argmin(costs[feas])]
            cands = np.vstack([cands, sample_plans(n_cand // 2, rng, around=inc)]); costs = plan_cost(cands)
    best, q_search = None, bb.n - q0
    for _ in range(8):
        gp, Xc, Xb, pf = assess(); feas = pf >= 0.9
        if not feas.any(): break
        i = int(np.flatnonzero(feas)[np.argmin(costs[feas])])
        mi = bb.margin(Xc[i]); low = np.argsort(mi)[: max(8, int(2 * eps * nW))]
        Xq = np.vstack([Xq, Xc[i][low]]); yq = np.append(yq, mi[low])
        if robust_margin(mi[None], eps)[0][0] >= 0: best = cands[i].copy(); break
    gp, Xc, Xb, pf = assess()
    return best, dict(q_search=q_search, q_cert=bb.n - q0 - q_search, hist=hist, data=[Xq, yq], gp=gp, cands=cands, pf=pf)


def fmt(plan):
    return "sin plan" if plan is None else f"HbA1c {plan[0]:+.2f}, Hb {plan[1]:+.2f}, bloqueo={'sí' if plan[2] else 'no'}"


# ---------------- Descubrimiento causal SIN orden dado ----------------
import itertools as _it
_SUBSETS = [tuple(c) for r in range(5) for c in _it.combinations(range(5), r)]

def _local_bic(X):
    """BIC gaussiano de cada variable con cada posible conjunto de padres (5 x 16 regresiones)."""
    n = len(X); out = {}
    for j in range(5):
        for S in _SUBSETS:
            if j in S: continue
            Z = np.column_stack([np.ones(n)] + [X[:, k] for k in S]); r = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
            out[(j, S)] = -0.5 * n * np.log(r @ r / n) - 0.5 * np.log(n) * (len(S) + 1)
    return out

def best_dags(X, forbidden=(), tol=1e-6):
    """Búsqueda exacta: el mejor grafo por BIC entre TODOS los órdenes posibles.
    forbidden: flechas (padre, hijo) que el experto descarta. Devuelve todos los grafos empatados:
    con datos gaussianos, los grafos de una misma clase de equivalencia de Markov empatan."""
    sc = _local_bic(X); fb = {(IDX[k], IDX[j]) for k, j in forbidden}; found = {}
    for perm in _it.permutations(range(5)):
        tot, edges = 0.0, []
        for pos, j in enumerate(perm):
            pred = [k for k in perm[:pos] if (k, j) not in fb]
            S = max((S for S in _SUBSETS if set(S) <= set(pred)), key=lambda S: sc[(j, S)])
            tot += sc[(j, S)]; edges += [(VARS[k], VARS[j]) for k in S]
        found[frozenset(edges)] = tot
    mx = max(found.values()); return [g for g, v in found.items() if v >= mx - tol]

def bootstrap_dags(X, B=100, rng=None, forbidden=()):
    rng = rng or np.random.default_rng(0); cnt = Counter()
    for _ in range(B):
        gs = best_dags(X[rng.integers(0, len(X), len(X))], forbidden)
        for g in gs: cnt[g] += 1 / len(gs)                 # el peso se reparte entre los empatados
    return [(g, c / B) for g, c in cnt.most_common()]

def sample_worlds_general(X, graphs, n_worlds, rng):
    w = sample_worlds(X, graphs, n_worlds, rng); w["sweeps"] = 5; return w
