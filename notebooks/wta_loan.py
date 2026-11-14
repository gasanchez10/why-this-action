"""Caso de préstamos de 7 variables de Karimi, von Kügelgen, Schölkopf y Valera (NeurIPS 2020, Apéndice E.1.2).

Contiene: el modelo causal verdadero, la caja negra, mecanismos NO lineales con incertidumbre
(nuestra versión por bootstrap y la de ellos por procesos gaussianos) y los buscadores.
Acciones: intervenciones duras do(X_j = x_j + delta_j) sobre educación, monto, ingreso y ahorro,
igual que en su artículo. El grafo se supone conocido, igual que en su artículo.
"""
import numpy as np, itertools, warnings
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel as Const
warnings.filterwarnings("ignore")

VARS = ["G", "A", "E", "L", "D", "I", "S"]             # género, edad, educación, monto, duración, ingreso, ahorro
IDX = {v: i for i, v in enumerate(VARS)}
PARENTS = {"E": ["G", "A"], "L": ["A", "G"], "D": ["A", "G", "L"], "I": ["A", "G", "E"], "S": ["I"]}
ACTIONABLE = ["E", "L", "I", "S"]; ACT = [IDX[v] for v in ACTIONABLE]
sig = lambda z: 1 / (1 + np.exp(-z))


# ---------------- El mundo verdadero (ecuaciones E.11 a E.18) ----------------
def f_true(v, X):
    G, A, E, L, I = (X[..., IDX[k]] for k in "GAELI")
    if v == "E": return -1 + 0.5 * G + sig(0.1 * A)            # va DENTRO de una sigmoide junto con el ruido
    if v == "L": return 1 + 0.01 * (A - 5) * (5 - A) + G
    if v == "D": return -1 + 0.1 * A + 2 * G + L
    if v == "I": return -4 + 0.1 * (A + 35) + 2 * G + G * E
    if v == "S": return -4 + 1.5 * (I > 0) * I

def sample_loan(n, rng):
    X = np.zeros((n, 7))
    X[:, 0] = rng.random(n) < 0.5
    X[:, 1] = -35 + rng.gamma(10, 3.5, n)
    X[:, 2] = -0.5 + sig(f_true("E", X) + 0.5 * rng.standard_normal(n))
    for v, sd in [("L", 2.0), ("D", 3.0), ("I", 2.0), ("S", 5.0)]:
        X[:, IDX[v]] = f_true(v, X) + sd * rng.standard_normal(n)
    return X

def label_logit(X):
    L, D, I, S = (X[..., IDX[k]] for k in "LDIS")
    return 0.3 * (-L - D + I + S + I * S)

def true_counterfactual(x, deltas):
    """Contrafactual con el modelo verdadero. deltas: (C, 4) sobre E, L, I, S; 0 = no se interviene."""
    C = len(deltas); X = np.tile(x, (C, 1)); xf = x[None]
    for v in ["E", "L", "D", "I", "S"]:
        j = IDX[v]
        if v == "E": u = np.log((x[j] + 0.5) / (0.5 - x[j])) - f_true("E", xf)        # el ruido se recupera invirtiendo la sigmoide
        else: u = x[j] - f_true(v, xf)
        nat = (-0.5 + sig(f_true("E", X) + u)) if v == "E" else f_true(v, X) + u
        if v in ACTIONABLE:
            d = deltas[:, ACTIONABLE.index(v)]; X[:, j] = np.where(d != 0, x[j] + d, nat)
        else: X[:, j] = nat
    return X


class LoanProblem:
    def __init__(self, seed=3, n=8000):
        rng = np.random.default_rng(seed)
        Xb = sample_loan(n, rng); yb = rng.random(n) < sig(label_logit(Xb))
        self.scaler = StandardScaler().fit(Xb)
        self.clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=3000, random_state=0).fit(self.scaler.transform(Xb), yb)
        self.lo, self.hi = np.percentile(Xb, 1, axis=0), np.percentile(Xb, 99, axis=0)
        self.range = (Xb.max(0) - Xb.min(0))[ACT]                                     # el costo se normaliza por el rango
    def blackbox(self): return BlackBox(self)
    def cost(self, deltas): return np.sqrt(((deltas / self.range) ** 2).sum(-1))       # norma L2 normalizada, como en su artículo

class BlackBox:
    def __init__(self, prob): self.prob, self.n = prob, 0
    def proba(self, X):
        X = np.atleast_2d(X); self.n += len(X)
        return np.concatenate([self.prob.clf.predict_proba(self.prob.scaler.transform(X[i:i + 200000]))[:, 1] for i in range(0, len(X), 200000)])
    def margin(self, X):
        p = np.clip(self.proba(X), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


# ---------------- Mecanismos no lineales con incertidumbre: NUESTRA versión ----------------
class BootstrapSCM:
    """Cada mecanismo es una regresión ridge sobre rasgos aleatorios de Fourier (aproxima un kernel RBF)
    más términos lineales. La incertidumbre son n_worlds re-ajustes sobre remuestreos de los datos."""
    def __init__(self, X, n_worlds=50, D=60, ls=1.5, lam=1.0, seed=0):
        rng = np.random.default_rng(seed); n = len(X); self.nW = n_worlds; self.m = {}
        boots = [rng.integers(0, n, n) for _ in range(n_worlds)]
        for v, pa in PARENTS.items():
            p = [IDX[k] for k in pa]; mu, sd = X[:, p].mean(0), X[:, p].std(0) + 1e-9
            Om = rng.standard_normal((D, len(p))) / ls; b = rng.uniform(0, 2 * np.pi, D)
            phi = lambda Z, mu=mu, sd=sd, Om=Om, b=b: np.concatenate(
                [np.sqrt(2 / D) * np.cos(((Z - mu) / sd) @ Om.T + b), (Z - mu) / sd, np.ones(Z.shape[:-1] + (1,))], -1)
            P = phi(X[:, p]); y = X[:, IDX[v]]; R = lam * np.eye(P.shape[1]); R[-1, -1] = 1e-8
            th = np.stack([np.linalg.solve(P[i].T @ P[i] + R, P[i].T @ y[i]) for i in boots])
            self.m[v] = (p, phi, th, np.linalg.solve(P.T @ P + R, P.T @ y))
    def counterfactual(self, x, deltas, point=False):
        """(C, nW, 7). Abducción con ruido aditivo: x_j nuevo = x_j + f(padres nuevos) - f(padres factuales)."""
        C, nW = len(deltas), (1 if point else self.nW); X = np.tile(x, (C, nW, 1))
        for v in ["E", "L", "D", "I", "S"]:
            j = IDX[v]; p, phi, th, th0 = self.m[v]; th = th0[None] if point else th
            nat = x[j] + np.einsum("cwd,wd->cw", phi(X[:, :, p]), th) - (phi(x[p][None]) @ th.T)
            if v in ACTIONABLE:
                d = deltas[:, ACTIONABLE.index(v)][:, None]; X[:, :, j] = np.where(d != 0, x[j] + d, nat)
            else: X[:, :, j] = nat
        return X


# ---------------- Mecanismos con incertidumbre: la versión de Karimi et al. (GP-SCM) ----------------
class GPSCM:
    """Un proceso gaussiano con ruido aditivo gaussiano por mecanismo (su ecuación 5).
    Contrafactuales por muestreo ancestral con el posterior del ruido (sus proposiciones 5 y 6)."""
    def __init__(self, X, seed=0):
        self.X = X; self.h = {}
        for v, pa in PARENTS.items():
            p = [IDX[k] for k in pa]; mu, sd = X[:, p].mean(0), X[:, p].std(0) + 1e-9
            y = X[:, IDX[v]]; my = y.mean()
            gp = GaussianProcessRegressor(Const(1.0) * RBF(np.ones(len(p))) + WhiteKernel(0.5), random_state=seed).fit((X[:, p] - mu) / sd, y - my)
            k = gp.kernel_; self.h[v] = (p, mu, sd, my, k.k1.k1.constant_value, np.atleast_1d(k.k1.k2.length_scale), k.k2.noise_level)
    @staticmethod
    def _k(A, B, s2, ls):
        A, B = A / ls, B / ls
        return s2 * np.exp(-0.5 * np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))
    def prepare(self, x):
        """Añade al individuo a los datos (como en su artículo) y precalcula lo que no depende de la acción."""
        self.x = x; self.c = {}; Xa = np.vstack([self.X, x[None]])
        for v, (p, mu, sd, my, s2, ls, n2) in self.h.items():
            Z = (Xa[:, p] - mu) / sd; y = Xa[:, IDX[v]] - my
            K = self._k(Z, Z, s2, ls) + n2 * np.eye(len(Z)); cf = cho_factor(K, lower=True)
            alpha = cho_solve(cf, y); Ainv_FF = cho_solve(cf, np.eye(len(Z))[:, -1])[-1]
            self.c[v] = (Z, cf, alpha, n2 * alpha[-1], n2 - n2 ** 2 * Ainv_FF)        # media y varianza del ruido del individuo
    def sample(self, deltas, M, rng):
        """(C, M, 7): M muestras del contrafactual para cada acción."""
        x = self.x; C = len(deltas); X = np.tile(x, (C, M, 1))
        for v in ["E", "L", "D", "I", "S"]:
            j = IDX[v]; p, mu, sd, my, s2, ls, n2 = self.h[v]; Z, cf, alpha, muF, sF = self.c[v]
            const = np.ptp(X[:, :, p], axis=1).max() == 0                              # ¿los padres no varían entre muestras?
            Zt = ((X[:, 0, p] if const else X[:, :, p].reshape(-1, len(p))) - mu) / sd
            mean = np.empty(len(Zt)); var = np.empty(len(Zt))
            for i in range(0, len(Zt), 20000):
                kt = self._k(Zt[i:i + 20000], Z, s2, ls)
                mean[i:i + 20000] = my + muF + kt @ alpha
                Vv = solve_triangular(cf[0], kt.T, lower=True); var[i:i + 20000] = sF + s2 - (Vv ** 2).sum(0)
            sd_ = np.sqrt(np.maximum(var, 1e-10))
            draw = (mean[:, None] + sd_[:, None] * rng.standard_normal((C, M))) if const else (mean + sd_ * rng.standard_normal(len(Zt))).reshape(C, M)
            draw = np.where((X[:, :, p] == x[p]).all(-1), x[j], draw)                  # solo cambian los descendientes de lo intervenido
            if v in ACTIONABLE:
                d = deltas[:, ACTIONABLE.index(v)][:, None]; X[:, :, j] = np.where(d != 0, x[j] + d, draw)
            else: X[:, :, j] = draw
        return X


# ---------------- Espacio de acciones ----------------
def action_grid(prob, x, bins=10):
    """El espacio discretizado de la fuerza bruta de su artículo: todos los subconjuntos de variables
    accionables, con 'bins' valores por variable dentro del rango de los datos."""
    vals = [np.linspace(prob.lo[j], prob.hi[j], bins) - x[j] for j in ACT]; out = []
    for r in range(1, 5):
        for S in itertools.combinations(range(4), r):
            for combo in itertools.product(*[vals[s] for s in S]):
                d = np.zeros(4); d[list(S)] = combo; out.append(d)
    D = np.array(out); return D[(D != 0).any(1)]

def sample_actions(prob, x, n, rng, around=None, scale=0.08):
    lo, hi = prob.lo[ACT] - x[ACT], prob.hi[ACT] - x[ACT]
    if around is None:
        D = rng.uniform(lo, hi, (n, 4)) * (rng.random((n, 4)) < 0.5)
    else:
        D = np.clip(around + scale * prob.range * rng.standard_normal((n, 4)), lo, hi) * (around != 0)
    return D[(D != 0).any(1)]


# ---------------- Buscadores ----------------
def robust_margin(M, eps):
    k = int(np.floor(eps * M.shape[1])); idx = np.argpartition(M, k, axis=1)[:, k]
    return M[np.arange(len(M)), idx], idx

def brute_force(prob, x, bb, criterion, bins=10):
    """Recorre toda la malla y devuelve la acción válida más barata según 'criterion' (función de acciones -> bool)."""
    D = action_grid(prob, x, bins); ok = np.concatenate([criterion(D[i:i + 1500], bb) for i in range(0, len(D), 1500)])
    return (D[ok][np.argmin(prob.cost(D[ok]))] if ok.any() else None), len(D)

def karimi_criterion(gpscm, M, gamma, rng):
    """E[h] - gamma * sqrt(Var[h]) >= 0.5, por Monte Carlo sobre el contrafactual del GP-SCM (sus ecs. 8 y 9)."""
    def crit(D, bb):
        Xs = gpscm.sample(D, M, rng); h = bb.proba(Xs.reshape(-1, 7)).reshape(len(D), M)
        return h.mean(1) - gamma * h.std(1) >= 0.5
    return crit

def make_gp():
    kernel = Const(1.0) * DotProduct(sigma_0=1.0) + Const(1.0) * RBF(length_scale=2.0) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)

def solve_bo(prob, x, scm, eps, bb, n_iter=35, n_cand=1000, seed=0):
    """Nuestra optimización bayesiana compuesta, con mecanismos no lineales e intervenciones duras."""
    sc = prob.scaler; rng = np.random.default_rng(seed); nW = scm.nW; q0 = bb.n
    cands = sample_actions(prob, x, n_cand, rng); costs = prob.cost(cands)
    P0 = cands[rng.choice(len(cands), 12, replace=False)]
    Xq = np.vstack([x[None], scm.counterfactual(x, P0)[np.arange(12), rng.integers(0, nW, 12)]]); yq = bb.margin(Xq)
    cache = {}
    def assess():
        if cache.get("n") != len(cands): cache.update(n=len(cands), Xc=scm.counterfactual(x, cands))   # el modelo causal es barato y se calcula una vez
        gp = make_gp().fit(sc.transform(Xq), yq); Xc = cache["Xc"]; flat = Xc.reshape(-1, 7)
        mu = np.concatenate([gp.predict(sc.transform(flat[i:i + 40000])) for i in range(0, len(flat), 40000)]).reshape(len(cands), nW)
        mu_b, w_b = robust_margin(mu, eps); Xb = Xc[np.arange(len(cands)), w_b]
        return Xc, Xb, norm.cdf(mu_b / np.maximum(gp.predict(sc.transform(Xb), return_std=True)[1], 1e-6))
    for it in range(n_iter):
        Xc, Xb, pf = assess(); feas = pf >= 0.9
        c_star = costs[feas].min() if feas.any() else np.inf
        acq = pf if not np.isfinite(c_star) else np.maximum(c_star - costs, 0) * pf
        if acq.max() <= 1e-9: acq = pf * (1 - pf)
        i = int(np.argmax(acq)); Xq = np.vstack([Xq, Xb[i][None]]); yq = np.append(yq, bb.margin(Xb[i]))
        if it in (n_iter // 3, 2 * n_iter // 3) and feas.any():
            inc = cands[feas][np.argmin(costs[feas])]
            cands = np.vstack([cands, sample_actions(prob, x, n_cand // 2, rng, around=inc)]); costs = prob.cost(cands)
    best, q_search = None, bb.n - q0
    for _ in range(8):
        Xc, Xb, pf = assess(); feas = pf >= 0.9
        if not feas.any(): break
        i = int(np.flatnonzero(feas)[np.argmin(costs[feas])]); mi = bb.margin(Xc[i]); low = np.argsort(mi)[: max(8, int(2 * eps * nW))]
        Xq = np.vstack([Xq, Xc[i][low]]); yq = np.append(yq, mi[low])
        if robust_margin(mi[None], eps)[0][0] >= 0: best = cands[i].copy(); break
    return best, dict(q_search=q_search, q_cert=bb.n - q0 - q_search)
