"""Problems of Section 7: graph, data, black box and (when it exists) the true SCM.

Synthetic: perioperative (running example), synth3 (Karimi et al. 2020, 3 variables), random DAGs, loan (7 variables).
Real data: German Credit, Adult, Sachs. For real data there is no true SCM, so "validity under the true SCM"
cannot be computed; experiments use a held-out re-fit as a proxy and say so.
"""
from dataclasses import dataclass
import os, warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .scm import Spec, TrueSCM
warnings.filterwarnings("ignore")
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw")
sig = lambda z: 1 / (1 + np.exp(-z))


class BlackBox:
    """Query-only access to the model. margin(X) = logit(h(X)) - logit(tau); every row is one counted query.
    Each call is logged with its purpose (the trace of Section 5)."""
    def __init__(self, proba, tau=0.5): self.proba, self.tau, self.n, self.log = proba, tau, 0, []

    def margin(self, X, purpose="other"):
        X = np.atleast_2d(X); self.n += len(X); self.log.append((purpose, len(X)))
        p = np.clip(self.proba(X), 1e-6, 1 - 1e-6); return np.log(p / (1 - p)) - np.log(self.tau / (1 - self.tau))

    def by_purpose(self):
        out = {}
        for p, k in self.log: out[p] = out.get(p, 0) + k
        return out


@dataclass
class Case:
    name: str
    spec: Spec
    X: np.ndarray                       # observational data used to fit the SCM
    proba: object                       # the black box h(x)
    true: TrueSCM = None                # ground truth, synthetic cases only
    pool: np.ndarray = None             # individuals to explain (held out from X)
    kind: str = "linear"                # mechanism family suggested for the estimates
    notes: str = ""

    def blackbox(self): return BlackBox(self.proba)

    def individuals(self, n, rng, band=(-3.0, -0.3)):
        """n individuals with an unfavourable prediction, margin inside `band`."""
        m = BlackBox(self.proba).margin(self.pool); ok = np.where((m > band[0]) & (m < band[1]))[0]
        return self.pool[rng.choice(ok, min(n, len(ok)), replace=False)]


def _fit(X, y, model, seed=0):
    if model == "logistic": clf = make_pipeline(StandardScaler(), LogisticRegression(C=100.0, max_iter=2000))
    elif model == "gbm": clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed)
    else: clf = make_pipeline(StandardScaler(), MLPClassifier((16, 16), max_iter=3000, random_state=seed))
    clf.fit(X, y); return lambda Z, clf=clf: clf.predict_proba(Z)[:, 1]


def _sample(spec, g, noise, n, rng, link=None):
    X = np.zeros((n, spec.d))
    for j in spec.order:
        u = noise[j](n, rng); z = (g[j](X[:, spec.pa[j]]) if spec.pa[j] else 0.0) + u
        X[:, j] = link[j](z) if link and j in link else z
    return X


def _lin(w, b=0.0): return lambda Z, w=np.asarray(w, float), b=b: Z @ w + b


def _synthetic(name, spec, g, noise, label_logit, n_obs, seed, model, kind, link=None, link_inv=None, notes="", n_bb=6000):
    rng = np.random.default_rng(seed); Xb = _sample(spec, g, noise, n_bb, rng, link)
    yb = rng.random(n_bb) < sig(label_logit(Xb)); proba = _fit(Xb[:, spec.inp], yb, model, seed)
    full = lambda Z, proba=proba, inp=spec.inp: proba(np.atleast_2d(Z)[:, inp])
    return Case(name, spec, _sample(spec, g, noise, n_obs, rng, link), full, TrueSCM(spec, g, link, link_inv),
                _sample(spec, g, noise, 3000, rng, link), kind, notes)


# ---------------------------------------------------------------- running example
def perioperative(n_obs=300, seed=0, model="mlp"):
    """Linear running example: Age, HbA1c, eGFR, Hb, SBP. Actions: lower HbA1c, raise Hb, block Hb -> SBP."""
    names = ["Age", "HbA1c", "eGFR", "Hb", "SBP"]
    spec = Spec(names, {"eGFR": ["Age", "HbA1c"], "Hb": ["Age", "eGFR"], "SBP": ["Hb"]}, ["HbA1c", "Hb"],
                {"HbA1c": (-1.5, 0.0), "Hb": (0.0, 2.0)}, {"HbA1c": 1.0, "Hb": 0.5}, [("Hb", "SBP")])
    g = {2: _lin([-0.5, -6.0], 140.0), 3: _lin([-0.02, 0.06], 8.56), 4: _lin([6.0], 87.0)}
    sd = [10.0, 1.2, 8.0, 0.8, 10.0]; mu = [62.0, 7.5, 0, 0, 0]
    noise = {j: (lambda n, rng, j=j: mu[j] + sd[j] * rng.standard_normal(n)) for j in range(5)}
    beta = np.array([-0.03, -1.1, 0.02, 0.8, -0.08])
    c = _synthetic("perioperative", spec, g, noise, lambda X: 11.89 + X @ beta, n_obs, seed, model, "linear")
    c.beta = beta; return c


# ---------------------------------------------------------------- Karimi et al. (2020), 3 variables
def synth3(kind="linear", n_obs=250, seed=0, model="mlp"):
    """3-variable SCMs of Karimi et al. (NeurIPS 2020), equations taken from their public code (loadSCM.py).
    The label rule below (logistic in the standardised sum) is OUR choice; theirs must be checked before reporting."""
    spec = Spec(["x1", "x2", "x3"], {"x2": ["x1"], "x3": ["x1", "x2"]}, ["x1", "x2", "x3"],
                {v: (-4.0, 4.0) for v in ["x1", "x2", "x3"]}, None, [("x1", "x2"), ("x2", "x3")], hard=("x1", "x2", "x3"))
    def mog(n, rng): return np.where(rng.random(n) < 0.5, -2 + 1.5 * rng.standard_normal(n), 1 + rng.standard_normal(n))
    if kind == "linear":
        g = {1: _lin([-1.0]), 2: _lin([0.05, 0.25])}; s2 = 1.0
    else:
        g = {1: lambda Z: 3 / (1 + np.exp(-2 * Z[..., 0])) - 1, 2: lambda Z: -0.5 * (0.1 * Z[..., 0] + 0.5 * Z[..., 1] ** 2)}; s2 = np.sqrt(0.1)
    noise = {0: mog, 1: lambda n, rng: s2 * rng.standard_normal(n), 2: lambda n, rng: rng.standard_normal(n)}
    X0 = _sample(spec, g, noise, 5000, np.random.default_rng(99)); s = X0.sum(1); m, sd = s.mean(), s.std()
    spec.kap = 1 / X0.std(0)
    return _synthetic(f"synth3-{kind}", spec, g, noise, lambda X: 2.5 * (X.sum(1) - m) / sd, n_obs, seed, model,
                      "linear" if kind == "linear" else "rff", notes="label rule is ours; check against Karimi et al. (2020)")


# ---------------------------------------------------------------- random DAGs
def random_dag(d=10, n_obs=400, seed=0, model="mlp", nonlinear=False, n_block=2):
    """Random DAG with d nodes. Only the roots are actionable. Edge weights have random signs, so some paths work
    against the action; the blockable edges are chosen among the edges below the actionable roots."""
    rng = np.random.default_rng(1000 + seed); names = [f"v{i}" for i in range(d)]; r = max(2, d // 5); parents, w = {}, {}
    for j in range(r, d):
        pa = sorted(rng.choice(j, size=min(j, rng.integers(1, 4)), replace=False)); parents[names[j]] = [names[k] for k in pa]
        w[j] = rng.uniform(0.5, 1.2, len(pa)) * rng.choice([-1, 1], len(pa))
    desc_edges = [(names[k], names[j]) for j in range(r, d) for k in [names.index(p) for p in parents[names[j]]]]
    blk = [desc_edges[i] for i in rng.choice(len(desc_edges), min(n_block, len(desc_edges)), replace=False)]
    spec = Spec(names, parents, names[:r], {v: (-3.0, 3.0) for v in names[:r]}, None, blk)
    f = (lambda z: 2 * np.tanh(z / 2)) if nonlinear else (lambda z: z)
    g = {j: (lambda Z, w=w[j]: f(Z @ w)) for j in w}
    noise = {j: (lambda n, rng: rng.standard_normal(n)) for j in range(d)}
    beta = rng.uniform(0.3, 1.0, d) * rng.choice([-1, 1], d); X0 = _sample(spec, g, noise, 4000, np.random.default_rng(5)); s = X0 @ beta
    c = _synthetic(f"dag{d}" + ("-nl" if nonlinear else ""), spec, g, noise, lambda X: 2.0 * (X @ beta - s.mean()) / s.std(), n_obs, seed, model,
                   "rff" if nonlinear else "linear")
    c.beta = 2.0 * beta / s.std(); return c


# ---------------------------------------------------------------- loan, 7 variables (Karimi et al. 2020, App. E.1.2)
def loan(n_obs=250, seed=0, model="mlp"):
    names = ["G", "A", "E", "L", "D", "I", "S"]
    spec = Spec(names, {"E": ["G", "A"], "L": ["G", "A"], "D": ["G", "A", "L"], "I": ["G", "A", "E"], "S": ["I"]},
                ["E", "L", "I", "S"], {"E": (0.0, 0.6), "L": (-6.0, 0.0), "I": (0.0, 6.0), "S": (0.0, 12.0)}, None,
                [("L", "D"), ("I", "S")], hard=("E", "L", "I", "S"))
    g = {2: lambda Z: -1 + 0.5 * Z[..., 0] + sig(0.1 * Z[..., 1]),
         3: lambda Z: 1 + 0.01 * (Z[..., 1] - 5) * (5 - Z[..., 1]) + Z[..., 0],
         4: lambda Z: -1 + 0.1 * Z[..., 1] + 2 * Z[..., 0] + Z[..., 2],
         5: lambda Z: -4 + 0.1 * (Z[..., 1] + 35) + 2 * Z[..., 0] + Z[..., 0] * Z[..., 2],
         6: lambda Z: -4 + 1.5 * (Z[..., 0] > 0) * Z[..., 0]}
    noise = {0: lambda n, rng: (rng.random(n) < 0.5).astype(float), 1: lambda n, rng: -35 + rng.gamma(10, 3.5, n),
             2: lambda n, rng: 0.5 * rng.standard_normal(n), 3: lambda n, rng: 2 * rng.standard_normal(n),
             4: lambda n, rng: 3 * rng.standard_normal(n), 5: lambda n, rng: 2 * rng.standard_normal(n), 6: lambda n, rng: 5 * rng.standard_normal(n)}
    link = {2: lambda z: -0.5 + sig(z)}; link_inv = {2: lambda x: np.log((np.clip(x, -0.499, 0.499) + 0.5) / (0.5 - np.clip(x, -0.499, 0.499)))}
    X0 = _sample(spec, g, noise, 5000, np.random.default_rng(5), link); spec.kap = 1 / np.ptp(X0[:, spec.act], axis=0)
    lab = lambda X: 0.3 * (-X[:, 3] - X[:, 4] + X[:, 5] + X[:, 6] + X[:, 5] * X[:, 6])
    c = _synthetic("loan7", spec, g, noise, lab, n_obs, seed, model, "rff", link, link_inv)
    c.link, c.link_inv = link, link_inv; return c


# ---------------------------------------------------------------- real data
def _real(name, spec, X, y, seed, n_obs, notes, kind="rff"):
    rng = np.random.default_rng(seed); p = rng.permutation(len(X)); X, y = X[p], y[p]; n3 = len(X) // 3
    proba = _fit(X[:n3][:, spec.inp], y[:n3], "gbm", seed); full = lambda Z, proba=proba, inp=spec.inp: proba(np.atleast_2d(Z)[:, inp])
    spec.kap = 1 / (X[:, spec.act].std(0) + 1e-9)
    c = Case(name, spec, X[n3:n3 + n_obs], full, None, X[2 * n3:], kind, notes); c.X_holdout = X[n3 + n_obs:2 * n3][:n_obs]; return c


def german(n_obs=300, seed=0):
    """German Credit (UCI). Graph over sex, age, credit amount and duration as in Karimi et al. (2021): sex -> credit,
    age -> credit, credit -> duration. TO VERIFY against their figure before reporting."""
    import pandas as pd
    df = pd.read_csv(os.path.join(RAW, "german.csv"), header=None)
    X = np.c_[df[8].isin(["A92", "A95"]).astype(float), df[12].astype(float), np.log(df[4].astype(float)), df[1].astype(float)]
    spec = Spec(["sex", "age", "log_credit", "duration"], {"log_credit": ["sex", "age"], "duration": ["log_credit"]},
                ["log_credit", "duration"], {"log_credit": (-1.5, 0.0), "duration": (-24.0, 0.0)}, None, [("log_credit", "duration")])
    return _real("german", spec, X, (df[20] == 1).values, seed, n_obs, "graph to verify; no true SCM", "linear")


def adult(n_obs=1000, seed=0):
    """Adult (UCI). Simplified graph in the spirit of Nabi & Shpitser (2018) and Chiappa (2019): sex, age, married are
    upstream; education <- sex, age; hours <- sex, age, married, education. TO VERIFY before reporting."""
    import pandas as pd
    df = pd.read_csv(os.path.join(RAW, "adult.csv"), header=None, skipinitialspace=True)
    X = np.c_[(df[9] == "Female").astype(float), df[0].astype(float), df[5].astype(str).str.startswith("Married").astype(float),
              df[4].astype(float), df[12].astype(float)]
    spec = Spec(["sex", "age", "married", "education", "hours"], {"education": ["sex", "age"], "hours": ["sex", "age", "married", "education"]},
                ["education", "hours"], {"education": (0.0, 5.0), "hours": (0.0, 25.0)}, None, [("education", "hours")])
    return _real("adult", spec, X, df[14].astype(str).str.startswith(">50K").values, seed, n_obs, "graph to verify; no true SCM")


def sachs(n_obs=400, seed=0):
    """Sachs et al. (2005) protein data with the 17-edge consensus graph. The classification task (Akt above its median)
    is OUR placeholder so that there is a black box to explain; Akt is removed from the inputs (15 edges remain)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(RAW, "sachs.csv")); df.columns = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"]
    L = np.log(df.clip(lower=1e-3)); y = (L["Akt"] > L["Akt"].median()).values; names = [c for c in L.columns if c != "Akt"]
    parents = {"Raf": ["PKC", "PKA"], "Mek": ["PKC", "PKA", "Raf"], "Erk": ["PKA", "Mek"], "Jnk": ["PKC", "PKA"], "P38": ["PKC", "PKA"],
               "PKA": ["PKC"], "PIP2": ["Plcg", "PIP3"], "PIP3": ["Plcg"]}
    spec = Spec(names, parents, ["PKC", "PKA", "Mek"], {v: (-2.0, 2.0) for v in ["PKC", "PKA", "Mek"]}, None, [("Mek", "Erk"), ("PKA", "Erk")])
    return _real("sachs", spec, L[names].values, y, seed, n_obs, "classification task is a placeholder; no true SCM")


CASES = {"perioperative": perioperative, "synth3-linear": lambda **k: synth3("linear", **k), "synth3-anm": lambda **k: synth3("anm", **k),
         "dag10": lambda **k: random_dag(10, **k), "dag20": lambda **k: random_dag(20, **k), "dag30": lambda **k: random_dag(30, **k),
         "loan7": loan, "german": german, "adult": adult, "sachs": sachs}
