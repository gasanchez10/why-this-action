"""Structural causal models, the node-edge counterfactual operator (Definition 4) and path-restricted states.

An *action* is a vector a = [delta_1..delta_nA, b_1..b_nB]: changes on the actionable variables followed by
binary blocks on the blockable edges. All operators are vectorised over C candidate actions and S versions
(estimates) of the SCM and return arrays of shape (C, S, d).
"""
from dataclasses import dataclass, field
import itertools
import numpy as np


@dataclass
class Spec:
    """Graph, action space and cost of a recourse problem."""
    names: list
    parents: dict                                   # name -> list of parent names
    actionable: list
    bounds: dict                                    # name -> (lo, hi) on the change delta
    kappa: dict = None                              # name -> cost weight; cost(a) = sum_j kappa_j |delta_j|
    blockable: list = field(default_factory=list)   # list of (parent, child)
    hard: tuple = ()                                # actionable variables set by hard interventions
    inputs: list = None                             # model inputs (default: all variables)

    def __post_init__(self):
        self.d = len(self.names); self.idx = {v: i for i, v in enumerate(self.names)}
        self.pa = {self.idx[v]: [self.idx[k] for k in self.parents.get(v, [])] for v in self.names}
        self.ch = {j: [c for c in range(self.d) if j in self.pa[c]] for j in range(self.d)}
        self.act = [self.idx[v] for v in self.actionable]; self.nA = len(self.act)
        self.kap = np.array([(self.kappa or {}).get(v, 1.0) for v in self.actionable])
        self.lo = np.array([self.bounds[v][0] for v in self.actionable]); self.hi = np.array([self.bounds[v][1] for v in self.actionable])
        self.edges = [(k, j) for j in range(self.d) for k in self.pa[j]]; self.eidx = {e: i for i, e in enumerate(self.edges)}
        self.blk = [(self.idx[k], self.idx[j]) for k, j in self.blockable]; self.nB = len(self.blk)
        self.hard_idx = {self.idx[v] for v in self.hard}
        self.inp = [self.idx[v] for v in (self.inputs or self.names)]
        order, seen = [], set()
        def visit(j):
            if j in seen: return
            for k in self.pa[j]: visit(k)
            seen.add(j); order.append(j)
        for j in range(self.d): visit(j)
        self.order = order

    def cost(self, A):
        A = np.atleast_2d(A); return (np.abs(A[:, :self.nA]) * self.kap).sum(1)

    def edge(self, k, j): return (self.idx[k], self.idx[j])

    def describe(self, a):
        if a is None: return "no plan"
        parts = [f"{self.actionable[i]} {a[i]:+.3g}" for i in range(self.nA) if a[i] != 0]
        parts += [f"block {self.names[k]}->{self.names[j]}" for i, (k, j) in enumerate(self.blk) if a[self.nA + i] > 0.5]
        return ", ".join(parts) or "do nothing"

    def suffixes(self, j):
        """All directed paths from node j to a model input, as tuples of the nodes AFTER j (() if j is an input)."""
        out = [()] if j in self.inp else []
        for c in self.ch[j]: out += [(c,) + s for s in self.suffixes(c)]
        return out


class SCM:
    """Base class: S versions of the mechanisms x_j = link_j(g_j(pa_j) + u_j). Subclasses implement _g."""
    spec: Spec; S: int
    link = {}; link_inv = {}                                    # optional per-node output transforms (identity by default)

    def _g(self, j, Z):                                         # Z: (..., S, |pa_j|) -> (..., S)
        raise NotImplementedError

    def _lat(self, j, x):                                       # latent scale of node j (inverse link)
        return self.link_inv[j](x) if j in self.link_inv else x

    def _obs(self, j, z): return self.link[j](z) if j in self.link else z

    def cuts(self, A, extra_cut=(), lifted=()):
        """(C, nE) boolean matrix of cut edges: blocks of each action, minus `lifted` blocks, plus `extra_cut` edges."""
        sp = self.spec; A = np.atleast_2d(A); cut = np.zeros((len(A), len(sp.edges)), bool)
        for i, e in enumerate(sp.blk):
            if e not in lifted: cut[:, sp.eidx[e]] = A[:, sp.nA + i] > 0.5
        for e in extra_cut: cut[:, sp.eidx[e]] = True
        return cut

    def counterfactual(self, x, A, extra_cut=(), lifted=(), gamma=None, block_ok=None, shift=None, abduct=True):
        """Node-edge counterfactual operator Phi (Definition 4). Returns (C, S, d).
        gamma: fraction of each change actually achieved (Q1). extra_cut: failing edges D (Q2).
        lifted: blocks that do not hold, R (Q3). block_ok: optional (S,) flags, per version, of whether blocks hold.
        shift: {node: (S,) array} added to the mechanism output of nodes whose inputs changed (conformal layer).
        abduct=False replaces the individual's noise by a draw from the residuals (baseline WITHOUT abduction)."""
        sp = self.spec; A = np.atleast_2d(np.asarray(A, float)); C = len(A); x = np.asarray(x, float)
        delta = A[:, :sp.nA] * (1.0 if gamma is None else np.asarray(gamma)); cut = self.cuts(A, extra_cut, lifted)
        X = np.tile(np.asarray(x, float), (C, self.S, 1)); apos = {j: i for i, j in enumerate(sp.act)}
        for j in sp.order:
            pa = sp.pa[j]; dj = delta[:, apos[j]][:, None] if j in apos else None
            if not pa:
                if dj is not None: X[:, :, j] = x[j] + dj
                continue
            Z = X[:, :, pa].copy()
            for q, k in enumerate(pa):
                c = cut[:, sp.eidx[(k, j)]][:, None]
                if block_ok is not None and (k, j) in sp.blk and (k, j) not in extra_cut: c = c & (np.asarray(block_ok) > 0.5)[None, :]
                Z[:, :, q] = np.where(c, x[k], Z[:, :, q])
            xF = np.broadcast_to(np.asarray(x, float)[pa], (1, self.S, len(pa)))
            lat = self._lat(j, x[j]) + self._g(j, Z) - self._g(j, xF)
            if shift is not None or not abduct:
                moved = (Z != xF).any(-1)
                if not abduct: lat = np.where(moved, self._g(j, Z) + self.noise_draw[j][None, :], lat)
                if shift is not None and j in shift: lat = lat + moved * np.asarray(shift[j])
            nat = self._obs(j, lat)
            if dj is None: X[:, :, j] = nat
            elif j in sp.hard_idx: X[:, :, j] = np.where(dj != 0, x[j] + dj, nat)
            else: X[:, :, j] = nat + dj
        return X

    # ---------- path-restricted states (Appendix A; node copies as in Shapley Flow) ----------
    def paths(self, A, B):
        """P(A, B): paths that start where the two actions differ (a changed variable or a cut edge) and reach the model."""
        sp = self.spec; A, B = np.asarray(A, float), np.asarray(B, float)
        cA, cB = self.cuts(A)[0], self.cuts(B)[0]; both = {e for e in sp.edges if cA[sp.eidx[e]] and cB[sp.eidx[e]]}
        def ok(j, s):                                           # the path does not use an edge cut in both actions
            seq = (j,) + s; return all((seq[i], seq[i + 1]) not in both for i in range(len(seq) - 1))
        P = []
        for i, j in enumerate(sp.act):
            if A[i] != B[i]: P += [("v", j, s) for s in sp.suffixes(j) if ok(j, s)]
        for e in sp.edges:
            if cA[sp.eidx[e]] != cB[sp.eidx[e]]: P += [("e", e, s) for s in sp.suffixes(e[1]) if ok(e[1], s)]
        return P

    def path_state(self, x, A, B, P, T):
        """Model inputs x^{A->B}(T): the paths in T follow B, all others follow A. Returns (S, d)."""
        sp = self.spec; A, B = np.asarray(A, float), np.asarray(B, float); x = np.asarray(x, float)
        cA, cB = self.cuts(A)[0], self.cuts(B)[0]; on = {P[i] for i in T}; apos = {j: i for i, j in enumerate(sp.act)}; memo = {}
        def val(j, s):
            if (j, s) in memo: return memo[(j, s)]
            dj = 0.0
            if j in apos:
                i = apos[j]; dj = B[i] if (A[i] != B[i] and ("v", j, s) in on) else A[i]
                if A[i] == B[i]: dj = A[i]
            pa = sp.pa[j]
            if not pa: out = np.full(self.S, x[j] + dj)
            else:
                Z = np.empty((1, self.S, len(pa)))
                for q, k in enumerate(pa):
                    e = (k, j); ia = sp.eidx[e]
                    c = cB[ia] if (cA[ia] != cB[ia] and ("e", e, s) in on) else cA[ia]
                    Z[0, :, q] = x[k] if c else val(k, (j,) + s)
                xF = np.broadcast_to(x[pa], (1, self.S, len(pa)))
                nat = self._obs(j, self._lat(j, x[j]) + self._g(j, Z) - self._g(j, xF))[0]
                out = np.where(dj != 0, x[j] + dj, nat) if (j in sp.hard_idx and j in apos) else nat + dj
            memo[(j, s)] = out; return out
        X = np.tile(x, (self.S, 1))
        for t in sp.inp: X[:, t] = val(t, ())
        return X


class Ensemble(SCM):
    """Estimated SCM: version 0 is the point estimate, versions 1..S-1 are bootstrap re-fits (Section 3).
    kind='linear' fits linear mechanisms; kind='rff' adds random Fourier features (a kernel approximation)."""
    def __init__(self, spec, X, S=20, kind="linear", D=60, ls=1.5, lam=1e-3, seed=0, link=None, link_inv=None):
        self.spec, self.S, self.kind = spec, S, kind; rng = np.random.default_rng(seed); n = len(X)
        self.link, self.link_inv = link or {}, link_inv or {}; self.phi, self.theta, self.resid, self.noise_draw = {}, {}, {}, {}
        boots = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(S - 1)]
        for j in spec.order:
            pa = spec.pa[j]
            if not pa: continue
            mu, sd = X[:, pa].mean(0), X[:, pa].std(0) + 1e-9
            if kind == "rff":
                Om = rng.standard_normal((D, len(pa))) / ls; b = rng.uniform(0, 2 * np.pi, D)
                phi = lambda Z, mu=mu, sd=sd, Om=Om, b=b: np.concatenate(
                    [np.sqrt(2 / D) * np.cos(((Z - mu) / sd) @ Om.T + b), (Z - mu) / sd, np.ones(Z.shape[:-1] + (1,))], -1)
            else:
                phi = lambda Z, mu=mu, sd=sd: np.concatenate([(Z - mu) / sd, np.ones(Z.shape[:-1] + (1,))], -1)
            Pm = phi(X[:, pa]); y = self._lat(j, X[:, j]); R = lam * n * np.eye(Pm.shape[1]) if kind == "rff" else 1e-9 * np.eye(Pm.shape[1])
            R[-1, -1] = 1e-9
            self.phi[j] = phi; r0 = y - Pm @ np.linalg.solve(Pm.T @ Pm + R, Pm.T @ y)
            self.resid[j] = r0; self.noise_draw[j] = rng.choice(r0, S)
            self.theta[j] = np.stack([np.linalg.solve(Pm[i].T @ Pm[i] + R, Pm[i].T @ y[i]) for i in boots])

    def _g(self, j, Z): return np.einsum("...sd,sd->...s", self.phi[j](Z), self.theta[j])

    def subset(self, idx):
        """A view with only the versions in idx (e.g. [0] for the point estimate)."""
        o = object.__new__(Ensemble); o.__dict__.update(self.__dict__); o.S = len(idx)
        o.theta = {j: th[idx] for j, th in self.theta.items()}; o.noise_draw = {j: v[idx] for j, v in self.noise_draw.items()}; return o


class TrueSCM(SCM):
    """A known SCM given by functions g_j(parent array (..., |pa_j|)) -> (...). Used as ground truth (S = 1)."""
    def __init__(self, spec, g, link=None, link_inv=None):
        self.spec, self.S, self.g = spec, 1, g; self.link, self.link_inv = link or {}, link_inv or {}

    def _g(self, j, Z): return self.g[j](Z)


class MultiGraph(SCM):
    """Versions of the SCM that do not share a graph (uncertainty over the DAG). Wraps one Ensemble per graph
    and concatenates their versions. All graphs must share variable names, actions and blockable edges."""
    def __init__(self, ensembles):
        self.members = ensembles; self.spec = ensembles[0].spec; self.S = sum(e.S for e in ensembles)

    def counterfactual(self, x, A, extra_cut=(), lifted=(), gamma=None, block_ok=None, **kw):
        outs, s0 = [], 0
        for e in self.members:
            ec = [c for c in extra_cut if c in e.spec.eidx]; bo = None if block_ok is None else np.asarray(block_ok)[s0:s0 + e.S]
            A2 = np.atleast_2d(A)[:, :e.spec.nA + e.spec.nB]
            outs.append(e.counterfactual(x, A2, ec, [l for l in lifted if l in e.spec.eidx], gamma, bo)); s0 += e.S
        return np.concatenate(outs, axis=1)


def linear_path_contributions(ens, x, a, beta, s=0):
    """Proposition 8(ii): in the linear case, the margin gained by an action is the sum over directed paths that
    avoid the cut set of pi_p = beta_t * delta_source * prod(weights). Returns {(source, suffix): pi_p} for version s."""
    sp = ens.spec; cut = ens.cuts(a)[0]; w = {}
    for j, th in ens.theta.items():
        pa = sp.pa[j]; Z0 = np.zeros((1, ens.S, len(pa))); base = ens._g(j, Z0)[0, s]
        for q, k in enumerate(pa):
            Z1 = Z0.copy(); Z1[0, :, q] = 1.0; w[(k, j)] = ens._g(j, Z1)[0, s] - base
    out = {}
    for i, j in enumerate(sp.act):
        if a[i] == 0: continue
        for suf in sp.suffixes(j):
            seq = (j,) + suf
            if any(cut[sp.eidx[(seq[q], seq[q + 1])]] for q in range(len(seq) - 1)): continue
            out[(j, suf)] = beta[seq[-1]] * a[i] * np.prod([w[(seq[q], seq[q + 1])] for q in range(len(seq) - 1)])
    return out
