"""Questions about an explanation (Definition 9), the path game and its credits (Definition 10, Lemma 11),
the fidelity F, and the trajectory contrasts of Section 6.

Fidelity: `fidelity` = sum_p v({p}) / (sum_p v({p}) + sum_{|T|>=2} mu(T)), i.e. the share of the total gain obtained by
adding the paths one by one (equation 11 as we read it from the draft; TO CHECK against the source, in particular
whether it has absolute values). `fidelity_abs` is the variant with absolute values, which stays in [0, 1]."""
import copy, itertools
from math import factorial
import numpy as np
from .scm import Spec
from .search import robust, certify, solve


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- questions q = (gamma, D, R)
def answer(scm, x, a, bb, eps, gamma=None, D=(), R=(), block_ok=None, surrogate=None, purpose="question", tol=0.1):
    """What happens to the plan if only a fraction gamma of each change is achieved (Q1), the edges in D fail (Q2)
    or the blocks in R do not hold (Q3). Exact mode evaluates the model under every version. With `surrogate`
    (GP mode) it predicts with the search surrogate and evaluates the model only where the prediction is unsure."""
    Xc = scm.counterfactual(x, a, extra_cut=D, lifted=R, gamma=gamma, block_ok=block_ok)[0]; n0 = bb.n
    if surrogate is None: M = bb.margin(Xc, purpose)
    else:
        mu, sd = surrogate.predict(Xc[:, scm.spec.inp]); M = mu.copy(); unsure = np.abs(mu) < 2 * sd + tol                    # near the threshold the model itself is asked
        if unsure.any(): M[unsure] = bb.margin(Xc[unsure], purpose + " (targeted)")
    r = robust(M[None], eps)[0][0]; k = int((M >= 0).sum()); lo, hi = wilson(k, len(M))
    return dict(robust_margin=float(r), valid=bool(r >= 0), p_success=k / len(M), ci=(lo, hi), queries=bb.n - n0, margins=M)


def success_probability(scm, x, a, bb, weights=None, **q):
    """p-hat: (weighted) fraction of versions in which the plan crosses the threshold, with a Wilson interval."""
    out = answer(scm, x, a, bb, 0.0, **q); ok = out["margins"] >= 0
    if weights is not None:
        w = np.asarray(weights) / np.sum(weights); p = float((w * ok).sum()); n_eff = 1 / (w ** 2).sum(); return p, wilson(p * n_eff, n_eff)
    return out["p_success"], out["ci"]


def smallest_fractions(scm, x, a, bb, eps, tol=0.02):
    """Q1: for each changed variable, the smallest fraction of its change that keeps robust validity (bisection)."""
    sp = scm.spec; out = {}
    for i in range(sp.nA):
        if a[i] == 0: continue
        lo, hi = 0.0, 1.0
        while hi - lo > tol:
            mid = (lo + hi) / 2; g = np.ones(sp.nA); g[i] = mid
            if answer(scm, x, a, bb, eps, gamma=g)["valid"]: hi = mid
            else: lo = mid
        out[sp.actionable[i]] = hi
    return out


def used_edges(scm, a):
    """Edges on directed paths from a changed variable to the model that the plan does not cut."""
    sp = scm.spec; cut = scm.cuts(a)[0]; E = set()
    for i, j in enumerate(sp.act):
        if a[i] == 0: continue
        for s in sp.suffixes(j):
            seq = (j,) + s; es = [(seq[q], seq[q + 1]) for q in range(len(seq) - 1)]
            if not any(cut[sp.eidx[e]] for e in es): E.update(es)
    return sorted(E)


def single_points_of_failure(scm, x, a, bb, eps):
    """Q2: the edges the plan uses whose failure alone makes it lose robust validity."""
    sp = scm.spec; out = {}
    for e in used_edges(scm, a):
        r = answer(scm, x, a, bb, eps, D=[e]); out[f"{sp.names[e[0]]}->{sp.names[e[1]]}"] = (not r["valid"], r["p_success"])
    return out


def block_reliance(scm, x, a, bb, eps):
    """Q3: for each block of the plan, whether the plan survives and in which fraction of versions, if it does not hold."""
    sp = scm.spec; out = {}
    for i, e in enumerate(sp.blk):
        if a[sp.nA + i] > 0.5:
            r = answer(scm, x, a, bb, eps, R=[e]); out[f"{sp.names[e[0]]}->{sp.names[e[1]]}"] = (r["valid"], r["p_success"])
    return out


def alternative(spec, scm, x, bb, eps, forbid_vars=(), forbid_blocks=False, D=(), R=(), gamma=None, **kw):
    """Q4: the cheapest robustly valid plan under a restriction (re-solves Algorithm 1)."""
    sp = copy.copy(spec); sp.lo, sp.hi = spec.lo.copy(), spec.hi.copy()
    for v in forbid_vars: i = spec.actionable.index(v); sp.lo[i] = sp.hi[i] = 0.0
    scm2 = copy.copy(scm); scm2.spec = sp
    return solve(sp, scm2, x, bb, eps, blocks=not forbid_blocks, kw=dict(extra_cut=D, lifted=R, gamma=gamma), **kw)


# ---------------------------------------------------------------- path game
def _label(sp, p):
    kind, src, suf = p; chain = lambda j, s: " -> ".join([sp.names[j]] + [sp.names[k] for k in s] + ["model"])
    return f"change {chain(src, suf)}" if kind == "v" else f"block {sp.names[src[0]]}->{sp.names[src[1]]} | {chain(src[1], suf)}"


def path_game(scm, x, A, B, bb, version=0, max_exact=11, n_perm=300, seed=0, purpose="game"):
    """Contrast A -> B: v(T) = m(x^{A->B}(T)) - m(x^{A->B}(empty)) and its Shapley credits, computed under one
    version (the point estimate by default). Exact for up to `max_exact` paths, permutation sampling above."""
    sp = scm.spec; P = scm.paths(A, B); n = len(P); val = {}
    def values(Ts):
        new = [T for T in Ts if T not in val]
        if new:
            X = np.array([scm.path_state(x, A, B, P, T)[version] for T in new]); m = bb.margin(X, purpose)
            for T, mm in zip(new, m): val[T] = mm
    values([frozenset()]); base = val[frozenset()]; out = dict(paths=P, labels=[_label(sp, p) for p in P], exact=n <= max_exact)
    if n == 0: return dict(out, credits=np.zeros(0), total=0.0, fidelity=1.0, singles=np.zeros(0))
    if n <= max_exact:
        subs = [frozenset(c) for k in range(n + 1) for c in itertools.combinations(range(n), k)]; values(subs)
        v = {T: val[T] - base for T in subs}; phi = np.zeros(n)
        for T in subs:
            for i in range(n):
                if i not in T: phi[i] += factorial(len(T)) * factorial(n - len(T) - 1) / factorial(n) * (v[T | {i}] - v[T])
        mob = {T: sum((-1) ** (len(T) - len(U)) * v[frozenset(U)] for k in range(len(T) + 1) for U in itertools.combinations(sorted(T), k)) for T in subs if len(T) >= 2}
        singles = np.array([v[frozenset([i])] for i in range(n)]); inter = sum(abs(m) for m in mob.values())
        Fa = np.abs(singles).sum() / (np.abs(singles).sum() + inter) if np.abs(singles).sum() + inter > 0 else 1.0
        tot = v[frozenset(range(n))]; F = singles.sum() / tot if abs(tot) > 1e-12 else 1.0      # eq. (11) as read from the draft: sum of singles over singles + interactions = total
        return dict(out, credits=phi, total=tot, fidelity=F, fidelity_abs=Fa, singles=singles, mobius=mob)
    rng = np.random.default_rng(seed); phi = np.zeros(n); full = frozenset(range(n)); values([full] + [frozenset([i]) for i in range(n)])
    for _ in range(n_perm):
        perm = rng.permutation(n); Ts = [frozenset(perm[:k]) for k in range(n + 1)]; values(Ts)
        for k in range(n): phi[perm[k]] += (val[Ts[k + 1]] - val[Ts[k]]) / n_perm
    return dict(out, credits=phi, total=val[full] - base, fidelity=None, singles=np.array([val[frozenset([i])] - base for i in range(n)]))


def aggregate(scm, game):
    """Credits per edge (sum over the paths that use it) and per changed variable (sum over the paths that start at it)."""
    sp = scm.spec; by_edge, by_var = {}, {}
    for (kind, src, suf), c in zip(game["paths"], game["credits"]):
        seq = ((src,) if kind == "v" else (src[0], src[1])) + suf
        for q in range(len(seq) - 1):
            k = f"{sp.names[seq[q]]}->{sp.names[seq[q + 1]]}"; by_edge[k] = by_edge.get(k, 0.0) + c
        k = sp.names[src] if kind == "v" else f"block {sp.names[src[0]]}->{sp.names[src[1]]}"; by_var[k] = by_var.get(k, 0.0) + c
    return by_edge, by_var


def explanation_game(scm, x, a, bb, **kw):
    """The contrast 'do nothing -> the plan': the credits split the gain of the plan over the factual state."""
    return path_game(scm, x, np.zeros_like(a), a, bb, **kw)


def action_level_shapley(scm, x, a, bb, version=0):
    """STAND-IN baseline (not Shapley Flow, DAG-SHAP, causal Shapley or CF-SHAP): Shapley values of the action's
    components (each change, each block) as players. It sees which component matters, not along which path."""
    idx = [i for i in range(len(a)) if a[i] != 0]; n = len(idx)
    subs = [frozenset(c) for k in range(n + 1) for c in itertools.combinations(range(n), k)]; A = np.zeros((len(subs), len(a)))
    for r, T in enumerate(subs):
        for t in T: A[r, idx[t]] = a[idx[t]]
    m = bb.margin(scm.counterfactual(x, A)[:, version], "game (action-level)"); v = dict(zip(subs, m - m[0])); phi = np.zeros(n)
    for T in subs:
        for i in range(n):
            if i not in T: phi[i] += factorial(len(T)) * factorial(n - len(T) - 1) / factorial(n) * (v[T | {i}] - v[T])
    sp = scm.spec; names = [sp.actionable[i] if i < sp.nA else "block %s->%s" % tuple(sp.names[k] for k in sp.blk[i - sp.nA]) for i in idx]
    return dict(zip(names, phi))


# ---------------------------------------------------------------- Section 6: the search trajectory as an explanation
def all_edges_view(scm):
    """A copy of the SCM in which every edge can be cut, used for the contrast 'what the graph adds'."""
    sp = scm.spec; sp2 = copy.copy(sp); sp2.blk = list(sp.edges); sp2.nB = len(sp2.blk); s2 = copy.copy(scm); s2.spec = sp2
    def lift(a):
        b = np.zeros(sp.nA + sp2.nB); b[:sp.nA] = a[:sp.nA]
        for i, e in enumerate(sp.blk): b[sp.nA + sp2.blk.index(e)] = a[sp.nA + i]
        return b
    return s2, lift


def trajectory_contrasts(scm, x, res, bb, max_jumps=4, **kw):
    """Incumbent jumps, the cheapest near-miss against the final plan, and what causal propagation adds."""
    sp = scm.spec; out = dict(jumps=[], near_miss=None, graph=None); a = res.action
    if a is None: return out
    inc = res.incumbents[-(max_jumps + 1):]
    for (A, cA), (B, cB) in zip(inc[:-1], inc[1:]):
        g = path_game(scm, x, A, B, bb, **kw); out["jumps"].append(dict(frm=sp.describe(A), to=sp.describe(B), saved=cA - cB, game=g))
    nm = [n for n in res.near_misses if n[1] < res.cost]
    if nm:
        n = min(nm, key=lambda t: t[1]); g = path_game(scm, x, n[0], a, bb, **kw)
        out["near_miss"] = dict(action=sp.describe(n[0]), cost=n[1], robust_margin=n[2], game=g)
    s2, lift = all_edges_view(scm); b = lift(a); A = b.copy(); A[sp.nA:] = 1.0
    out["graph"] = dict(game=path_game(s2, x, A, b, bb, **kw))
    return out


def ficha(case, scm, x, a, bb, eps, lang="es"):
    """Experiment D: the one-page sheet shown to clinicians. Text only; every number is recomputed from the model."""
    sp = scm.spec; r = answer(scm, x, a, bb, eps); g = explanation_game(scm, x, a, bb); be, bv = aggregate(scm, g)
    q1, q2, q3 = smallest_fractions(scm, x, a, bb, eps), single_points_of_failure(scm, x, a, bb, eps), block_reliance(scm, x, a, bb, eps)
    L = [f"FICHA — caso {case.name}", "", "Paciente: " + ", ".join(f"{n} = {v:.3g}" for n, v in zip(sp.names, x)),
         f"Plan recomendado: {sp.describe(a)}  (costo {sp.cost(a)[0]:.3f})",
         f"Funciona en {100 * r['p_success']:.0f} % de las {scm.S} versiones del modelo causal (intervalo de 95 %: {100 * r['ci'][0]:.0f} a {100 * r['ci'][1]:.0f} %). Riesgo tolerado: {100 * eps:.0f} %.",
         "", "¿Por qué esta acción? Aporte de cada camino a la ganancia del plan:"]
    L += [f"  {c:+.3f}  {lab}" for lab, c in sorted(zip(g["labels"], g["credits"]), key=lambda t: -abs(t[1]))]
    L += [f"  Sumados uno por uno, los caminos dan el {100 * g['fidelity']:.0f} % de la ganancia; el resto son interacciones entre caminos (fidelidad)." if g["fidelity"] is not None else "",
          "", "Si el cambio se logra solo en parte: fracción mínima que mantiene el plan válido"]
    L += [f"  {k}: {100 * v:.0f} %" for k, v in q1.items()] + ["", "Si una relación causal no se cumple:"]
    L += [f"  {k}: " + ("el plan DEJA de ser válido" if f else "el plan sigue siendo válido") + f" (funciona en {100 * p:.0f} % de las versiones)" for k, (f, p) in q2.items()]
    L += ["", "Si el bloqueo no se sostiene:"] + [f"  {k}: " + ("el plan sigue siendo válido" if ok else "el plan DEJA de ser válido") + f" (funciona en {100 * p:.0f} %)" for k, (ok, p) in q3.items()]
    return "\n".join(L)
