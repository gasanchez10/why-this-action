"""Experiments of Section 7 (A, B, C, D, E) and of the three-layer extension (X).

  python -m experiments.run C --cases perioperative loan7 --seeds 3 --n 3
  python -m experiments.run all --mode quick        # every experiment, small sizes
  python -m experiments.run all --mode full         # sizes for the paper (hours)

Each experiment appends one row per (case, data seed, individual, method) to results/<experiment>_<case>.csv.
A data seed changes the observational sample, the black box and the individuals. Then: python -m experiments.summarize
"""
import argparse, os, sys, time, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wta.data import CASES, BlackBox
from wta.scm import Ensemble
from wta.search import solve, reference, random_search, action_space_bo, grid_search, robust, certify
from wta import explain as ex, extensions as xt

OUT = os.environ.get("WTA_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")); os.makedirs(OUT, exist_ok=True)
EPS = 0.10


def save(exp, case, rows):
    if not rows: return
    path = os.path.join(OUT, f"{exp}_{case}.csv"); keys = list(dict.fromkeys(k for r in rows for k in r)); new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        if new: w.writeheader()
        w.writerows(rows)


def setup(name, seed, S=30, n=2):
    case = CASES[name](seed=seed); kw = dict(link=getattr(case, "link", None), link_inv=getattr(case, "link_inv", None))
    ens = Ensemble(case.spec, case.X, S=S, kind=case.kind, seed=seed, **kw); xs = case.individuals(n, np.random.default_rng(seed + 77))
    if case.true is not None: truth, tname = case.true, "true SCM"
    else: truth, tname = Ensemble(case.spec, case.X_holdout, S=1, kind=case.kind, seed=seed + 1, **kw), "held-out re-fit (proxy)"
    fm = lambda X, p=case.proba: BlackBox(p).margin(X)                       # free evaluations, for measuring only
    return case, ens, xs, truth, tname, fm


def measure(case, ens, truth, fm, x, a, eps=EPS, **q):
    if a is None: return dict(found=0)
    return dict(found=1, cost=float(case.spec.cost(a)[0]), robust_valid=int(robust(fm(ens.counterfactual(x, a, **q)[0])[None], eps)[0][0] >= 0),
                valid_truth=int(fm(truth.counterfactual(x, a, **q)[0])[0] >= 0), plan=case.spec.describe(a))


# ---------------------------------------------------------------- C: search
def exp_C(name, seeds, n, budget=50, budgets=(20, 40, 80, 160), with_gpscm=True, **_):
    for seed in seeds:
        case, ens, xs, truth, tname, fm = setup(name, seed, n=n); sp = case.spec; rows = []
        point = ens.subset([0]); gps = None
        if with_gpscm and sp.nA <= 4 and case.true is not None and len(case.X) <= 400: gps = xt.GPSCM(sp, case.X, seed)
        for ii, x in enumerate(xs):
            ref, rc = reference(sp, ens, x, fm, EPS, seed=seed); base = dict(case=name, seed=seed, ind=ii, truth=tname, ref_cost=rc)
            def add(method, a, queries, sec, note=""):
                r = dict(base, method=method, queries=queries, seconds=round(sec, 1), note=note, **measure(case, ens, truth, fm, x, a))
                if r["found"] and np.isfinite(rc): r["gap_pct"] = 100 * (r["cost"] / rc - 1)
                rows.append(r)
            add("reference optimum (robust problem, free queries)", ref, 0, 0)
            if case.true is not None:
                t = time.time(); o, _ = reference(sp, case.true, x, fm, 0.0, seed=seed); add("oracle (true SCM, free queries)", o, 0, time.time() - t)
            runs = [("Algorithm 1", dict()), ("Algorithm 1 + polish (extension)", dict(polish=True)),
                    ("ablation: no blocking", dict(blocks=False)), ("ablation: no certification", dict(do_certify=False)),
                    ("ablation: no causal propagation (non-causal stand-in)", dict(propagate=False)),
                    ("baseline: no abduction (stand-in for causal BO without abduction)", dict(kw=dict(abduct=False)))]
            for method, kw in runs:
                t = time.time(); bb = case.blackbox(); r = solve(sp, ens, x, bb, EPS, budget=budget, scale_X=case.X, seed=seed, **kw); add(method, r.action, bb.n, time.time() - t)
            t = time.time(); bb = case.blackbox(); r = solve(sp, point, x, bb, 0.0, budget=budget, scale_X=case.X, seed=seed); add("point estimate (one version)", r.action, bb.n, time.time() - t)
            t = time.time(); bb = case.blackbox(); r = action_space_bo(sp, ens, x, bb, EPS, n_eval=30, seed=seed); add("ablation: surrogate on actions", r.action, bb.n, time.time() - t)
            t = time.time(); bb = case.blackbox(); r = random_search(sp, ens, x, bb, EPS, budget=2000, seed=seed); add("baseline: random search with certification (2000 queries)", r.action, bb.n, time.time() - t)
            t = time.time(); bb = case.blackbox(); r = grid_search(sp, ens, x, bb, EPS, max_actions=3000); add("baseline: grid with our criterion", r.action, bb.n, time.time() - t)
            if gps is not None:
                t = time.time(); bb = case.blackbox(); gps.prepare(x)
                r = grid_search(sp, None, x, bb, EPS, max_actions=1500, blocks=False, valid_fn=gps.criterion(bb, gamma=1.0, M=30, seed=seed))
                add("baseline: Karimi et al. 2020 GP-SCM + grid (our re-implementation, gamma=1)", r.action, bb.n, time.time() - t)
            for b in budgets:
                t = time.time(); bb = case.blackbox(); r = solve(sp, ens, x, bb, EPS, budget=b, scale_X=case.X, seed=seed); add(f"Algorithm 1, budget {b}", r.action, bb.n, time.time() - t)
        save("C", name, rows); print("C", name, "seed", seed, "done", flush=True)


# ---------------------------------------------------------------- A: fidelity of the answers
def exp_A(name, seeds, n, budget=50, **_):
    for seed in seeds:
        case, ens, xs, truth, tname, fm = setup(name, seed, n=n); sp = case.spec; rows = []
        for ii, x in enumerate(xs):
            bb = case.blackbox(); res = solve(sp, ens, x, bb, EPS, budget=budget, scale_X=case.X, seed=seed); a = res.action
            if a is None: continue
            qs = [("Q0 plan as is", dict())] + [(f"Q1 only {int(100 * g)} % of each change", dict(gamma=g)) for g in (0.5, 0.75, 0.9)]
            qs += [(f"Q2 edge {sp.names[e[0]]}->{sp.names[e[1]]} fails", dict(D=[e])) for e in ex.used_edges(ens, a)]
            qs += [(f"Q3 block {sp.names[e[0]]}->{sp.names[e[1]]} does not hold", dict(R=[e])) for i, e in enumerate(sp.blk) if a[sp.nA + i] > 0.5]
            for qn, q in qs:
                e1 = ex.answer(ens, x, a, case.blackbox(), EPS, **q); e2 = ex.answer(ens, x, a, case.blackbox(), EPS, surrogate=res.surrogate, **q)
                tq = dict(gamma=q.get("gamma"), extra_cut=q.get("D", ()), lifted=q.get("R", ())); t_ok = int(fm(truth.counterfactual(x, a, **tq)[0])[0] >= 0)
                rows.append(dict(case=name, seed=seed, ind=ii, truth=tname, question=qn[:2], detail=qn, truth_valid=t_ok, p_exact=e1["p_success"], valid_exact=int(e1["valid"]),
                                 queries_exact=e1["queries"], p_gp=e2["p_success"], valid_gp=int(e2["valid"]), queries_gp=e2["queries"]))
            g_est = ex.explanation_game(ens, x, a, case.blackbox()); bbg = case.blackbox(); g_true = ex.explanation_game(truth, x, a, bbg) if case.true is not None else None
            al = ex.action_level_shapley(ens, x, a, case.blackbox())
            r = dict(case=name, seed=seed, ind=ii, truth=tname, question="Q5", detail="credits of the explanation game", n_paths=len(g_est["paths"]), n_components=len(al),
                     fidelity_F=g_est["fidelity"], fidelity_abs=g_est.get("fidelity_abs"), exact_game=int(g_est["exact"]))
            if g_true is not None and len(g_true["credits"]) == len(g_est["credits"]):
                r["credit_rel_error"] = float(np.abs(g_est["credits"] - g_true["credits"]).sum() / (np.abs(g_true["credits"]).sum() + 1e-12))
                r["same_top_path"] = int(np.argmax(np.abs(g_est["credits"])) == np.argmax(np.abs(g_true["credits"])))
            rows.append(r)
        save("A", name, rows); print("A", name, "seed", seed, "done", flush=True)


# ---------------------------------------------------------------- B: when blocking pays off
def exp_B(name, seeds, n, **_):
    for seed in seeds:
        case, ens, xs, truth, tname, fm = setup(name, seed, n=n); sp = case.spec; rows = []
        if sp.nB == 0: return
        for ii, x in enumerate(xs):
            a1, c1 = reference(sp, ens, x, fm, EPS, seed=seed); a0, c0 = reference(sp, ens, x, fm, EPS, seed=seed, blocks=False)
            r = dict(case=name, seed=seed, ind=ii, truth=tname, cost_with_blocks=c1, cost_without_blocks=c0, uses_block=int(a1 is not None and a1[sp.nA:].sum() > 0),
                     only_with_block=int(a1 is not None and a0 is None))
            if a1 is not None and a0 is not None: r["saving_pct"] = 100 * (1 - c1 / c0)
            if a1 is not None: r["valid_truth_with"] = measure(case, ens, truth, fm, x, a1)["valid_truth"]
            if a0 is not None: r["valid_truth_without"] = measure(case, ens, truth, fm, x, a0)["valid_truth"]
            rows.append(r)
        save("B", name, rows); print("B", name, "seed", seed, "done", flush=True)


# ---------------------------------------------------------------- E: stability of credits and trajectory contrasts
def exp_E(name, seeds, n, budget=50, n_search=4, **_):
    for seed in seeds:
        case, ens, xs, truth, tname, fm = setup(name, seed, n=n); sp = case.spec; rows = []
        for ii, x in enumerate(xs):
            costs, creds, tele, stepglob = [], [], [], []
            for s in range(n_search):
                bb = case.blackbox(); res = solve(sp, ens, x, bb, EPS, budget=budget, scale_X=case.X, seed=1000 * seed + s)
                if res.action is None: continue
                costs.append(res.cost); g = ex.explanation_game(ens, x, res.action, case.blackbox()); creds.append(ex.aggregate(ens, g)[1])
                inc = [i[0] for i in res.incumbents][-4:]
                if len(inc) >= 3:
                    steps = [ex.path_game(ens, x, A, B, case.blackbox()) for A, B in zip(inc[:-1], inc[1:])]; glob = ex.path_game(ens, x, inc[0], inc[-1], case.blackbox())
                    tele.append(abs(sum(g_["total"] for g_ in steps) - glob["total"]))
                    se = {}
                    for g_ in steps:
                        for k, v in ex.aggregate(ens, g_)[0].items(): se[k] = se.get(k, 0.0) + v
                    ge = ex.aggregate(ens, glob)[0]; keys = set(se) | set(ge)
                    stepglob.append(sum(abs(se.get(k, 0) - ge.get(k, 0)) for k in keys) / (sum(abs(ge.get(k, 0)) for k in keys) + 1e-12))
            if len(costs) < 2: continue
            keys = sorted(set(k for c in creds for k in c)); Cm = np.array([[c.get(k, 0.0) for k in keys] for c in creds])
            rows.append(dict(case=name, seed=seed, ind=ii, n_runs=len(costs), cost_mean=np.mean(costs), cost_cv_pct=100 * np.std(costs) / np.mean(costs),
                             credit_instability=float(Cm.std(0).sum() / (np.abs(Cm.mean(0)).sum() + 1e-12)), same_top_credit=int(len(set(np.argmax(np.abs(Cm), 1))) == 1),
                             telescoping_error=float(np.max(tele)) if tele else np.nan, step_vs_global_edge_gap=float(np.mean(stepglob)) if stepglob else np.nan))
        save("E", name, rows); print("E", name, "seed", seed, "done", flush=True)


# ---------------------------------------------------------------- D: the sheet for clinicians
def exp_D(name, seeds, n, budget=50, **_):
    for seed in seeds:
        case, ens, xs, truth, tname, fm = setup(name, seed, n=n)
        for ii, x in enumerate(xs):
            res = solve(case.spec, ens, x, case.blackbox(), EPS, budget=budget, scale_X=case.X, seed=seed)
            if res.action is None: continue
            with open(os.path.join(OUT, f"D_ficha_{name}_s{seed}_i{ii}.txt"), "w") as f: f.write(ex.ficha(case, ens, x, res.action, case.blackbox(), EPS))
    print("D", name, "done", flush=True)


# ---------------------------------------------------------------- X: three layers of error (extension)
def exp_X(name, seeds, n, budget=50, **_):
    for seed in seeds:
        case, ens, xs, truth, tname, fm = setup(name, seed, n=n); sp = case.spec; rows = []
        half = len(case.X) // 2; kw = dict(link=getattr(case, "link", None), link_inv=getattr(case, "link_inv", None))
        levels = [("expert gives the causal order", dict(order=sp.order)), ("no prior knowledge (approximate search)", dict(order=None))]
        ens_fit = Ensemble(sp, case.X[:half], S=1, kind=case.kind, seed=seed, **kw)             # for conformal: fit on one half, calibrate on the other
        cbox = xt.ConformalBoxSCM(ens_fit, case.X[half:], alpha=0.10, seed=seed)
        true_g = tuple(sorted((j, tuple(sorted(p))) for j, p in sp.pa.items() if p))
        for lname, lkw in levels:
            G = xt.bootstrap_graphs(case.X, B=60 if lkw["order"] is not None else 30, seed=seed, **lkw); mg = xt.graph_ensemble(sp, case.X, G, S=40, kind=case.kind, seed=seed, **kw)
            w_true = sum(w for g, w in G if tuple(sorted((j, tuple(sorted(p))) for j, p in g.items() if p)) == true_g)
            for ii, x in enumerate(xs):
                base = dict(case=name, seed=seed, ind=ii, truth=tname, graph_knowledge=lname, n_graphs=len(G), weight_true_graph=w_true)
                for method, scm in [("mechanisms only (graph assumed)", ens), ("conformal box on the mechanisms (graph assumed)", cbox), ("graph + mechanisms", mg)]:
                    if not method.startswith("graph") and lname != levels[0][0]: continue
                    eps = 0.0 if scm is cbox else EPS
                    bb = case.blackbox(); res = solve(sp, scm, x, bb, eps, budget=budget, scale_X=case.X, seed=seed); a = res.action
                    r = dict(base, method=method, queries=bb.n, found=int(a is not None))
                    if a is not None:
                        r.update(cost=res.cost, valid_truth=int(fm(truth.counterfactual(x, a)[0])[0] >= 0), plan=sp.describe(a))
                        lay = xt.layered_success(ens, mg, x, a, fm, block_fail=0.03, seed=seed)
                        r.update(p_mech=lay["mechanisms"][0], p_graph=lay["mechanisms + graph"][0], p_total=lay["+ recourse assumptions"][0])
                        cp = xt.conformal_plan(ens_fit, case.X[half:], x, a, fm, alpha=0.10, seed=seed)
                        r.update(conformal_L=cp["L"], conformal_worst_margin=cp["worst_margin"], conformal_pass=int(cp["worst_margin"] >= 0), conformal_min_n_eff=min(cp["n_eff"].values()) if cp["n_eff"] else np.nan)
                    rows.append(r)
        save("X", name, rows); print("X", name, "seed", seed, "done", flush=True)


EXPS = dict(A=exp_A, B=exp_B, C=exp_C, D=exp_D, E=exp_E, X=exp_X)
PLAN = {  # experiment -> (cases quick, cases full)
    "A": (["perioperative", "loan7", "dag10"], ["perioperative", "synth3-linear", "synth3-anm", "loan7", "dag10", "dag20"]),
    "B": (["perioperative", "dag10", "loan7", "german"], ["perioperative", "synth3-linear", "synth3-anm", "dag10", "dag20", "dag30", "loan7", "german", "adult", "sachs"]),
    "C": (["perioperative", "synth3-linear", "synth3-anm", "loan7", "dag10", "german"], ["perioperative", "synth3-linear", "synth3-anm", "loan7", "dag10", "dag20", "dag30", "german", "adult", "sachs"]),
    "D": (["perioperative"], ["perioperative"]),
    "E": (["perioperative", "loan7"], ["perioperative", "loan7", "dag10", "dag20"]),
    "X": (["perioperative", "dag10"], ["perioperative", "dag10", "dag20", "loan7"])}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--cases", nargs="*"); ap.add_argument("--mode", default="quick")
    ap.add_argument("--seeds", type=int); ap.add_argument("--n", type=int); ap.add_argument("--seed0", type=int, default=0); a = ap.parse_args()
    seeds = a.seeds or (2 if a.mode == "quick" else 10); n = a.n or (2 if a.mode == "quick" else 5)
    for e in (list(EXPS) if a.exp == "all" else [a.exp]):
        for c in (a.cases or PLAN[e][0 if a.mode == "quick" else 1]):
            t = time.time(); EXPS[e](c, range(a.seed0, a.seed0 + seeds), n); print(f"== {e} {c}: {time.time() - t:.0f} s", flush=True)
