"""Builds results/SUMMARY.md from the CSV files written by experiments.run."""
import os, sys, glob
import numpy as np, pandas as pd
OUT = os.environ.get("WTA_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))

def wilson(k, n, z=1.96):
    if n == 0: return "–"
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * p:.0f} % ({100 * max(0, c - h):.0f}–{100 * min(1, c + h):.0f})"

def rate(s): s = s.dropna(); return f"{int(s.sum())}/{len(s)} = " + wilson(s.sum(), len(s)) if len(s) else "–"
def load(e):
    fs = sorted(glob.glob(os.path.join(OUT, f"{e}_*.csv"))); return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
def md(df): return df.to_markdown(index=False) + "\n"

L = ["# Results summary", "", "Simulation results produced by `experiments/run.py`. Rates show count, percentage and 95 % Wilson interval.",
     "Rows with few cases are NOT conclusive. For real data (german, adult, sachs) 'valid in truth' uses a held-out re-fit of the SCM as a proxy.", ""]
d = load("C")
if d is not None:
    L += ["## C · Search", ""]
    for case, g in d.groupby("case"):
        n = g.groupby(["seed", "ind"]).ngroups; L += [f"### {case}  ({g.seed.nunique()} data seeds, {n} individuals; truth = {g.truth.iloc[0]})", ""]
        t = g.groupby("method", sort=False).apply(lambda q: pd.Series({"found": f"{int(q.found.sum())}/{len(q)}", "cost (mean)": round(q.cost.mean(), 3),
            "gap to reference, % (median / max)": f"{q.gap_pct.median():.1f} / {q.gap_pct.max():.1f}" if q.gap_pct.notna().any() else "–",
            "robustly valid": rate(q.robust_valid), "valid in truth": rate(q.valid_truth), "queries (median)": int(q.queries.median())})).reset_index()
        L += [md(t)]
d = load("A")
if d is not None:
    L += ["## A · Fidelity of the answers", "", "Each row of the first table is a kind of question. 'Agrees with truth' = the answer (plan still valid: yes/no) matches what happens under the true SCM.", ""]
    q = d[d.question != "Q5"]
    t = q.groupby(["case", "question"]).apply(lambda g: pd.Series({"questions": len(g), "exact mode agrees with truth": rate((g.p_exact >= 0.5) == (g.truth_valid == 1)),
        "GP mode agrees with exact mode": rate((g.p_gp >= 0.5) == (g.p_exact >= 0.5)), "queries exact (mean)": round(g.queries_exact.mean(), 1), "queries GP mode (mean)": round(g.queries_gp.mean(), 1)})).reset_index()
    L += [md(t)]; q5 = d[d.question == "Q5"]
    t = q5.groupby("case").apply(lambda g: pd.Series({"plans": len(g), "paths per plan (mean)": round(g.n_paths.mean(), 1), "action components (mean)": round(g.n_components.mean(), 1),
        "fidelity F (median)": round(g.fidelity_F.median(), 3), "fidelity, absolute-value variant (median)": round(g.fidelity_abs.median(), 3) if "fidelity_abs" in g else "–", "credit error vs true SCM (mean, relative)": round(g.credit_rel_error.mean(), 3) if "credit_rel_error" in g else "–",
        "same top path as true SCM": rate(g.same_top_path) if "same_top_path" in g else "–"})).reset_index()
    L += ["Credits of the explanation game (Q5):", "", md(t)]
d = load("B")
if d is not None:
    L += ["## B · When blocking pays off", "", "Reference optimum of the robust problem with and without edge blocking.", ""]
    t = d.groupby("case").apply(lambda g: pd.Series({"individuals": len(g), "optimum uses a block": rate(g.uses_block), "a plan exists only with blocking": rate(g.only_with_block),
        "saving when both exist, % (mean / max)": f"{g.saving_pct.mean():.1f} / {g.saving_pct.max():.1f}" if g.saving_pct.notna().any() else "–",
        "valid in truth, with": rate(g.valid_truth_with), "valid in truth, without": rate(g.valid_truth_without)})).reset_index()
    L += [md(t)]
d = load("E")
if d is not None:
    L += ["## E · Stability across search seeds", "", "Same individual and same SCM estimates, different search seeds.", ""]
    t = d.groupby("case").apply(lambda g: pd.Series({"individuals": len(g), "cost variation across seeds, % (mean)": round(g.cost_cv_pct.mean(), 2),
        "credit instability (0 = identical)": round(g.credit_instability.mean(), 3), "same top credit in all runs": rate(g.same_top_credit),
        "telescoping error (max)": f"{g.telescoping_error.max():.1e}", "step-by-step vs global edge credits (relative gap, mean)": round(g.step_vs_global_edge_gap.mean(), 3)})).reset_index()
    L += [md(t)]
d = load("X")
if d is not None:
    L += ["## X · Three layers of error (extension)", "", "p = share of versions in which the plan works; 'total' adds a 3 % chance that a block does not hold (expert input, illustrative).", ""]
    t = d.groupby(["case", "graph_knowledge", "method"], sort=False).apply(lambda g: pd.Series({"plan found": f"{int(g.found.sum())}/{len(g)}", "cost (mean)": round(g.cost.mean(), 3), "valid in truth": rate(g.valid_truth),
        "p mechanisms": round(g.p_mech.mean(), 3), "p + graph": round(g.p_graph.mean(), 3), "p total": round(g.p_total.mean(), 3), "passes localized conformal check": rate(g.conformal_pass),
        "weight of the true graph": round(g.weight_true_graph.mean(), 2), "queries (median)": int(g.queries.median())})).reset_index()
    L += [md(t)]
fs = sorted(glob.glob(os.path.join(OUT, "D_ficha_*.txt")))
if fs: L += ["## D · Sheet for clinicians (example)", "", "```", open(fs[0]).read(), "```", ""]
cal = os.path.join(OUT, "calibration.txt")
if os.path.exists(cal):
    L += ["## Calibration of the robust criterion (experiments/calibration.py)", "", "One individual per data seed, 40 data seeds; the plan is the reference optimum of the robust problem.", "", "```", open(cal).read().strip(), "```", ""]
open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(L)); print("written", os.path.join(OUT, "SUMMARY.md"))
