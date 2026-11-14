"""Numerical checks of the paper's statements. Run: python -m pytest tests -q   (or: python tests/test_propositions.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wta.data import perioperative, loan, random_dag, BlackBox
from wta.scm import Ensemble, linear_path_contributions
from wta.search import robust, solve, certify, sample_actions
from wta.explain import path_game, explanation_game, aggregate

RNG = np.random.default_rng(0)
def _setup(case, S=8):
    x = case.individuals(1, np.random.default_rng(1))[0]
    return x, Ensemble(case.spec, case.X, S=S, kind=case.kind, link=getattr(case, "link", None), link_inv=getattr(case, "link_inv", None))

def test_robust_definition():
    M = np.array([[-1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    assert robust(M, 0.0)[0][0] == -1.0 and robust(M, 0.10)[0][0] == 0.2 and robust(M, 0.25)[0][0] == 0.3

def test_prop7_consistency():
    """Zero change with any set of cut edges returns the factual state, for estimated and true models."""
    for case in (perioperative(), loan(), random_dag(10, nonlinear=True)):
        x, ens = _setup(case); sp = case.spec
        for scm in (ens, case.true):
            for blocks in ([0] * sp.nB, [1] * sp.nB):
                X = scm.counterfactual(x, np.r_[np.zeros(sp.nA), blocks], extra_cut=sp.edges[:2])
                assert np.abs(X - x).max() < 1e-8, case.name

def test_operator_closed_form():
    """Perioperative, true linear model: lowering HbA1c by 1 raises eGFR by 6 and Hb by 0.36; with the block, SBP stays."""
    case = perioperative(); x = case.pool[0]
    X = case.true.counterfactual(x, [[-1.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])[:, 0] - x
    assert np.allclose(X[0], [0, -1, 6, 0.36, 6 * 0.36]) and np.allclose(X[1], [0, -1, 6, 0.36, 0]) and np.allclose(X[2], [0, 0, 0, 1, 0])

def _linear_margin(case):
    bb = BlackBox(case.proba); x0 = case.pool[0]; m0 = bb.margin(x0)[0]
    return np.array([(bb.margin(x0 + np.eye(case.spec.d)[j])[0] - m0) for j in range(case.spec.d)])

def test_prop8_linear_paths():
    """Linear mechanisms and linear margin: the gain of an action is the sum of the contributions of the paths it
    does not cut, and cutting an edge removes exactly the paths through it."""
    for case in (perioperative(model="logistic"), random_dag(10, model="logistic")):
        x, ens = _setup(case); sp = case.spec; beta = _linear_margin(case); bb = BlackBox(case.proba)
        for a in sample_actions(sp, 6, RNG):
            gain = bb.margin(ens.counterfactual(x, a)[0, 0])[0] - bb.margin(x)[0]
            pi = linear_path_contributions(ens, x, a, beta)
            assert abs(gain - sum(pi.values())) < 1e-6 * (1 + abs(gain)), case.name
            a0 = a.copy(); a0[sp.nA:] = 0; pi0 = linear_path_contributions(ens, x, a0, beta); cut = ens.cuts(a)[0]
            through = sum(v for (j, s), v in pi0.items() if any(cut[sp.eidx[e]] for e in zip((j,) + s[:-1], s)))
            assert abs(sum(pi0.values()) - sum(pi.values()) - through) < 1e-6 * (1 + abs(through))

def test_path_state_endpoints():
    """With no path switched on the state is that of A; with all of them, that of B."""
    for case in (perioperative(), loan(), random_dag(10, nonlinear=True)):
        x, ens = _setup(case); sp = case.spec
        for _ in range(4):
            A, B = sample_actions(sp, 2, RNG); P = ens.paths(A, B)
            assert np.abs(ens.path_state(x, A, B, P, [])[:, sp.inp] - ens.counterfactual(x, A)[0][:, sp.inp]).max() < 1e-8, case.name
            assert np.abs(ens.path_state(x, A, B, P, range(len(P)))[:, sp.inp] - ens.counterfactual(x, B)[0][:, sp.inp]).max() < 1e-8, case.name

def test_lemma11_credits_add_up():
    for case in (perioperative(), loan()):
        x, ens = _setup(case); sp = case.spec; bb = BlackBox(case.proba)
        for _ in range(3):
            A, B = sample_actions(sp, 2, RNG); g = path_game(ens, x, A, B, bb)
            diff = bb.margin(ens.counterfactual(x, B)[0, 0])[0] - bb.margin(ens.counterfactual(x, A)[0, 0])[0]
            assert abs(g["credits"].sum() - diff) < 1e-8 and abs(g["total"] - diff) < 1e-8

def test_linear_credits_are_path_contributions():
    case = perioperative(model="logistic"); x, ens = _setup(case); beta = _linear_margin(case); bb = BlackBox(case.proba)
    a = np.array([-0.8, 1.5, 0.0]); g = explanation_game(ens, x, a, bb); pi = linear_path_contributions(ens, x, a, beta)
    for (kind, j, s), c in zip(g["paths"], g["credits"]): assert abs(c - pi[(j, s)]) < 1e-6
    assert g["fidelity"] > 1 - 1e-6

def test_prop13_telescoping():
    case = loan(); x, ens = _setup(case); bb = BlackBox(case.proba); A, B, C = sample_actions(case.spec, 3, RNG)
    t = [path_game(ens, x, P, Q, bb)["total"] for P, Q in ((A, B), (B, C), (A, C))]; assert abs(t[0] + t[1] - t[2]) < 1e-8

def test_prop12_certified_local_minimality():
    case = perioperative(); x, ens = _setup(case, S=20); sp = case.spec; bb = case.blackbox()
    res = solve(sp, ens, x, bb, 0.10, budget=40, scale_X=case.X); a = res.action; assert a is not None
    assert certify(ens, sp, x, a, BlackBox(case.proba), 0.10)[0] >= 0
    for i in range(len(a)):
        if a[i] != 0:
            b = a.copy(); b[i] = 0; assert certify(ens, sp, x, b, BlackBox(case.proba), 0.10)[0] < 0

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok  ", k)
