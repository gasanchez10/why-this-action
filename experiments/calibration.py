"""Is '(1 - eps) of the bootstrap versions' an honest success probability once the plan has been OPTIMISED?
For many data seeds: the reference robust plan (free queries) and whether it works under the true SCM.
  python -m experiments.calibration perioperative 40
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run import setup, reference
from experiments.summarize import wilson
name, n = sys.argv[1], int(sys.argv[2])
for S, eps in [(1, 0.0), (30, 0.10), (100, 0.10), (100, 0.02)]:
    ok, cost = [], []
    for seed in range(100, 100 + n):
        case, ens, xs, truth, tname, fm = setup(name, seed, S=S, n=1)
        if len(xs) == 0: continue
        a, c = reference(case.spec, ens, xs[0], fm, eps, n=6000, rounds=2, seed=seed)
        if a is not None: ok.append(fm(truth.counterfactual(xs[0], a)[0])[0] >= 0); cost.append(c)
    print(f"{name}: versions {S:>3}, tolerated risk {eps:.2f}: works under the true SCM in {int(sum(ok))}/{len(ok)} = {wilson(sum(ok), len(ok))}, mean cost {np.mean(cost):.3f}", flush=True)
