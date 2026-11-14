# Results summary

Simulation results produced by `experiments/run.py`. Rates show count, percentage and 95 % Wilson interval.
Rows with few cases are NOT conclusive. For real data (german, adult, sachs) 'valid in truth' uses a held-out re-fit of the SCM as a proxy.

## C · Search

### dag10  (2 data seeds, 4 individuals; truth = true SCM)

| method                                                                      | found   |   cost (mean) | gap to reference, % (median / max)   | robustly valid       | valid in truth       |   queries (median) |
|:----------------------------------------------------------------------------|:--------|--------------:|:-------------------------------------|:---------------------|:---------------------|-------------------:|
| reference optimum (robust problem, free queries)                            | 4/4     |         0.681 | 0.0 / 0.0                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                  0 |
| oracle (true SCM, free queries)                                             | 4/4     |         0.679 | -1.2 / 6.8                           | 2/4 = 50 % (15–85)   | 4/4 = 100 % (51–100) |                  0 |
| Algorithm 1                                                                 | 4/4     |         0.686 | 1.1 / 1.4                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                230 |
| Algorithm 1 + polish (extension)                                            | 4/4     |         0.686 | 1.1 / 1.4                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                350 |
| ablation: no blocking                                                       | 4/4     |         0.782 | 1.4 / 45.3                           | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                200 |
| ablation: no certification                                                  | 4/4     |         0.692 | 1.9 / 3.1                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                 50 |
| ablation: no causal propagation (non-causal stand-in)                       | 4/4     |         2.011 | 178.1 / 328.5                        | 3/4 = 75 % (30–95)   | 3/4 = 75 % (30–95)   |                 59 |
| baseline: no abduction (stand-in for causal BO without abduction)           | 4/4     |         2.093 | 202.0 / 969.3                        | 3/4 = 75 % (30–95)   | 3/4 = 75 % (30–95)   |                515 |
| point estimate (one version)                                                | 4/4     |         0.649 | -4.7 / -3.5                          | 0/4 = 0 % (0–49)     | 2/4 = 50 % (15–85)   |                 57 |
| ablation: surrogate on actions                                              | 4/4     |         0.695 | 0.6 / 3.9                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                900 |
| baseline: random search with certification (2000 queries)                   | 4/4     |         0.683 | 0.3 / 0.5                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |               2000 |
| baseline: grid with our criterion                                           | 4/4     |         0.808 | 11.7 / 92.0                          | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |              87480 |
| baseline: Karimi et al. 2020 GP-SCM + grid (our re-implementation, gamma=1) | 4/4     |         0.912 | 53.2 / 68.7                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |              45630 |
| Algorithm 1, budget 20                                                      | 4/4     |         0.686 | 1.1 / 1.4                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                200 |
| Algorithm 1, budget 40                                                      | 4/4     |         0.686 | 1.1 / 1.4                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                220 |
| Algorithm 1, budget 80                                                      | 4/4     |         0.685 | 0.7 / 1.3                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                275 |
| Algorithm 1, budget 160                                                     | 4/4     |         0.685 | 0.5 / 0.8                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                370 |

### german  (2 data seeds, 4 individuals; truth = held-out re-fit (proxy))

| method                                                            | found   |   cost (mean) | gap to reference, % (median / max)   | robustly valid       | valid in truth       |   queries (median) |
|:------------------------------------------------------------------|:--------|--------------:|:-------------------------------------|:---------------------|:---------------------|-------------------:|
| reference optimum (robust problem, free queries)                  | 4/4     |         0.217 | 0.0 / 0.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                  0 |
| Algorithm 1                                                       | 4/4     |         0.239 | 0.3 / 364.3                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                425 |
| Algorithm 1 + polish (extension)                                  | 4/4     |         0.233 | 0.3 / 256.8                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                590 |
| ablation: no blocking                                             | 4/4     |         0.219 | 0.9 / 6.2                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                380 |
| ablation: no certification                                        | 4/4     |         0.222 | 0.3 / 138.5                          | 2/4 = 50 % (15–85)   | 2/4 = 50 % (15–85)   |                 50 |
| ablation: no causal propagation (non-causal stand-in)             | 4/4     |         0.217 | 0.1 / 6.1                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                 61 |
| baseline: no abduction (stand-in for causal BO without abduction) | 4/4     |         0.219 | 1.0 / 4.4                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                440 |
| point estimate (one version)                                      | 4/4     |         0.219 | 1.1 / 3.8                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                 60 |
| ablation: surrogate on actions                                    | 4/4     |         0.232 | 7.4 / 15.1                           | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                900 |
| baseline: random search with certification (2000 queries)         | 4/4     |         0.217 | 0.0 / 0.1                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |               2000 |
| baseline: grid with our criterion                                 | 4/4     |         0.248 | 16.3 / 114.9                         | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |              86640 |
| Algorithm 1, budget 20                                            | 4/4     |         0.239 | 0.3 / 364.3                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                395 |
| Algorithm 1, budget 40                                            | 4/4     |         0.239 | 0.3 / 364.3                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                415 |
| Algorithm 1, budget 80                                            | 4/4     |         0.239 | 0.3 / 364.3                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                455 |
| Algorithm 1, budget 160                                           | 4/4     |         0.217 | 0.3 / 2.8                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                595 |

### loan7  (2 data seeds, 4 individuals; truth = true SCM)

| method                                                                      | found   |   cost (mean) | gap to reference, % (median / max)   | robustly valid       | valid in truth       |   queries (median) |
|:----------------------------------------------------------------------------|:--------|--------------:|:-------------------------------------|:---------------------|:---------------------|-------------------:|
| reference optimum (robust problem, free queries)                            | 4/4     |         0.064 | 0.0 / 0.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                  0 |
| oracle (true SCM, free queries)                                             | 4/4     |         0.053 | -10.0 / 0.0                          | 2/4 = 50 % (15–85)   | 4/4 = 100 % (51–100) |                  0 |
| Algorithm 1                                                                 | 4/4     |         0.065 | 0.9 / 1.3                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                350 |
| Algorithm 1 + polish (extension)                                            | 4/4     |         0.065 | 0.9 / 1.3                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                470 |
| ablation: no blocking                                                       | 4/4     |         0.065 | 0.7 / 1.5                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                260 |
| ablation: no certification                                                  | 4/4     |         0.081 | 23.2 / 29.4                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                 50 |
| ablation: no causal propagation (non-causal stand-in)                       | 4/4     |         0.087 | 17.4 / 85.9                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                 58 |
| baseline: no abduction (stand-in for causal BO without abduction)           | 4/4     |         0.047 | -10.0 / 1.4                          | 2/4 = 50 % (15–85)   | 2/4 = 50 % (15–85)   |                500 |
| point estimate (one version)                                                | 4/4     |         0.059 | -3.5 / 0.8                           | 2/4 = 50 % (15–85)   | 4/4 = 100 % (51–100) |                 60 |
| ablation: surrogate on actions                                              | 4/4     |         0.075 | 0.4 / 25.8                           | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                900 |
| baseline: random search with certification (2000 queries)                   | 4/4     |         0.072 | 0.1 / 76.7                           | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |               2003 |
| baseline: grid with our criterion                                           | 4/4     |         0.084 | 30.7 / 244.7                         | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |              75000 |
| baseline: Karimi et al. 2020 GP-SCM + grid (our re-implementation, gamma=1) | 4/4     |         0.077 | 31.7 / 316.8                         | 3/4 = 75 % (30–95)   | 4/4 = 100 % (51–100) |              38880 |
| Algorithm 1, budget 20                                                      | 4/4     |         0.065 | 0.9 / 1.3                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                320 |
| Algorithm 1, budget 40                                                      | 4/4     |         0.065 | 0.9 / 1.3                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                340 |
| Algorithm 1, budget 80                                                      | 4/4     |         0.065 | 0.9 / 1.3                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                380 |
| Algorithm 1, budget 160                                                     | 4/4     |         0.065 | 0.9 / 1.3                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                460 |

### perioperative  (2 data seeds, 4 individuals; truth = true SCM)

| method                                                                      | found   |   cost (mean) | gap to reference, % (median / max)   | robustly valid       | valid in truth       |   queries (median) |
|:----------------------------------------------------------------------------|:--------|--------------:|:-------------------------------------|:---------------------|:---------------------|-------------------:|
| reference optimum (robust problem, free queries)                            | 4/4     |         0.911 | 0.0 / 0.0                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                  0 |
| oracle (true SCM, free queries)                                             | 4/4     |         0.915 | 0.6 / 1.5                            | 3/4 = 75 % (30–95)   | 4/4 = 100 % (51–100) |                  0 |
| Algorithm 1                                                                 | 4/4     |         0.918 | 0.8 / 0.9                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                245 |
| Algorithm 1 + polish (extension)                                            | 4/4     |         0.918 | 0.8 / 0.9                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                365 |
| ablation: no blocking                                                       | 4/4     |         1.053 | 11.8 / 40.6                          | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                305 |
| ablation: no certification                                                  | 4/4     |         0.9   | 0.1 / 0.2                            | 3/4 = 75 % (30–95)   | 2/4 = 50 % (15–85)   |                 50 |
| ablation: no causal propagation (non-causal stand-in)                       | 4/4     |         1.092 | 18.5 / 38.5                          | 1/4 = 25 % (5–70)    | 1/4 = 25 % (5–70)    |                 58 |
| baseline: no abduction (stand-in for causal BO without abduction)           | 3/4     |         1.005 | -36.5 / 225.5                        | 1/3 = 33 % (6–79)    | 1/3 = 33 % (6–79)    |                245 |
| point estimate (one version)                                                | 4/4     |         0.904 | -1.0 / 0.9                           | 1/4 = 25 % (5–70)    | 2/4 = 50 % (15–85)   |                 54 |
| ablation: surrogate on actions                                              | 4/4     |         0.911 | 0.0 / 0.1                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                900 |
| baseline: random search with certification (2000 queries)                   | 4/4     |         0.913 | 0.2 / 0.4                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |               2002 |
| baseline: grid with our criterion                                           | 4/4     |         0.919 | 0.7 / 1.8                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |              86640 |
| baseline: Karimi et al. 2020 GP-SCM + grid (our re-implementation, gamma=1) | 4/4     |         1.115 | 17.7 / 57.0                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |              43320 |
| Algorithm 1, budget 20                                                      | 4/4     |         0.918 | 0.8 / 0.9                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                215 |
| Algorithm 1, budget 40                                                      | 4/4     |         0.918 | 0.8 / 0.9                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                235 |
| Algorithm 1, budget 80                                                      | 4/4     |         0.918 | 0.8 / 0.9                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                275 |
| Algorithm 1, budget 160                                                     | 4/4     |         0.918 | 0.8 / 0.9                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                355 |

### synth3-anm  (2 data seeds, 4 individuals; truth = true SCM)

| method                                                                      | found   |   cost (mean) | gap to reference, % (median / max)   | robustly valid       | valid in truth       |   queries (median) |
|:----------------------------------------------------------------------------|:--------|--------------:|:-------------------------------------|:---------------------|:---------------------|-------------------:|
| reference optimum (robust problem, free queries)                            | 4/4     |         0.537 | 0.0 / 0.0                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                  0 |
| oracle (true SCM, free queries)                                             | 4/4     |         0.545 | -1.9 / 7.2                           | 1/4 = 25 % (5–70)    | 4/4 = 100 % (51–100) |                  0 |
| Algorithm 1                                                                 | 4/4     |         0.54  | 0.6 / 0.9                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                170 |
| Algorithm 1 + polish (extension)                                            | 4/4     |         0.54  | 0.6 / 0.9                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                290 |
| ablation: no blocking                                                       | 4/4     |         0.541 | 0.2 / 8.5                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                170 |
| ablation: no certification                                                  | 4/4     |         0.613 | 12.9 / 80.7                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                 50 |
| ablation: no causal propagation (non-causal stand-in)                       | 4/4     |         0.914 | 62.2 / 79.7                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                 57 |
| baseline: no abduction (stand-in for causal BO without abduction)           | 4/4     |         0.636 | 21.7 / 56.0                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                350 |
| point estimate (one version)                                                | 4/4     |         0.494 | -9.1 / -4.7                          | 0/4 = 0 % (0–49)     | 1/4 = 25 % (5–70)    |                 55 |
| ablation: surrogate on actions                                              | 4/4     |         0.555 | 0.3 / 9.8                            | 4/4 = 100 % (51–100) | 2/4 = 50 % (15–85)   |                900 |
| baseline: random search with certification (2000 queries)                   | 4/4     |         0.539 | 0.2 / 0.6                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |               2008 |
| baseline: grid with our criterion                                           | 4/4     |         0.763 | 53.0 / 282.3                         | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |              87480 |
| baseline: Karimi et al. 2020 GP-SCM + grid (our re-implementation, gamma=1) | 4/4     |         0.713 | 28.1 / 205.9                         | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |              39930 |
| Algorithm 1, budget 20                                                      | 4/4     |         0.54  | 0.6 / 0.9                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                140 |
| Algorithm 1, budget 40                                                      | 4/4     |         0.54  | 0.6 / 0.9                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                160 |
| Algorithm 1, budget 80                                                      | 4/4     |         0.54  | 0.6 / 0.9                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                200 |
| Algorithm 1, budget 160                                                     | 4/4     |         0.54  | 0.6 / 0.9                            | 4/4 = 100 % (51–100) | 3/4 = 75 % (30–95)   |                280 |

### synth3-linear  (2 data seeds, 4 individuals; truth = true SCM)

| method                                                                      | found   |   cost (mean) | gap to reference, % (median / max)   | robustly valid       | valid in truth       |   queries (median) |
|:----------------------------------------------------------------------------|:--------|--------------:|:-------------------------------------|:---------------------|:---------------------|-------------------:|
| reference optimum (robust problem, free queries)                            | 4/4     |         0.533 | 0.0 / 0.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                  0 |
| oracle (true SCM, free queries)                                             | 4/4     |         0.434 | -18.3 / -17.4                        | 0/4 = 0 % (0–49)     | 4/4 = 100 % (51–100) |                  0 |
| Algorithm 1                                                                 | 4/4     |         0.536 | 0.3 / 1.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                245 |
| Algorithm 1 + polish (extension)                                            | 4/4     |         0.536 | 0.3 / 1.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                365 |
| ablation: no blocking                                                       | 4/4     |         0.536 | 0.4 / 1.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                215 |
| ablation: no certification                                                  | 4/4     |         0.558 | 3.8 / 11.8                           | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                 50 |
| ablation: no causal propagation (non-causal stand-in)                       | 4/4     |         0.563 | 5.4 / 6.1                            | 3/4 = 75 % (30–95)   | 4/4 = 100 % (51–100) |                 58 |
| baseline: no abduction (stand-in for causal BO without abduction)           | 4/4     |         0.51  | 5.2 / 5.8                            | 3/4 = 75 % (30–95)   | 3/4 = 75 % (30–95)   |                200 |
| point estimate (one version)                                                | 4/4     |         0.475 | -9.7 / -8.2                          | 0/4 = 0 % (0–49)     | 4/4 = 100 % (51–100) |                 57 |
| ablation: surrogate on actions                                              | 4/4     |         0.575 | 0.1 / 19.1                           | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                900 |
| baseline: random search with certification (2000 queries)                   | 4/4     |         0.539 | 0.2 / 5.6                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |               2002 |
| baseline: grid with our criterion                                           | 4/4     |         0.674 | 26.8 / 63.9                          | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |              87480 |
| baseline: Karimi et al. 2020 GP-SCM + grid (our re-implementation, gamma=1) | 4/4     |         0.809 | 60.8 / 115.2                         | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |              39930 |
| Algorithm 1, budget 20                                                      | 4/4     |         0.536 | 0.3 / 1.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                215 |
| Algorithm 1, budget 40                                                      | 4/4     |         0.536 | 0.3 / 1.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                235 |
| Algorithm 1, budget 80                                                      | 4/4     |         0.536 | 0.3 / 1.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                275 |
| Algorithm 1, budget 160                                                     | 4/4     |         0.536 | 0.3 / 1.0                            | 4/4 = 100 % (51–100) | 4/4 = 100 % (51–100) |                355 |

## A · Fidelity of the answers

Each row of the first table is a kind of question. 'Agrees with truth' = the answer (plan still valid: yes/no) matches what happens under the true SCM.

| case          | question   |   questions | exact mode agrees with truth   | GP mode agrees with exact mode   |   queries exact (mean) |   queries GP mode (mean) |
|:--------------|:-----------|------------:|:-------------------------------|:---------------------------------|-----------------------:|-------------------------:|
| dag10         | Q0         |           4 | 2/4 = 50 % (15–85)             | 4/4 = 100 % (51–100)             |                     30 |                     23.8 |
| dag10         | Q1         |          12 | 11/12 = 92 % (65–99)           | 12/12 = 100 % (76–100)           |                     30 |                     13.8 |
| dag10         | Q2         |          29 | 28/29 = 97 % (83–99)           | 28/29 = 97 % (83–99)             |                     30 |                     18.5 |
| dag10         | Q3         |           1 | 1/1 = 100 % (21–100)           | 1/1 = 100 % (21–100)             |                     30 |                      0   |
| loan7         | Q0         |           4 | 4/4 = 100 % (51–100)           | 4/4 = 100 % (51–100)             |                     30 |                     19   |
| loan7         | Q1         |          12 | 11/12 = 92 % (65–99)           | 12/12 = 100 % (76–100)           |                     30 |                     11   |
| loan7         | Q2         |           2 | 2/2 = 100 % (34–100)           | 2/2 = 100 % (34–100)             |                     30 |                      0   |
| perioperative | Q0         |           4 | 2/4 = 50 % (15–85)             | 4/4 = 100 % (51–100)             |                     30 |                     27.8 |
| perioperative | Q1         |          12 | 12/12 = 100 % (76–100)         | 12/12 = 100 % (76–100)           |                     30 |                      5.4 |
| perioperative | Q2         |           6 | 6/6 = 100 % (61–100)           | 6/6 = 100 % (61–100)             |                     30 |                      0   |
| perioperative | Q3         |           4 | 4/4 = 100 % (51–100)           | 4/4 = 100 % (51–100)             |                     30 |                     17.5 |

Credits of the explanation game (Q5):

| case          |   plans |   paths per plan (mean) |   action components (mean) |   fidelity F (median) |   fidelity, absolute-value variant (median) |   credit error vs true SCM (mean, relative) | same top path as true SCM   |
|:--------------|--------:|------------------------:|---------------------------:|----------------------:|--------------------------------------------:|--------------------------------------------:|:----------------------------|
| dag10         |       4 |                    14.5 |                        1.2 |                 1.29  |                                       0.106 |                                       0.119 | 3/4 = 75 % (30–95)          |
| loan7         |       4 |                     1.5 |                        1   |                 1     |                                       0.987 |                                       0.041 | 3/4 = 75 % (30–95)          |
| perioperative |       4 |                     4.5 |                        2   |                 0.839 |                                       0.639 |                                       0.042 | 4/4 = 100 % (51–100)        |

## B · When blocking pays off

Reference optimum of the robust problem with and without edge blocking.

| case          |   individuals | optimum uses a block   | a plan exists only with blocking   | saving when both exist, % (mean / max)   | valid in truth, with   | valid in truth, without   |
|:--------------|--------------:|:-----------------------|:-----------------------------------|:-----------------------------------------|:-----------------------|:--------------------------|
| dag10         |             4 | 1/4 = 25 % (5–70)      | 0/4 = 0 % (0–49)                   | 7.7 / 31.0                               | 2/4 = 50 % (15–85)     | 2/4 = 50 % (15–85)        |
| german        |             4 | 2/4 = 50 % (15–85)     | 0/4 = 0 % (0–49)                   | 0.0 / 0.0                                | 4/4 = 100 % (51–100)   | 4/4 = 100 % (51–100)      |
| loan7         |             4 | 3/4 = 75 % (30–95)     | 0/4 = 0 % (0–49)                   | -0.0 / 0.0                               | 4/4 = 100 % (51–100)   | 4/4 = 100 % (51–100)      |
| perioperative |             4 | 4/4 = 100 % (51–100)   | 0/4 = 0 % (0–49)                   | 14.2 / 28.5                              | 2/4 = 50 % (15–85)     | 2/4 = 50 % (15–85)        |

## E · Stability across search seeds

Same individual and same SCM estimates, different search seeds.

| case          |   individuals |   cost variation across seeds, % (mean) |   credit instability (0 = identical) | same top credit in all runs   |   telescoping error (max) |   step-by-step vs global edge credits (relative gap, mean) |
|:--------------|--------------:|----------------------------------------:|-------------------------------------:|:------------------------------|--------------------------:|-----------------------------------------------------------:|
| loan7         |             4 |                                    0.28 |                                0.003 | 4/4 = 100 % (51–100)          |                   1.8e-15 |                                                      0.031 |
| perioperative |             4 |                                    0.26 |                                0.031 | 4/4 = 100 % (51–100)          |                   4.4e-16 |                                                      0.023 |

## X · Three layers of error (extension)

p = share of versions in which the plan works; 'total' adds a 3 % chance that a block does not hold (expert input, illustrative).

| case          | graph_knowledge                         | method                                          | plan found   |   cost (mean) | valid in truth       |   p mechanisms |   p + graph |   p total | passes localized conformal check   |   weight of the true graph |   queries (median) |
|:--------------|:----------------------------------------|:------------------------------------------------|:-------------|--------------:|:---------------------|---------------:|------------:|----------:|:-----------------------------------|---------------------------:|-------------------:|
| dag10         | expert gives the causal order           | mechanisms only (graph assumed)                 | 4/4          |         0.686 | 2/4 = 50 % (15–85)   |          0.95  |       0.943 |     0.936 | 0/4 = 0 % (0–49)                   |                       0.15 |                230 |
| dag10         | expert gives the causal order           | conformal box on the mechanisms (graph assumed) | 0/4          |       nan     | –                    |        nan     |     nan     |   nan     | –                                  |                       0.15 |                 50 |
| dag10         | expert gives the causal order           | graph + mechanisms                              | 4/4          |         0.682 | 2/4 = 50 % (15–85)   |          0.917 |       0.931 |     0.924 | 0/4 = 0 % (0–49)                   |                       0.15 |                424 |
| dag10         | no prior knowledge (approximate search) | graph + mechanisms                              | 3/4          |         2.141 | 2/3 = 67 % (21–94)   |          0.667 |       0.908 |     0.908 | 0/3 = 0 % (0–56)                   |                       0    |                450 |
| perioperative | expert gives the causal order           | mechanisms only (graph assumed)                 | 4/4          |         0.918 | 2/4 = 50 % (15–85)   |          0.967 |       0.97  |     0.921 | 1/4 = 25 % (5–70)                  |                       0.47 |                245 |
| perioperative | expert gives the causal order           | conformal box on the mechanisms (graph assumed) | 3/4          |         1.328 | 3/3 = 100 % (44–100) |          1     |       1     |     0.967 | 2/3 = 67 % (21–94)                 |                       0.47 |                230 |
| perioperative | expert gives the causal order           | graph + mechanisms                              | 4/4          |         0.916 | 2/4 = 50 % (15–85)   |          0.908 |       0.951 |     0.902 | 1/4 = 25 % (5–70)                  |                       0.47 |                296 |
| perioperative | no prior knowledge (approximate search) | graph + mechanisms                              | 4/4          |         1.087 | 4/4 = 100 % (51–100) |          1     |       0.961 |     0.941 | 1/4 = 25 % (5–70)                  |                       0.03 |                300 |

## D · Sheet for clinicians (example)

```
FICHA — caso perioperative

Paciente: Age = 67.8, HbA1c = 6.85, eGFR = 59.4, Hb = 11.4, SBP = 170
Plan recomendado: HbA1c -0.643, block Hb->SBP  (costo 0.643)
Funciona en 100 % de las 30 versiones del modelo causal (intervalo de 95 %: 89 a 100 %). Riesgo tolerado: 10 %.

¿Por qué esta acción? Aporte de cada camino a la ganancia del plan:
  +0.744  change HbA1c -> model
  +0.171  change HbA1c -> eGFR -> Hb -> model
  +0.149  change HbA1c -> eGFR -> model
  +0.049  block Hb->SBP | SBP -> model
  -0.046  change HbA1c -> eGFR -> Hb -> SBP -> model
  Sumados uno por uno, los caminos dan el 87 % de la ganancia; el resto son interacciones entre caminos (fidelidad).

Si el cambio se logra solo en parte: fracción mínima que mantiene el plan válido
  HbA1c: 100 %

Si una relación causal no se cumple:
  HbA1c->eGFR: el plan DEJA de ser válido (funciona en 0 % de las versiones)
  eGFR->Hb: el plan DEJA de ser válido (funciona en 0 % de las versiones)

Si el bloqueo no se sostiene:
  Hb->SBP: el plan DEJA de ser válido (funciona en 0 %)
```

## Calibration of the robust criterion (experiments/calibration.py)

One individual per data seed, 40 data seeds; the plan is the reference optimum of the robust problem.

```
dag10: versions   1, tolerated risk 0.00: works under the true SCM in 19/40 = 48 % (33–63), mean cost 1.191
dag10: versions  30, tolerated risk 0.10: works under the true SCM in 36/39 = 92 % (80–97), mean cost 1.184
dag10: versions 100, tolerated risk 0.10: works under the true SCM in 36/40 = 90 % (77–96), mean cost 1.314
dag10: versions 100, tolerated risk 0.02: works under the true SCM in 39/39 = 100 % (91–100), mean cost 1.269
perioperative: versions   1, tolerated risk 0.00: works under the true SCM in 27/40 = 68 % (52–80), mean cost 0.748
perioperative: versions  30, tolerated risk 0.10: works under the true SCM in 37/40 = 92 % (80–97), mean cost 0.757
perioperative: versions 100, tolerated risk 0.10: works under the true SCM in 37/40 = 92 % (80–97), mean cost 0.758
perioperative: versions 100, tolerated risk 0.02: works under the true SCM in 39/40 = 98 % (87–100), mean cost 0.763
```
