# Does the context help the search? A first measurement

Two synthetic families whose truth is known, three seeds each, searched
twice with the same light settings (three levels, four random starts, 100
optimiser iterations, two worker processes): once blind, once with the
context's hints (`Experiment.abcd_2014(**context.search_hints())`).
[run.py](run.py) reproduces it; [results.json](results.json) holds every row.

- **Quarterly:** monthly data, a trend, a cycle of three months (a period
  the 2014 random starts do not aim at) and noise.
- **Jump:** monthly data, a trend, a yearly cycle and a level jump of 8
  units in mid-2014, the date the context names.

| family | seed | BIC blind | BIC guided | what changed |
|---|---|---|---|---|
| quarterly | 0 | 389.1 | 389.1 | the same model |
| quarterly | 1 | **383.6** | 404.6 | both find the cycle; blind fits better by 21 |
| quarterly | 2 | 432.7 | **397.8** | blind misses the cycle (finds 0.625 years); guided finds 0.25 |
| jump | 0 | 494.0 | **490.8** | guided finds the yearly cycle; blind does not |
| jump | 1 | 474.8 | **392.9** | guided finds the jump 0.04 years from the date and the cycle |
| jump | 2 | 489.3 | 489.3 | the same model |

What the six pairs say:

- The expected cycle is recovered 3 times out of 3 with the context, 2 out
  of 3 without for the quarter; the yearly cycle 3 out of 3 against 1 out
  of 3 in the jump family.
- The jump is found as a changepoint once, with the context. There, the
  measured effect is 7.3 units, 95 % interval [6.6, 8.1], around the true
  8. Without it, the effects read through the smooth terms are lower and
  wider (for seed 0, [-0.1, 8.6]).
- When the blind search already finds the truth, the context changes
  nothing (two ties). Once, it costs 21 BIC: half the random starts go to
  the hints, so a light search explores less. A full search, with nine
  starts, should not pay that price; this is to be measured.

Six pairs on synthetic data are a first sign, not a proof. The next
measurement runs more seeds, the full search settings, a misleading
context (a wrong period, a wrong date) and real series with known events.
