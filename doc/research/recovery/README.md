# Does the search find the structure that drew a series?

Forty series drawn from the grammar itself (`autostat.synthetic`): a random
structure one or two rewrites from a base kernel, parameters drawn as the
search's starts are and the search's bounds respected, ten years of monthly
values, white noise at a fifth of the signal's spread. Each series was
searched with the 2014 grammar and light settings (three levels, four random
starts, 100 optimiser iterations, two worker processes), then the true
structure was refitted from its true parameters to compare the two scores.
[run.py](run.py) reproduces it, [summarise.py](summarise.py) makes the table,
[results.json](results.json) holds every row. 2.3 hours of search in all.

| component | present | found when present | absent | invented when absent |
|---|---|---|---|---|
| cycle | 10 | 7 | 30 | 1 |
| changepoint or window | 14 | 11 | 26 | 2 |
| linear trend | 18 | 16 | 22 | 5 |
| smooth term | 14 | 9 | 26 | 4 |

- The search's model scores within 2 of the refitted truth, or better, on
  34 series of 40. The exact structure is found on 6 only: most of the time
  another structure explains the data as well; the data cannot tell them
  apart.
- On 6 series the truth scores better by more than 2. On one the search
  found the right structure and fitted it worse (a local optimum of the
  light settings). On the 5 others it found another structure.
- **The cycles are the weak point.** Five of those six misses hold a
  periodic kernel. Where the search finds the cycle, it finds its period
  almost exactly (0.706, 0.895, 0.691 and 1.267 for 0.706, 0.895, 0.691 and
  1.273 years); where it misses, it settles on a wrong period (0.688 for
  0.305, 0.261 for 1.029, 0.95 for 1.19) or on none. A true period is found
  within 5 % on 4 series of 10.

| seed | true structure | found | true period | found period | BIC, truth minus found |
|---|---|---|---|---|---|
| 0 | `CW((C + Per) x Per, C)` | `CW(WN, C)` | 0.436, 0.5 | none | -174.6 |
| 6 | `(C + Per) x Lin` | `Lin x SE + SE` | 1.348 | none | -158.2 |
| 27 | `Per + WN` | `Per + WN` | 0.305 | 0.688 | -71.6 |
| 29 | `(C + Per) x SE` | `Lin + Per x SE + WN` | 1.19 | 0.95 | -44.6 |
| 35 | `Per` | `Per x SE + WN` | 1.029 | 0.261 | -126.3 |
| 38 | `Lin x Per` | `Lin x SE + WN` | 1.294 | none | -117.6 |

Why the period: the 2014 starts draw a period around one of three fixed
centres, one year, about a week and about a day (`exp(0)`, `exp(-3.95)` and
`exp(-5.9)` years, spread 2 on the log scale). On monthly data the week and
the day lie mostly below the shortest admissible period. The optimiser
seldom climbs from there to a long cycle, since the likelihood of a period
has many local optima; a fit left there is discarded as out of bounds. Only
the first centre is in range; it reaches a period far from one year by its
spread alone. The second measurement below starts half the periods at
the peaks of the data's periodogram instead.

## Second measurement: periodogram starts

The same forty series, the same settings and seeds, with
`spectral_starts=3`: the periods of the data's three highest Lomb-Scargle
peaks join the start hints, so half the periodic starts begin at one of
them (`python doc/research/recovery/run.py results-spectral.json spectral`,
2.0 hours of search, rows in
[results-spectral.json](results-spectral.json)).

| component | present | found, fixed centres | found, periodogram | invented when absent, fixed / periodogram |
|---|---|---|---|---|
| cycle | 10 | 7 | 10 | 1 / 0 |
| changepoint or window | 14 | 11 | 10 | 2 / 2 |
| linear trend | 18 | 16 | 15 | 5 / 5 |
| smooth term | 14 | 9 | 8 | 4 / 4 |

- **A true period is found within 5 % on 9 series of 10**, against 4. Every
  cycle is found; none is invented. The miss left is seed 0, two cycles of
  0.436 and 0.5 years multiplied inside a window, where 0.70 is found.
- The search's model scores within 2 of the refitted truth, or better, on
  35 series of 40 (34 before); no fit falls short of the right structure
  any more (one before). The exact structure: 6 of 40 again.
- Series by series, the search's best BIC is lower by more than 2 on 7
  series, higher by more than 2 on 2, within 2 on 31. The gains are the
  cycles: seed 35 from -17.9 to -144.2, seed 38 from 85.0 to -27.8, seed
  6 from 956.4 to 815.6, seed 27 from 481.7 to 405.4. The two losses carry
  no cycle error. On seed 19 the period is found as before (0.691) but the
  changepoint is lost (545.0 to 639.7); on seed 31, which has no cycle, a
  window is fitted worse (441.6 to 458.1). Half the periodic starts are
  hints now, so the random starts left differ: with four random starts, a
  different draw moves a search with no relation to the cycles.
- The changepoint row loses one series and the linear and smooth rows one
  each, within the same variation.

The published presets stay as published: `spectral_starts` is 0 in
`Experiment.abcd_2014()` and `Experiment.cks_2013()`; a caller asks for
the periodogram starts explicitly (`--spectral 3` on the command line).

Forty draws at light settings are a first measure. The table counts what
the grammar's own prior produces, where products with constants and
switches are frequent; real series are not drawn from it.
