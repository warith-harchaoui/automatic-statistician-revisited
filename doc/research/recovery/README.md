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

Forty draws at light settings are a first measure. The table counts what
the grammar's own prior produces, where products with constants and
switches are frequent; real series are not drawn from it.
