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

Six pairs on synthetic data are a first sign, not a proof. The second
measurement below runs more seeds, the full settings and a misleading
context.

## Second measurement: five seeds, full settings, a misleading context

The same two families, five seeds each, the published search settings
(three levels, nine random starts, 250 iterations, two worker processes)
and three arms per series: blind, the true context and a misleading one.
The misleading context names a cycle of five months instead of three for
the quarterly family; for the jump family it keeps the yearly cycle but
dates the event in 2012 instead of mid-2014. Every arm is read against the
true context. `python run.py full.json 5 full` reproduces it in 3.7 hours
on two cores; [full.json](full.json) holds the 30 rows.

| family | seed | BIC blind | BIC true | BIC wrong | expected structure found (blind / true / wrong) |
|---|---|---|---|---|---|
| quarterly | 0 | 389.1 | 389.1 | **381.2** | cycle: yes / yes / yes |
| quarterly | 1 | 393.0 | **374.2** | 387.1 | cycle: no (0.5) / yes / no |
| quarterly | 2 | 409.0 | 453.8 | **399.9** | cycle: no (0.5) / no (0.5) / no (0.5) |
| quarterly | 3 | 417.4 | 417.4 | 417.4 | cycle: yes / yes / yes |
| quarterly | 4 | 429.9 | **405.2** | **405.2** | cycle: no (1.0) / yes / yes |
| jump | 0 | **480.3** | 485.1 | 485.1 | cycle: yes / yes / yes; jump: no / no / no |
| jump | 1 | 474.8 | **455.9** | **455.9** | cycle: no / yes / yes; jump: no / no / no |
| jump | 2 | 489.3 | **484.5** | **484.5** | cycle: yes / yes / yes; jump: no / no / no |
| jump | 3 | 454.8 | **450.0** | **450.0** | cycle: yes / yes / yes; jump: yes / yes / yes |
| jump | 4 | 480.9 | 480.9 | 480.9 | cycle: yes / yes / yes; jump: no / no / no |

What the thirty runs say:

- **The expected cycle is found more often with the context:** the
  quarter 4 times out of 5 against 2 without; the year 5 out of 5 against
  4. Where the blind search misses the quarter, it settles on a cycle of
  half a year or a year, a multiple that fits less well (by 19 and 25 BIC
  for seeds 1 and 4).
- **The misleading context does no harm here.** No run kept a cycle of
  five months or a changepoint in 2012: BIC rejected both every time. The
  wrong period even found the quarter once where the blind search did not
  (seed 4). The likely reason is not knowledge but diversity: the hints
  change where half of the random starts land, so the search explores
  other basins. Part of the gain credited to the true context in the
  first measurement is therefore that diversity, not the period itself.
- **The context costs once.** For quarterly seed 2, the true arm ends on
  a cycle without a trend, 45 BIC worse than the blind run; no arm finds
  the quarter there. The hints remain a starting point: a search can still
  stop in a poor basin.
- **The event date did not help in the full settings.** The jump is found
  as a changepoint once out of five, by all three arms alike, 0.07 years
  from the true date. With the true or the wrong date the results are the
  same, so the location hints had no measurable effect here. The jump of
  mid-2014 is often absorbed by the smooth terms instead.
- **The measured effect of the event is biased low.** The counterfactual
  estimate (the model without its switches, fitted before the date and
  predicting the year after) gives 5.1 to 6.7 units for a true jump of 8.
  The interval covers 8 in four seeds out of five; in seed 3, where a
  changepoint is found, it reads [5.7, 7.6] and excludes it. The likely
  cause is that the smooth terms carry part of the jump into the
  prediction from before the date. This
  feature measures direction and order of magnitude, not the exact size.

Ten series on synthetic data still do not make a proof, but the picture
is consistent: the cycle hints help to recover an expected period that the
generic starts miss and BIC rejects a wrong hint. The event hints do not,
as they stand, help to locate a jump. The next steps are real
series with known events and a comparison with generic feature sets
(catch22, tsfresh).

## Third measurement: a level shift for each known event

Why did the event date not help? Fitted directly, the model of the right
shape beats the search's best by about 60 BIC on several jump series: the
search does not reach it. The grammar adds a changepoint only by turning
one term into `CP(term, term)` while the greedy levels prefer the yearly
cycle first. Logged level by level (seed 2), the search keeps
`Lin + SE + WN` (500.3) over the same plus a shift (547.0), then
`Lin + Per + SE + WN` (484.5) just ahead of `Lin + SE + WN` plus a shift
(488.1). The shift of the final model would win, but it would come at a
fourth level that never runs.

Two changes follow, active only when the context names an event date:
every level proposes each kept model plus a level shift
`CP(Zero, Const)` with its fitted parameters kept; one last stage after
the levels does the same for the final model. BIC still decides.
The quarterly runs and the blind runs carry no date, so they do not
change; the ten jump runs with a date were run again with the same
settings. [event_stage.json](event_stage.json) holds all 30 rows.

| seed | BIC true (before → after) | BIC wrong date (before → after) | jump found (true / wrong) | effect with the true date, 95 % interval |
|---|---|---|---|---|
| 0 | 485.1 → **417.6** | 485.1 → **419.9** | yes, 0.08 / yes, 0.03 | 6.6 [5.6, 7.6] |
| 1 | 455.9 → **383.3** | 455.9 → **383.3** | yes, 0.08 / yes, 0.08 | 7.1 [6.6, 7.6] |
| 2 | 484.5 → **416.2** | 484.5 → 484.5 | yes, 0.00 / no | 7.4 [6.8, 8.0] |
| 3 | 450.0 → **438.9** | 450.0 → 450.0 | yes, 0.07 / yes, 0.07 | 6.6 [5.7, 7.6] |
| 4 | 480.9 → **412.4** | 480.9 → 480.9 | yes, 0.00 / no | 7.8 [7.2, 8.4] |

The gaps are in years, from the true date (mid-2014); with monthly data
the first point after the jump lies 0.08 years after it.

What the ten runs say:

- **With the true date, the jump is found 5 times out of 5** instead of
  once. BIC improves by 11 to 73. The shift lands at most one month
  from the date.
- **The measured effect is closer to the truth and tighter:** 6.6 to 7.8
  for a true 8, with intervals three to eight times narrower than
  through the smooth terms in the four seeds where no changepoint was
  found before. It is still biased low; the interval covers 8 in two
  seeds out of five. The bias is smaller than before but has not gone.
- **A wrong date creates no false changepoint.** No run kept a shift in
  2012. In two seeds out of five a shift that started at a random
  date moved to the real jump, so a date roughly right is enough to make the
  search look for a jump. Read against the context that named 2012, such
  a run would say what matters: the expected event is absent and an
  unexpected changepoint sits in mid-2014.
