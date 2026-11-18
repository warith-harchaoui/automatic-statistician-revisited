# Would a changepoint detector help the search?

The 2014 grammar builds a changepoint from a smooth term, then places it.
On the jump family it missed the jump six times of six
([propose](../propose/README.md)). A language model told the date fixes
that (mean BIC 411.7 against 486.0); told nothing, it guesses dates and
can do worse than the grammar. A changepoint detector gives dates from
the data alone, in milliseconds. This note asks where a detector should
look (the raw values or the components a decomposition extracts) and
what the literature says about each.

## A pilot: where the detector looks

[pilot.py](pilot.py) runs PELT (`ruptures`) on the two synthetic families
of [context_search](../context_search/README.md), five seeds each: the
jump family (trend, yearly cycle, a jump of 8 in mid-2014, noise 1) and
the quarterly family (trend, three-month cycle, no jump). The period is
not given: it is the highest Lomb-Scargle peak of the detrended values.
Every penalty is BIC-like. It runs in seconds.

| reading of the series | jump: breaks in 5 series | jump: one at mid-2014 | quarterly: false breaks in 5 |
|---|---|---|---|
| raw values, constant segments | 71 | 5 of 5 | 5 |
| STL trend, constant segments | 33 | 5 of 5 | 5 |
| STL remainder, constant segments | 14 | 5 of 5 | 0 |
| remainder of one line-and-harmonics fit | 28 | 5 of 5 | 0 |
| **line and harmonics inside the cost** | **5** | **5 of 5** | **0** |
| **STL seasonal removed, line inside the cost** | **5** | **5 of 5** | **0** |

- **On the raw values, a constant-mean detector cuts every cycle.** The
  yearly cycle gives about 14 breaks per series; the quarterly trend is
  cut once into two steps. The true date is among them, unrecognisable.
- **On the components of a decomposition, it depends which.** The STL
  trend is smoothed: the jump becomes a ramp, cut into steps (33 breaks).
  The STL remainder carries what the smoothed trend could not follow: the
  jump is found, split in three (2014.12, 2014.54, 2014.96), the span of
  the trend smoother. The remainder of one global fit with no break is two
  ramps, cut many times.
- **The detector works when its cost holds the structure.** With a line
  and two harmonics fitted per segment, or the STL seasonal part removed
  and a line per segment, it finds one break per series, at 2014.54 every
  time; none on the quarterly family.

The same question on the components of the Gaussian process the grammar
keeps (`Lin + SE + WN` on the jump family, three seeds, refitted with the
light settings):

| seed | SE lengthscale | breaks in the SE component | line and harmonics in the cost | breaks in the residual |
|---|---|---|---|---|
| 0 | 0.25 year | 14 | 2014.54 | none |
| 1 | 0.94 year | 7 | 2013.71, 2015.38 | 15 |
| 2 | 0.25 year | 12 | 2014.54 | none |

- **The residual of a flexible fit hides the jump.** A smooth term with a
  short lengthscale absorbs the cycle and the jump together; what is left
  is white (seeds 0 and 2). Model checking on residuals would pass.
- **The smooth component holds the jump.** Read with the cycle inside the
  cost, it gives the date on two seeds of three. With a long lengthscale
  (seed 1) the step is smeared over a year and read as two breaks around
  it. A component's shape is that of the kernel which made it: a
  detector on it reads the kernel as much as the data.

## What the literature says

- **Detectors alone are weak; they help as starts.** On TCPDBench, 37
  annotated real series, the best methods at their default settings
  barely beat a method that never reports a change: F1 0.698 for binary
  segmentation, 0.674 for PELT, 0.645 for "no change"; Prophet's
  changepoints score 0.472 (Burg and Williams, [arXiv
  2003.06222](https://arxiv.org/abs/2003.06222), Table 2). Tuned per
  series they reach 0.86 to 0.89. In autostat the detector would only
  propose; BIC would decide.
- **A detector assumes a model between changes; a wrong one invents
  changes.** Standard costs assume independent noise and a constant mean;
  with drift and autocorrelation they "substantially over-estimate the
  number of changes". DeCAFS models a random-walk drift and AR(1) noise
  inside the cost (Romano, Rigaill, Runge and Fearnhead, JASA 2022,
  [arXiv 2005.01379](https://arxiv.org/abs/2005.01379)). The pilot's raw
  and global-remainder rows are that failure; its last two rows are the
  remedy, with trend and cycle in the cost.
- **Decompose then detect smears the jump.** STL's trend "does not follow
  the abrupt change" and its seasonal part is "highly affected by the
  level shifts" (Wen et al., RobustSTL, AAAI 2019, [arXiv
  1812.01767](https://arxiv.org/abs/1812.01767)); RobustSTL fits the trend
  with a sparse L1 penalty to keep steps. The pilot's STL rows show both
  effects.
- **Trend and season can each break.** BFAST iterates a decomposition and
  tests breaks in the trend and in the seasonal coefficients separately
  (Verbesselt et al., Remote Sensing of Environment 2010, R package
  [bfast](https://cran.r-project.org/package=bfast)). BEAST averages over
  decompositions with any number of changepoints and returns the
  probability of a change at each date (Zhao et al., Remote Sensing of
  Environment 2019, [Rbeast](https://cran.r-project.org/package=Rbeast)):
  the closest older relative of a Bayesian answer to "is there a jump".
- **Changes in spread or in the cycle need another cost.** Kernel
  changepoint detection compares distributions rather than means; it chooses
  the number of changes by a non-asymptotic penalty (Arlot, Celisse and
  Harchaoui, [JMLR 2019](https://jmlr.org/papers/v20/16-155.html)). That
  is what a `CW` on a noise or a cycle term models.
- **A date found on the data is not tested on the same data.** A p-value
  for a change at a detected date is anti-conservative unless it
  conditions on the detection (Jewell, Fearnhead and Witten, JRSS B 2022;
  Carrington and Fearnhead, [arXiv
  2301.05636](https://arxiv.org/abs/2301.05636) for more power, [arXiv
  2405.15670](https://arxiv.org/abs/2405.15670) for changes in variance).
  autostat compares structures by BIC after refitting the location; the
  detector's own statistic should never be reported as evidence.
- **Libraries.** `ruptures` (Truong, Oudre and Vayatis, review in Signal
  Processing 167, 2020) has the costs used here, `linear` among them.
  `skchange` (Tveten, Kolstø and Moen, [arXiv
  2608.19767](https://arxiv.org/abs/2608.19767)) adds penalties calibrated
  by simulation on change-free data.
- **Structure search does not use detectors.** AutoGP (Saad et al., ICML
  2023) has a changepoint operator in its grammar and moves it by Monte
  Carlo; I found no paper that seeds a kernel search with a detector's
  dates.

## What follows

1. **A detector as a proposer.** Run the line-and-harmonics detector
   (period from the periodogram, as `--spectral` finds it) and pass its
   dates to the search as dated proposals, `CP(C, C, t)`, through the
   `Proposer` interface without a language model. Measure it as the
   proposals were: grammar, detector, language model told the truth, on
   both families; the quarterly family measures what false dates cost.
2. **A last look at the smooth components.** After the search, read each
   smooth component with the cycle inside the cost; a break found there
   becomes a dated proposal for one more stage, as the event stage does
   with a context's date.
3. **Changes in spread and in a cycle's size.** A kernel or variance cost
   on the residual and on a periodic component could propose `CW` windows.
   Later, once 1 and 2 are measured.
4. **Not worth doing.** A constant-mean detector on the raw values or on
   STL components: it over-segments; the search would fit dozens of
   false dates.

The pilot has one easy jump of eight noise units. Before any claim, the
measurement needs smaller jumps, several breaks, changes of slope and
real series.

## Step 1 measured: the detector as a proposer

`autostat.breaks` runs the line-and-harmonics detector (PELT, cycle from
the periodogram) and passes at most `break_starts` dates to the search.
At the first level it proposes its own model, `Lin + Per + WN` plus one
`CP(Zero, C)` shift per date, started from one regression: slope, cycle
amplitude, shift sizes and noise as the data give them, the switch steep.
At every level it adds a shift at each date the best model does not
already hold. BIC decides; nothing is forced. From random starts the same
structure ended 400 to 600 BIC; from the regression's values it ends
near 400 at once.

`run.py` searches both families with three arms, five seeds, three
levels, four random starts and 100 iterations. `spectral` adds the
periodogram's peak alone, which separates what the cycle brings from
what the dates bring. BIC of the best model, lower is better:

| Family | Arm | Seeds 0 to 4 | Median | Jump kept |
|---|---|---|---|---|
| jump | grammar | 494.0, 474.8, 489.3, 498.1, 485.6 | 489.3 | 0 / 5 |
| jump | spectral | 480.3, 473.6, 489.3, 454.8, 485.6 | 480.3 | 1 / 5 |
| jump | breaks | 483.2, 471.7, 406.7, 431.0, 402.8 | 431.0 | 5 / 5 |
| quarterly | grammar | 389.1, 383.3, 412.6, 417.4, 422.3 | 412.6 | |
| quarterly | spectral | 389.1, 381.0, 397.8, 433.7, 408.7 | 397.8 | |
| quarterly | breaks | 389.1, 383.3, 412.6, 417.4, 422.3 | 412.6 | |

- **The jump.** With the dates, the search keeps a shift at the jump on
  every seed, 0.005 to 0.08 years from it, sized 6.5 to 7.8 against 8.
  Three seeds of five end on the true structure, `CP(Zero, C) + Lin +
  Per + WN`, 67 to 83 BIC below the grammar, with a band on the jump's
  size about 1.3 wide where the grammar's smooth step had 6 to 9. On
  the other two the cycle is missed (`SE` in its place), the band stays
  7 to 9 wide and the gain is 3 to 11. The periodogram's peak alone
  kept the jump once of five.
- **No false break.** On the quarterly family the detector finds no
  date; the arm is the grammar, number for number.
- **Cost.** The jump family's searches took 4.6 times as long (19
  minutes against 4 on average), on a machine shared with another
  benchmark; each shift adds a model per date and level.

`real.py` runs the same detector on the thirteen series of the 2014
reports and compares its dates with the switches of the published
models, as a share of each series' span:

| Series | Detector | Published switch | Distance / span |
|---|---|---|---|
| solar | 1714.00, 1779.00, 1644.00 | 1642.87, 1715.97 | 0.005, 0.157, 0.003 |
| call-centre | 1974.13, 1968.96, 1973.37 | 1974.10, 1973.36, 1973.75 | 0.002, 0.295, 0.001 |
| radio | 1945.63, 1950.21, 1947.37 | 1942.02, 1945.53 | 0.005, 0.235, 0.093 |
| unemployment | 1970.37, 1962.88, 1950.46 | 1962.35, 1966.60 | 0.111, 0.015, 0.351 |
| sulphuric | 1989.29, 1976.29, 1980.29 | 1964.30, 1990.63, 1969.59, 1977.21 | 0.035, 0.024, 0.080 |
| gas-production | 1969.46, 1993.13, 1994.21 | 1967.29, 1971.13 | 0.042, 0.556, 0.583 |
| internet | 2005.03, 2005.05, 2004.95 | 2004.98, 2005.01 | 0.072, 0.175, 0.157 |
| wheat | 1817.50, 1809.50, 1845.50 | 1741.83, 1775.12 | 0.115, 0.093, 0.191 |
| wages | 1972.50, 1912.50, 1920.50 | 1604.65, 1704.63 | 0.365, 0.283, 0.294 |

On solar, call-centre and radio the strongest date falls within half a
percent of the span from a published switch; the window of solar is
found at both ends. Wheat and wages agree on nothing: their published
switches sit in the first part of a long series, where the detector,
cutting one line and one cycle, sees a slow bend rather than a break.
On internet the cycle found is one day; the dates crowd one week.

On airline, Mauna Loa, temperature and births the published model has
no switch and the detector proposes one to three dates; BIC then decides
whether a shift pays. The flag is `--breaks N` (`break_starts` in an
`Experiment`); the presets leave it off, as published.
