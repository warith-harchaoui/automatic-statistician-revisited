# Why are the port's 90 % intervals too narrow?

On the extrapolation benchmark the port's 90 % intervals hold 72 % of the
held-out tails, Chronos-2's 80 %. [run.py](run.py) reads the depth-5
searches already kept in `benchmark/results/searches/abcd_2014-depth5`
(nothing is searched again), diagnoses the misses and measures two
remedies; [results.json](results.json) holds every row. About an hour.

Each forecast is scored by its coverage and by the interval score of
Gneiting and Raftery (2007) for a 90 % band, width plus 20 times the
distance by which a value falls outside, divided by the mean absolute
one-step change of the training part. Lower is better: a band wins by
being narrow and by holding the values.

## Diagnosis: off-centre more than narrow

The standardised error of the tail, `z = (y - mean) / sd`, has mean 0 and
spread 1 when the forecast is calibrated.

| series | mean of z | spread of z | coverage |
|---|---|---|---|
| airline | 0.73 | 0.91 | 0.80 |
| solar | 0.93 | 1.48 | 0.66 |
| mauna | 0.95 | 0.91 | 0.78 |
| wheat | 0.38 | 0.89 | 0.84 |
| temperature | 0.13 | 0.97 | 0.91 |
| internet | −0.11 | 1.02 | 0.91 |
| call centre | −0.19 | 0.45 | 1.00 |
| radio | 0.42 | 0.58 | 1.00 |
| gas | 0.86 | 0.74 | 0.81 |
| sulphuric | −1.82 | 0.62 | 0.43 |
| unemployment | 0.71 | 0.62 | 0.93 |
| births | 1.12 | 1.07 | 0.70 |
| wages | 3.79 | 1.74 | 0.04 |

- **The bands are about as wide as the scatter they must hold.** The
  median spread of z is 0.91; on 11 series of 13 it is at most 1.07.
- **The misses come from forecasts off-centre.** The mean of z exceeds 0.5
  in size on 8 series. Wages (3.79) and sulphuric (−1.82) carry most of
  the misses: their extrapolated trend leaves the data. The median series
  holds 81 %; the pooled 72 % is weighted by the long series.
- **Ten series of thirteen lie above the forecast.** The mean of z is
  positive on 10: the port's forecasts tend to fall short of the tail.

## Two remedies

| forecast | pooled coverage | median coverage | median interval score | interval score better / worse than the best model | median RMSE ratio |
|---|---|---|---|---|---|
| best model | 0.72 | 0.81 | 8.49 | | 1 |
| best model, widened on the training part | 0.87 | 0.93 | 9.86 | 5 / 5 | 1 |
| mixture, T = 1 (the benchmark's) | 0.73 | 0.81 | 8.67 | 5 / 7 | 1.000 |
| mixture, T = 2 | 0.74 | 0.83 | 8.85 | 6 / 6 | 1.000 |
| mixture, T = 5 | 0.81 | 0.83 | 8.92 | 8 / 5 | 0.999 |
| mixture, T = 10 | 0.84 | 0.92 | 8.90 | 8 / 5 | 0.975 |

- **Widening on the training part** refits the kept structure on the first
  90 % of the training part, forecasts the last 10 % and widens the band
  by the 90 % quantile of `|z|` there, over 1.645. No held-out value is
  read. It lifts coverage to 87 % and halves the interval score of wages,
  but the factor is unstable: 9.8 on call centre, whose interval score
  climbs from 11 to 109. It helps on 5 series and hurts on 5.
- **Tempering the mixture** raises the BIC weights to `1 / T`, so more
  structures share the forecast. At T = 10 coverage reaches 84 %, the
  interval score improves on 8 series of 13 and the median RMSE falls by
  2.5 %. It cuts the error on call centre, the port's worst loss to 2014,
  by 65 %, on wheat and radio by 15 %; it adds 37 % on airline. T was not chosen on the
  training part here, so these rows are a sensitivity, not a method.

Neither remedy is a clear win; the benchmark keeps the best model and the
untempered mixture. Since the misses are bias, the next step is the
selection score, not the band: a structure chosen for how it extrapolates
on the training part (see `doc/research/optimiser`). A temperature chosen by
the same rolling-origin forecast would make the tempered mixture a method.
