# Extrapolation benchmark results

Paper table reproduced from the published 2014 predictions to within 2.2e-10.

Standardised RMSE: each series' RMSE divided by the best RMSE on that
series among every method listed, so 1.00 is the best. Median over the
13 series; lower is better.

MASE and MdASE: mean and median absolute error of each series scaled
by the mean absolute one-step change of its training part; MASE is
their median (and mean MASE their mean) across series; lower is better.

MAPE and MdAPE: mean and median absolute percentage error of each
series, in per cent, then their median (MAPE, MdAPE) and mean (mean
MAPE) across series; lower is better. A zero actual value has no
percentage error and is left out.

| method | family | median | mean | series won | 90 % coverage | MASE | mean MASE | MdASE | MAPE | mean MAPE | MdAPE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chronos2 | foundation | 1.47 | 1.65 | 1 | 0.80 | 1.04 | 3.87 | 0.81 | 8.9 | 19.5 | 8.2 |
| chronos_bolt_base | foundation | 1.53 | 1.70 | 1 | 0.72 | 1.10 | 3.52 | 0.83 | 9.5 | 18.9 | 7.7 |
| timestral_209m | foundation | 1.57 | 1.65 | 1 | 0.86 | 0.97 | 3.87 | 0.78 | 8.6 | 18.7 | 8.0 |
| kumo_lags | foundation | 1.68 | 2.50 | 2 |  | 2.02 | 4.05 | 1.53 | 9.4 | 30.7 | 8.1 |
| timesfm_2p5 | foundation | 1.72 | 2.17 | 0 | 0.67 | 2.45 | 4.62 | 2.35 | 13.7 | 21.4 | 11.6 |
| auto_arima | classical | 1.79 | 3.22 | 0 | 0.76 | 2.46 | 4.56 | 1.76 | 14.8 | 27.6 | 13.1 |
| nhits | deep learning | 1.81 | 2.09 | 1 |  | 1.76 | 2.99 | 1.75 | 14.3 | 22.8 | 11.3 |
| TCI | 2014 baselines | 1.88 | 2.27 | 1 |  | 1.52 | 4.38 | 1.39 | 14.9 | 21.8 | 11.2 |
| auto_theta | classical | 1.89 | 2.21 | 0 | 0.87 | 1.55 | 3.85 | 1.44 | 15.5 | 26.7 | 14.5 |
| nbeats | deep learning | 1.91 | 1.96 | 0 |  | 1.39 | 2.93 | 1.15 | 15.1 | 20.7 | 13.0 |
| auto_ets | classical | 1.96 | 2.45 | 0 | 0.85 | 2.29 | 4.10 | 1.62 | 15.2 | 27.6 | 11.8 |
| lightgbm_lags | machine learning | 2.08 | 4.38 | 1 |  | 2.59 | 4.85 | 2.26 | 22.5 | 42.4 | 22.3 |
| seasonal_naive | classical | 2.08 | 3.04 | 0 | 0.83 | 3.34 | 4.26 | 3.01 | 19.0 | 24.8 | 18.0 |
| ABCD-2014 | structure search (2014 published) | 2.12 | 2.05 | 1 |  | 1.46 | 4.90 | 1.19 | 17.9 | 19.0 | 14.1 |
| SP | 2014 baselines | 2.19 | 3.22 | 0 |  | 2.72 | 5.23 | 2.36 | 21.6 | 26.2 | 15.1 |
| naive | classical | 2.35 | 3.14 | 0 | 0.88 | 2.53 | 4.58 | 2.33 | 21.5 | 29.7 | 16.7 |
| CKS-2013 | structure search (2014 published) | 2.48 | 2.17 | 2 |  | 1.25 | 4.40 | 1.19 | 13.8 | 17.8 | 12.4 |
| Eureqa | symbolic regression (2014 published) | 2.82 | 48.11 | 1 |  | 3.26 | 22.39 | 3.39 | 16.3 | 205.7 | 14.7 |
| CP | 2014 baselines | 2.83 | 3.19 | 0 |  | 2.28 | 7.09 | 2.09 | 17.8 | 30.3 | 17.6 |
| pysr | symbolic regression | 2.87 | 2.76 | 0 |  | 2.30 | 4.79 | 2.18 | 18.7 | 26.5 | 15.5 |
| MKL | 2014 baselines | 2.92 | 2.92 | 0 |  | 2.15 | 4.94 | 2.09 | 17.6 | 26.2 | 17.4 |
| SE | 2014 baselines | 3.20 | 8.30 | 1 |  | 5.03 | 9.57 | 5.17 | 46.2 | 48.6 | 38.1 |
| LN | 2014 baselines | 3.20 | 5.83 | 0 |  | 3.25 | 7.41 | 2.39 | 16.7 | 45.2 | 16.1 |
| gplearn | symbolic regression | 3.60 | 8.84 | 0 |  | 3.07 | 7.71 | 2.77 | 29.3 | 54.4 | 22.3 |
