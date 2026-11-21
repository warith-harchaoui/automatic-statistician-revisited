# Why does the port forecast better than the 2014 system?

On the extrapolation benchmark, the port searching five levels scores a
median standardised RMSE of 1.51 where the forecasts the 2014 system
published score 2.12. The grammar, the starts and the score are the same.
Three explanations were open: the port's optimiser (L-BFGS in `torch`,
where GPML used conjugate gradients), a better search, or other structures
that extrapolate better.

The 2014 run kept every model of its ten levels. The thirteen
`*_result.txt` files of `results/2013-12-10-extrap-GPSS` in
[gpss-research](https://github.com/jamesrobertlloyd/gpss-research) (commit
`2a64958`) are copied to `benchmark/published/searches_2014/`.
[run.py](run.py) scores three models on each series, all in the port's
code and on the same training part; [results.json](results.json)
holds every row. A few minutes in all.

| series | BIC 2014 | refitted | port | RMSE 2014 | refitted | port |
|---|---|---|---|---|---|---|
| airline | 999.4 | 999.4 | 996.9 | 30.70 | 30.75 | **15.05** |
| solar | **−823.0** | −823.0 | −777.5 | 0.482 | 0.482 | **0.450** |
| mauna | 257.3 | 257.3 | **248.4** | 1.373 | 1.373 | **1.125** |
| wheat | 2738.9 | 2738.9 | **2736.5** | **50.5** | 50.5 | 52.9 |
| temperature | **4339.6** | 4339.6 | 4349.2 | **2.755** | 2.755 | 2.763 |
| internet | **17485.0** | 17485.0 | 17564.0 | 10453 | 10453 | **9086** |
| call centre | **1490.2** | 1490.0 | 1531.1 | 13.6 | **13.5** | 37.2 |
| radio | 487.0 | 487.0 | 486.1 | 1.208 | 1.201 | **1.061** |
| gas | **6589.3** | 6589.3 | 6636.2 | 6217 | 6217 | **5982** |
| sulphuric | **3463.5** | 3463.5 | 3474.3 | **42.0** | 42.0 | 44.4 |
| unemployment | 3966.8 | 3966.8 | **3959.6** | 250.9 | 250.9 | **101.8** |
| births | 8470.6 | 8460.6 | **8322.7** | **27.5** | 44.1 | 41.7 |
| wages | **1067.6** | 1067.6 | 1078.8 | 19.65 | 19.65 | **16.23** |

"2014" is the 2014 winner with its own parameters, "refitted" the same
structure refitted by the port from those parameters, "port" the port's
depth-5 winner. BIC and RMSE: lower is better.

- **The port reproduces the 2014 forecasts.** Given the 2014 model, the
  port's likelihood gives the BIC the 2014 file recorded (to 0.001) and its
  posterior mean differs from the published forecast by at most 0.0003 on
  every series; the RMSEs agree to the last digit shown.
- **The optimiser explains nothing.** Refitting the 2014 structures with
  L-BFGS lowers BIC on two series only, call centre (0.2) and births
  (10.0). The 2014 optimiser had converged.
- **The port does not win by finding lower BIC.** Its winner has the lower
  BIC on 6 series and the higher on 7, by up to 79 (internet) and 47 (gas).
  Its forecasts are better on 8 series, 4 of them among those where its
  BIC is higher: solar, internet, gas and wages.
- **BIC on the training part does not rank extrapolations.** Where the
  port's BIC is lower its forecast is better on 4 series of 6; where it is
  higher, on 4 of 7. On births a refit 10 BIC better forecasts worse (RMSE
  44.1 against 27.5).

The port's gain therefore comes from the structures its five levels keep,
which are often simpler than the ten-level winners (temperature
`C + Per + SE` against `C + Per + SE + SE`; unemployment adds a `Lin` trend
2014 did not have). BIC chooses what explains the training part; what
continues it best is another question, which this benchmark answers only
series by series. A selection score built for extrapolation (a
rolling-origin forecast error on the training part, as `tuning.py` does for
LightGBM) is the next measurement.
