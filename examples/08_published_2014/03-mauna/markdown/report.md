---
author: "Some Math empowered by some AI guided by some researchers"
---

# An automatic report for the dataset: 03-mauna

*Produced by `autostat` (automatic-statistician-revisited), a Python 3 port of the Automatic Bayesian Covariance Discovery system of Lloyd et al. (AAAI 2014).*

![The data](data.svg)

## Executive summary

The structure search selected `WN + Lin x SE + Per x SE + Per x SE + SE` (Bayesian information criterion 307.1; negative log likelihood 109.4 on 545 points). It decomposes into 5 additive components, listed by how much each reduces extrapolation error:

1. A very smooth monotonically increasing function with marginal standard deviation increasing linearly away from 1982.
2. An approximately periodic function with a period of 1.0 years.
3. A smooth function.
4. Uncorrelated noise.
5. A very approximately periodic function with a period of 5.8 years.

| # | R² cumulative (%) | Residual variance explained (%) | Cross-validated MAE | MAE reduction (%) |
|---|---|---|---|---|
| data | | | 15.9 | |
| 1 | 98.5 | 98.5 | 1.9 | 88.0 |
| 2 | 99.8 | 88.7 | 0.773 | 59.3 |
| 3 | 100.0 | 88.7 | 0.52 | 32.8 |
| 4 | 100.0 | 71.5 | 0.52 | 0.0 |
| 5 | 100.0 | 100.0 | 0.522 | -0.3 |

MAE: mean absolute error over ten contiguous held-out blocks.

## The formula found

The structure search is a symbolic regression: it searches a grammar of expressions over kernels and keeps the one with the lowest Bayesian information criterion. This is the expression it kept, with every fitted parameter in place; $t$ is in years.

$$y(t) = f_{1}(t) + f_{2}(t) + f_{3}(t) + f_{4}(t) + f_{5}(t)$$

Each component is a Gaussian process with its own covariance:

$$f_{1}(t) \sim \mathrm{GP}\!\left(0,\ 15.22\,(t - 1983)(t' - 1983) \cdot \exp\!\left(-\frac{(t - t')^2}{2 \cdot 484^2}\right)\right)$$

$$f_{2}(t) \sim \mathrm{GP}\!\left(0,\ 2.617\,\frac{\exp\!\left(0.5434\cos\frac{2\pi (t - t')}{0.9998}\right) - I_0(0.5434)}{e^{0.5434} - I_0(0.5434)} \cdot \exp\!\left(-\frac{(t - t')^2}{2 \cdot 95.94^2}\right)\right)$$

$$f_{3}(t) \sim \mathrm{GP}\!\left(0,\ 0.6688\,\exp\!\left(-\frac{(t - t')^2}{2 \cdot 1.239^2}\right)\right)$$

$$f_{4}(t) \sim \mathrm{GP}\!\left(0,\ 0.04121\,\delta_{t t'}\right)$$

$$f_{5}(t) \sim \mathrm{GP}\!\left(0,\ 0.0356\,\frac{\exp\!\left(25.92\cos\frac{2\pi (t - t')}{5.847}\right) - I_0(25.92)}{e^{25.92} - I_0(25.92)} \cdot \exp\!\left(-\frac{(t - t')^2}{2 \cdot 3.771^2}\right)\right)$$

These components have a closed form, read from their posterior mean:

$$f_{1}(t) \approx u \left(0.01054 u + 0.8748\right) - 28.35,\quad u = t - 1958$$

A periodic kernel is exactly a sum of cosines, so a periodic component is a Fourier series in its period; it is written up to the harmonic after which the rest weighs less than 0.1 % of the component, hence the approximate sign.

The other components are smooth, quasi-periodic or noise without a short formula: a draw from such a process has no finite closed form; its covariance above is the formula.

Closed form of each component:

| # | component | closed form |
|---|---|---|
| 1 | A very smooth monotonically increasing function with marginal standard deviation increasing linearly away from 1982 | symbolic regression (PySR) on the posterior mean, largest gap 0.083 % of the component |
| 2 | An approximately periodic function with a period of 1.0 years | none: the covariance is the formula |
| 3 | A smooth function | none: the covariance is the formula |
| 4 | Uncorrelated noise | none: the covariance is the formula |
| 5 | A very approximately periodic function with a period of 5.8 years | none: the covariance is the formula |

## Search path and stopping criterion

The 2014 configuration searches a fixed number of levels and keeps the lowest BIC; it has no stopping rule. The table follows the winner of each level. A change of structure adds or removes terms; a level that keeps the structure only refits its parameters. BIC differences read on the scale of Kass and Raftery (1995): under 2 barely worth mentioning, 2 to 6 positive, 6 to 10 strong, over 10 very strong.

| level | BIC | change | structure | components | closed | params |
|---|---|---|---|---|---|---|
| 1 | 2424.4 |  | WN + SE | 2 | 0 | 3 |
| 2 | 893.3 | -1531.1 | WN + Per x SE + SE | 3 | 0 | 7 |
| 3 | 496.2 | -397.1 | WN + Per x Per x SE + Per x SE + Per x SE + SE | 5 | 0 | 17 |
| 4 | 371.2 | -125.0 | Lin + WN + Per x Per x SE + Per x SE + SE | 5 | 1 | 15 |
| 5 | 353.3 | -17.9 | Lin + WN + Per x SE + Per x SE + SE | 5 | 1 | 13 |
| 6 | 345.6 | -7.7 | Lin + WN + Per x SE + Per x SE + SE | 5 | 1 | 13 |
| 7 | 342.6 | -3.0 | WN + Lin x SE + Per x SE + Per x SE + SE | 5 | 0 | 14 |
| 8 | 334.1 | -8.4 | WN + Lin x SE + Per x SE + Per x SE + SE | 5 | 0 | 14 |
| 9 | 313.7 | -20.4 | WN + Lin x SE + Per x SE + Per x SE + SE | 5 | 0 | 14 |
| 10 (kept) | 307.1 | -6.6 | WN + Lin x SE + Per x SE + Per x SE + SE | 5 | 0 | 14 |

![BIC per search level](bic.svg)

Posterior weight of each structure met, exp(-BIC/2) normalised under a flat prior, after the Bayesian machine scientist (Guimerà et al., 2020): where the search keeps one structure, the weights say how much the data prefer it to the others. The 5 structure(s) under 0.001 are left out.

| structure | lowest BIC | weight |
|---|---|---|
| WN + Lin x SE + Per x SE + Per x SE + SE | 307.1 | 1.000 |

- The search ran 10 levels (the published 2014 search: ten levels, no stopping rule) and keeps the model of level 10, WN + Lin x SE + Per x SE + Per x SE + SE, BIC 307.1.
- That structure was first reached at level 7; the levels after it only refitted its parameters, lowering BIC by 35.5 in all.
- The best different structure, Lin + WN + Per x SE + Per x SE + SE (level 6), is 38.5 above it: very strong evidence for the kept one on the Kass and Raftery scale.
- Weighed by exp(-BIC/2) under a flat prior, as the Bayesian machine scientist weighs expressions, the structures met give the kept one a posterior weight of 1.000 and the next, Lin + WN + Per x SE + Per x SE + SE, 0.000: the kept one takes almost all the weight.
- The last level lowered BIC by 6.6 through its parameters only: the structure had settled, its fit had not quite.
- A rule stopping after 1 level(s) without a fall above 0.1 would not have stopped within these 10 levels.
- A rule stopping after 2 level(s) without a fall above 0.1 would not have stopped within these 10 levels.
- The kept expression has 14 parameters for 545 points: its BIC is a fit term 2 NLL = 218.9 plus a complexity term p ln n = 88.2, each extra parameter costing 6.30; it splits into 5 components, 0 of them with a closed form.

![Data and full posterior](fit.svg)

## Detailed discussion of the components

### Component 1: A very smooth monotonically increasing function with marginal standard deviation increasing linearly away from 1982

This component is a very smooth and monotonically increasing function. The marginal standard deviation of the function increases linearly away from 1982.

This component explains 98.5% of the data's variance on its own; with the components above it, 98.5%. Adding it changes the cross-validated MAE by -88.0%.

*Extrapolation.* This component is assumed to continue very smoothly but is also assumed to be stationary so its distribution will eventually return to the prior. The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of 484.0 years. The marginal standard deviation of the function is assumed to continue to increase linearly.

![Component 1](component_1.svg)

### Component 2: An approximately periodic function with a period of 1.0 years

This component is approximately periodic with a period of 1.0 years. Across periods the shape of this function varies very smoothly. The shape of this function within each period has a typical lengthscale of 2.6 months.

This component explains 1.1% of the data's variance on its own; with the components above it, 99.8%. Adding it changes the cross-validated MAE by -59.3%.

*Extrapolation.* This component is assumed to continue to be approximately periodic. The shape of the function is assumed to vary very smoothly between periods but will eventually return to the prior. The prior is entirely uncertain about the phase of the periodic function. Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the function.

![Component 2](component_2.svg)

### Component 3: A smooth function

This component is a smooth function with a typical lengthscale of 1.2 years.

This component explains 0.3% of the data's variance on its own; with the components above it, 100.0%. Adding it changes the cross-validated MAE by -32.8%.

*Extrapolation.* This component is assumed to continue smoothly but is also assumed to be stationary so its distribution will return to the prior. The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of 1.2 years.

![Component 3](component_3.svg)

### Component 4: Uncorrelated noise

This component models uncorrelated noise.

This component explains 0.0% of the data's variance on its own; with the components above it, 100.0%. Adding it changes the cross-validated MAE by -0.0%.

*Extrapolation.* This component assumes the uncorrelated noise will continue indefinitely.

![Component 4](component_4.svg)

### Component 5: A very approximately periodic function with a period of 5.8 years

This component is very approximately periodic with a period of 5.8 years. Across periods the shape of this function varies smoothly with a typical lengthscale of 3.8 years. Since this lengthscale is small relative to the period this component may more closely resemble a non-periodic smooth function.

This component explains 0.0% of the data's variance on its own; with the components above it, 100.0%. Adding it changes the cross-validated MAE by 0.3%.

*Extrapolation.* This component is assumed to continue to be approximately periodic. The shape of the function is assumed to vary smoothly between periods but will quickly return to the prior. The prior is entirely uncertain about the phase of the periodic function. Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the function.

![Component 5](component_5.svg)

## Model checking

Each statistic compares draws of a component's posterior with draws of its prior. A small p-value (bold below 0.05) points at a property of the data that the component's prior does not reproduce.

| # | QQ max | QQ min | ACF min | ACF min lag | Periodogram max | Periodogram peak | MMD |
|---|---|---|---|---|---|---|---|
| 1 | 0.536 | 0.608 | 0.336 | 0.394 | 0.546 | 0.964 | **0.000** |
| 2 | 0.092 | 0.248 | 0.782 | 0.878 | 0.368 | 0.814 | **0.000** |
| 3 | 0.592 | 0.590 | 0.548 | 0.916 | 0.488 | 0.588 | **0.000** |
| 4 | 0.878 | 0.912 | 0.966 | 0.954 | 0.998 | 1.000 | 0.330 |
| 5 | 0.886 | 0.850 | 0.982 | 0.960 | 0.896 | 0.928 | 0.417 |

QQ: quantile-quantile gaps. ACF: autocorrelation function. MMD: maximum mean discrepancy two-sample test.
