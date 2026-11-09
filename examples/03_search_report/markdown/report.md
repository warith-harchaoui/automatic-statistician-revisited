---
author: "Some Math empowered by some AI guided by some researchers"
---

# An automatic report for the dataset: airline passengers

*Produced by `autostat` (automatic-statistician-revisited), a Python 3 port of the Automatic Bayesian Covariance Discovery system of Lloyd et al. (AAAI 2014).*

![The data](data.svg)

## Executive summary

The structure search selected `Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE` (Bayesian information criterion 1108.9; negative log likelihood 522.2 on 144 points). It decomposes into 4 additive components, listed by how much each reduces extrapolation error:

1. A quadratic polynomial.
2. An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude.
3. A smooth function.
4. Uncorrelated noise with quadraticly varying standard deviation.

| # | R² cumulative (%) | Residual variance explained (%) | Cross-validated MAE | MAE reduction (%) |
|---|---|---|---|---|
| data | | | 280 | |
| 1 | 86.2 | 86.2 | 32.3 | 88.5 |
| 2 | 99.2 | 94.4 | 9.06 | 71.9 |
| 3 | 99.7 | 61.1 | 7.03 | 22.4 |
| 4 | 100.0 | 100.0 | 7.03 | 0.0 |

MAE: mean absolute error over ten contiguous held-out blocks.

## The formula found

The structure search is a symbolic regression: it searches a grammar of expressions over kernels and keeps the one with the lowest Bayesian information criterion. This is the expression it kept, with every fitted parameter in place; $t$ is in years.

$$y(t) = f_{1}(t) + f_{2}(t) + f_{3}(t) + f_{4}(t)$$

Each component is a Gaussian process with its own covariance:

$$f_{1}(t) \sim \mathrm{GP}\!\left(0,\ 0.8789\,(t - 1938)(t' - 1938) \cdot (t - 1938)(t' - 1938)\right)$$

$$f_{2}(t) \sim \mathrm{GP}\!\left(0,\ 15.84\,(t - 1945)(t' - 1945) \cdot \frac{\exp\!\left(2.296\cos\frac{2\pi (t - t')}{1.003}\right) - I_0(2.296)}{e^{2.296} - I_0(2.296)} \cdot \exp\!\left(-\frac{(t - t')^2}{2 \cdot 17^2}\right)\right)$$

$$f_{3}(t) \sim \mathrm{GP}\!\left(0,\ 74.28\,\exp\!\left(-\frac{(t - t')^2}{2 \cdot 0.505^2}\right)\right)$$

$$f_{4}(t) \sim \mathrm{GP}\!\left(0,\ 0.282\,(t - 1957)(t' - 1957) \cdot (t - 1950)(t' - 1950) \cdot \delta_{t t'}\right)$$

These components have a closed form, read from their posterior mean:

$$f_{1}(t) = 112.3 + 20.52\,(t - 1949) + 0.9375\,(t - 1949)^{2}$$

The other components are smooth, quasi-periodic or noise without a short formula: a draw from such a process has no finite closed form; its covariance above is the formula.

Closed form of each component:

| # | component | closed form |
|---|---|---|
| 1 | A quadratic polynomial | exact polynomial of degree 2 |
| 2 | An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude | none: the covariance is the formula |
| 3 | A smooth function | none: the covariance is the formula |
| 4 | Uncorrelated noise with quadraticly varying standard deviation | none: the covariance is the formula |

## Search path and stopping criterion

The 2014 configuration searches a fixed number of levels and keeps the lowest BIC; it has no stopping rule. The table follows the winner of each level. A change of structure adds or removes terms; a level that keeps the structure only refits its parameters. BIC differences read on the scale of Kass and Raftery (1995): under 2 barely worth mentioning, 2 to 6 positive, 6 to 10 strong, over 10 very strong.

| level | BIC | change | structure | components | closed | params |
|---|---|---|---|---|---|---|
| 1 | 1513.7 |  | SE + WN | 2 | 0 | 3 |
| 2 | 1236.4 | -277.3 | Per x SE + SE + WN | 3 | 0 | 7 |
| 3 | 1151.5 | -84.9 | Lin x Per x SE + Lin x SE + Per x SE + SE + WN | 5 | 0 | 15 |
| 4 | 1133.1 | -18.4 | Lin x Lin x Per x SE + Lin x Per x SE + Lin x SE + SE + WN | 5 | 0 | 17 |
| 5 | 1119.0 | -14.1 | Lin x Lin x Per + Lin x Lin x Per x SE + Lin x SE + SE + WN | 5 | 0 | 17 |
| 6 | 1118.5 | -0.5 | Lin x Lin x Per x SE + Lin x Lin x WN + Lin x SE + SE + WN | 5 | 0 | 15 |
| 7 | 1111.6 | -7.0 | Lin x Lin + Lin x Lin x Per x SE + Lin x Lin x WN + SE + WN | 5 | 1 | 15 |
| 8 | 1109.8 | -1.8 | Lin x Lin + Lin x Lin x Per x SE + Lin x Lin x WN + SE | 4 | 1 | 14 |
| 9 | 1108.9 | -0.8 | Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE | 4 | 1 | 13 |
| 10 | 1108.9 | -0.0 | Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE | 4 | 1 | 13 |
| 11 (kept) | 1108.9 | -0.0 | Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE | 4 | 1 | 13 |

![BIC per search level](bic.svg)

Posterior weight of each structure met, exp(-BIC/2) normalised under a flat prior, after the Bayesian machine scientist (Guimerà et al., 2020): where the search keeps one structure, the weights say how much the data prefer it to the others. The 4 structure(s) under 0.001 are left out.

| structure | lowest BIC | weight |
|---|---|---|
| Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE | 1108.9 | 0.516 |
| Lin x Lin + Lin x Lin x Per x SE + Lin x Lin x WN + SE | 1109.8 | 0.338 |
| Lin x Lin + Lin x Lin x Per x SE + Lin x Lin x WN + SE + WN | 1111.6 | 0.138 |
| Lin x Lin x Per x SE + Lin x Lin x WN + Lin x SE + SE + WN | 1118.5 | 0.004 |
| Lin x Lin x Per + Lin x Lin x Per x SE + Lin x SE + SE + WN | 1119.0 | 0.003 |

- The search ran 11 levels (stopping after 2 level(s) without a BIC fall above 0.1, at most 20) and keeps the model of level 11, Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE, BIC 1108.9.
- That structure was first reached at level 9; the levels after it only refitted its parameters, lowering BIC by 0.0 in all.
- The best different structure, Lin x Lin + Lin x Lin x Per x SE + Lin x Lin x WN + SE (level 8), is 0.8 above it: barely worth mentioning evidence for the kept one on the Kass and Raftery scale.
- Weighed by exp(-BIC/2) under a flat prior, as the Bayesian machine scientist weighs expressions, the structures met give the kept one a posterior weight of 0.516 and the next, Lin x Lin + Lin x Lin x Per x SE + Lin x Lin x WN + SE, 0.338: the data barely tell them apart.
- BIC stopped falling at level 9. The 2 level(s) searched past it kept its structure and lowered BIC by 0.00 in all, less than the tolerance of 0.1: going on buys nothing the data pay for.
- A rule stopping after 1 level(s) without a fall above 0.1 would have ended the search at level 10 and kept the same structure, less refitted.
- A rule stopping after 2 level(s) without a fall above 0.1 would have ended the search at level 11 and kept the same model.
- The kept expression has 13 parameters for 144 points: its BIC is a fit term 2 NLL = 1044.3 plus a complexity term p ln n = 64.6, each extra parameter costing 4.97; it splits into 4 components, 1 of them with a closed form.

![Data and full posterior](fit.svg)

## Detailed discussion of the components

### Component 1: A quadratic polynomial

This component is a quadratic polynomial.

This component explains 86.2% of the data's variance on its own; with the components above it, 86.2%. Adding it changes the cross-validated MAE by -88.5%.

*Extrapolation.* This component is assumed to continue as a quadratic polynomial.

![Component 1](component_1.svg)

### Component 2: An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude

This component is approximately periodic with a period of 1.0 years and varying amplitude. Across periods the shape of this function varies very smoothly. The amplitude of the function increases linearly. The shape of this function within each period has a typical lengthscale of 5.5 weeks.

This component explains 13.2% of the data's variance on its own; with the components above it, 99.2%. Adding it changes the cross-validated MAE by -71.9%.

*Extrapolation.* This component is assumed to continue to be approximately periodic. The shape of the function is assumed to vary very smoothly between periods but will eventually return to the prior. The amplitude of the function is assumed to continue to increase linearly. The prior is entirely uncertain about the phase of the periodic function. Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the function.

![Component 2](component_2.svg)

### Component 3: A smooth function

This component is a smooth function with a typical lengthscale of 6.1 months.

This component explains 1.0% of the data's variance on its own; with the components above it, 99.7%. Adding it changes the cross-validated MAE by -22.4%.

*Extrapolation.* This component is assumed to continue smoothly but is also assumed to be stationary so its distribution will return to the prior. The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of 6.1 months.

![Component 3](component_3.svg)

### Component 4: Uncorrelated noise with quadraticly varying standard deviation

This component models uncorrelated noise. The standard deviation of the noise varies quadraticly.

This component explains 0.3% of the data's variance on its own; with the components above it, 100.0%. Adding it changes the cross-validated MAE by -0.0%.

*Extrapolation.* This component assumes the uncorrelated noise will continue indefinitely. The standard deviation of the noise is assumed to continue to vary quadraticly.

![Component 4](component_4.svg)

## Model checking

Each statistic compares draws of a component's posterior with draws of its prior. A small p-value (bold below 0.05) points at a property of the data that the component's prior does not reproduce.

| # | QQ max | QQ min | ACF min | ACF min lag | Periodogram max | Periodogram peak | MMD |
|---|---|---|---|---|---|---|---|
| 1 | 0.336 | 0.334 | 0.982 | 0.988 | 0.626 | 0.974 | **0.000** |
| 2 | 0.878 | 0.054 | 0.832 | 0.770 | 0.698 | 0.600 | 0.050 |
| 3 | 0.724 | 0.744 | 0.542 | 0.506 | 0.750 | 0.622 | 0.129 |
| 4 | 0.882 | 0.838 | 0.956 | 0.972 | 0.952 | 0.950 | 0.326 |

QQ: quantile-quantile gaps. ACF: autocorrelation function. MMD: maximum mean discrepancy two-sample test.
