---
author: "Some Math empowered by some AI guided by some researchers"
---

# An automatic report for the dataset: 08-radio

*Produced by `autostat` (automatic-statistician-revisited), a Python 3 port of the Automatic Bayesian Covariance Discovery system of Lloyd et al. (AAAI 2014).*

![The data](data.svg)

## Executive summary

The structure search selected `CW(Per x SE + SE, C) + C + WN + SE` (Bayesian information criterion 530.7; negative log likelihood 227.0 on 240 points). It decomposes into 6 additive components, listed by how much each reduces extrapolation error:

1. A constant.
2. A smooth function.
3. An approximately periodic function with a period of 1.0 years. This function applies until Jan 1942 and from Jul 1945 onwards.
4. A constant. This function applies from Jan 1942 until Jul 1945.
5. Uncorrelated noise.
6. A smooth function. This function applies until Jan 1942 and from Jul 1945 onwards.

| # | R² cumulative (%) | Residual variance explained (%) | Cross-validated MAE | MAE reduction (%) |
|---|---|---|---|---|
| data | | | 8.08 | |
| 1 | -0.0 | -0.0 | 2.06 | 74.5 |
| 2 | 54.7 | 54.7 | 1.55 | 24.9 |
| 3 | 93.0 | 84.6 | 0.898 | 41.9 |
| 4 | 94.3 | 18.0 | 0.804 | 10.5 |
| 5 | 96.3 | 35.6 | 0.804 | 0.0 |
| 6 | 100.0 | 100.0 | 0.804 | -0.0 |

MAE: mean absolute error over ten contiguous held-out blocks.

## The formula found

The structure search is a symbolic regression: it searches a grammar of expressions over kernels and keeps the one with the lowest Bayesian information criterion. This is the expression it kept, with every fitted parameter in place; $t$ is in years.

$$y(t) = f_{1}(t) + f_{2}(t) + f_{3}(t) + f_{4}(t) + f_{5}(t) + f_{6}(t)$$

Each component is a Gaussian process with its own covariance:

$$f_{1}(t) \sim \mathrm{GP}\!\left(0,\ 57.77\right)$$

$$f_{2}(t) \sim \mathrm{GP}\!\left(0,\ 5.468\,\exp\!\left(-\frac{(t - t')^2}{2 \cdot 2.306^2}\right)\right)$$

$$f_{3}(t) \sim \mathrm{GP}\!\left(0,\ (1 - w_{1}(t))(1 - w_{1}(t'))\left[2.519\,\frac{\exp\!\left(0.7377\cos\frac{2\pi (t - t')}{0.9998}\right) - I_0(0.7377)}{e^{0.7377} - I_0(0.7377)} \cdot \exp\!\left(-\frac{(t - t')^2}{2 \cdot 8.874^2}\right)\right]\right)$$

$$f_{4}(t) \sim \mathrm{GP}\!\left(0,\ w_{1}(t)\,w_{1}(t')\left[4.076\right]\right)$$

$$f_{5}(t) \sim \mathrm{GP}\!\left(0,\ 0.09864\,\delta_{t t'}\right)$$

$$f_{6}(t) \sim \mathrm{GP}\!\left(0,\ (1 - w_{1}(t))(1 - w_{1}(t'))\left[0.4599\,\exp\!\left(-\frac{(t - t')^2}{2 \cdot 0.1171^2}\right)\right]\right)$$

The changes of regime weigh the components with:

$$w_{1}(t) = \tfrac{1}{4}\left(1 + \tanh\left(0.54\,(t - 1942)\right)\right)\left(1 - \tanh\left(0.54\,(t - 1946)\right)\right)$$

These components have a closed form, read from their posterior mean:

$$f_{1}(t) = 7.554$$

$$f_{4}(t) = -1.091\,w_{1}(t)$$

The other components are smooth, quasi-periodic or noise without a short formula: a draw from such a process has no finite closed form; its covariance above is the formula.

Closed form of each component:

| # | component | closed form |
|---|---|---|
| 1 | A constant | exact polynomial of degree 0 |
| 2 | A smooth function | none: the covariance is the formula |
| 3 | An approximately periodic function with a period of 1.0 years. This function applies until Jan 1942 and from Jul 1945 onwards | none: the covariance is the formula |
| 4 | A constant. This function applies from Jan 1942 until Jul 1945 | exact polynomial of degree 0 |
| 5 | Uncorrelated noise | none: the covariance is the formula |
| 6 | A smooth function. This function applies until Jan 1942 and from Jul 1945 onwards | none: the covariance is the formula |

## Search path and stopping criterion

The 2014 configuration searches a fixed number of levels and keeps the lowest BIC; it has no stopping rule. The table follows the winner of each level. A change of structure adds or removes terms; a level that keeps the structure only refits its parameters. BIC differences read on the scale of Kass and Raftery (1995): under 2 barely worth mentioning, 2 to 6 positive, 6 to 10 strong, over 10 very strong.

| level | BIC | change | structure | components | closed | params |
|---|---|---|---|---|---|---|
| 1 | 813.7 |  | SE + WN | 2 | 0 | 3 |
| 2 | 619.3 | -194.4 | Per x SE + SE + WN | 3 | 0 | 7 |
| 3 | 540.5 | -78.8 | Per x SE + SE + SE + WN | 4 | 0 | 9 |
| 4 | 534.2 | -6.4 | CW(Per x SE + SE, C) + SE + WN | 5 | 1 | 13 |
| 5 | 530.8 | -3.3 | C + CW(Per x SE + SE, C) + SE + WN | 6 | 2 | 14 |
| 6 | 530.7 | -0.1 | C + CW(Per x SE + SE, C) + SE + WN | 6 | 2 | 14 |
| 7 | 530.7 | -0.0 | C + CW(Per x SE + SE, C) + SE + WN | 6 | 2 | 14 |
| 8 (kept) | 530.7 | -0.0 | C + CW(Per x SE + SE, C) + SE + WN | 6 | 2 | 14 |
| 9 | 530.7 | +0.0 | C + CW(Per x SE + SE, C) + SE + WN | 6 | 2 | 14 |
| 10 | 530.7 | +0.0 | C + CW(Per x SE + SE, C) + SE + WN | 6 | 2 | 14 |

![BIC per search level](bic.svg)

Posterior weight of each structure met, exp(-BIC/2) normalised under a flat prior, after the Bayesian machine scientist (Guimerà et al., 2020): where the search keeps one structure, the weights say how much the data prefer it to the others. The 2 structure(s) under 0.001 are left out.

| structure | lowest BIC | weight |
|---|---|---|
| C + CW(Per x SE + SE, C) + SE + WN | 530.7 | 0.846 |
| CW(Per x SE + SE, C) + SE + WN | 534.2 | 0.148 |
| Per x SE + SE + SE + WN | 540.5 | 0.006 |

- The search ran 10 levels (the published 2014 search: ten levels, no stopping rule) and keeps the model of level 8, C + CW(Per x SE + SE, C) + SE + WN, BIC 530.7.
- That structure was first reached at level 5; the levels after it only refitted its parameters, lowering BIC by 0.1 in all.
- The best different structure, CW(Per x SE + SE, C) + SE + WN (level 4), is 3.5 above it: positive evidence for the kept one on the Kass and Raftery scale.
- Weighed by exp(-BIC/2) under a flat prior, as the Bayesian machine scientist weighs expressions, the structures met give the kept one a posterior weight of 0.846 and the next, CW(Per x SE + SE, C) + SE + WN, 0.148: the data favour the kept one without ruling the next out.
- BIC stopped falling at level 6. The 4 level(s) searched past it kept its structure and lowered BIC by 0.03 in all, less than the tolerance of 0.1: going on buys nothing the data pay for.
- A rule stopping after 1 level(s) without a fall above 0.1 would have ended the search at level 7 and kept the same structure, less refitted.
- A rule stopping after 2 level(s) without a fall above 0.1 would have ended the search at level 8 and kept the same model.
- The kept expression has 14 parameters for 240 points: its BIC is a fit term 2 NLL = 454.0 plus a complexity term p ln n = 76.7, each extra parameter costing 5.48; it splits into 6 components, 2 of them with a closed form.

![Data and full posterior](fit.svg)

## Detailed discussion of the components

### Component 1: A constant

This component is constant.

This component explains -0.0% of the data's variance on its own; with the components above it, -0.0%. Adding it changes the cross-validated MAE by -74.5%.

*Extrapolation.* This component is assumed to stay constant.

![Component 1](component_1.svg)

### Component 2: A smooth function

This component is a smooth function with a typical lengthscale of 2.3 years.

This component explains 54.7% of the data's variance on its own; with the components above it, 54.7%. Adding it changes the cross-validated MAE by -24.9%.

*Extrapolation.* This component is assumed to continue smoothly but is also assumed to be stationary so its distribution will return to the prior. The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of 2.3 years.

![Component 2](component_2.svg)

### Component 3: An approximately periodic function with a period of 1.0 years. This function applies until Jan 1942 and from Jul 1945 onwards

This component is approximately periodic with a period of 1.0 years. Across periods the shape of this function varies smoothly with a typical lengthscale of 8.9 years. The shape of this function within each period has a typical lengthscale of 2.2 months. This component applies until Jan 1942 and from Jul 1945 onwards.

This component explains 38.9% of the data's variance on its own; with the components above it, 93.0%. Adding it changes the cross-validated MAE by -41.9%.

*Extrapolation.* This component is assumed to continue to be approximately periodic. The shape of the function is assumed to vary smoothly between periods but will return to the prior. The prior is entirely uncertain about the phase of the periodic function. Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the function.

![Component 3](component_3.svg)

### Component 4: A constant. This function applies from Jan 1942 until Jul 1945

This component is constant. This component applies from Jan 1942 until Jul 1945.

This component explains 6.3% of the data's variance on its own; with the components above it, 94.3%. Adding it changes the cross-validated MAE by -10.5%.

*Extrapolation.* This component is assumed to stop before the end of the data and will therefore be extrapolated as zero.

![Component 4](component_4.svg)

### Component 5: Uncorrelated noise

This component models uncorrelated noise.

This component explains 2.4% of the data's variance on its own; with the components above it, 96.3%. Adding it changes the cross-validated MAE by -0.0%.

*Extrapolation.* This component assumes the uncorrelated noise will continue indefinitely.

![Component 5](component_5.svg)

### Component 6: A smooth function. This function applies until Jan 1942 and from Jul 1945 onwards

This component is a smooth function with a typical lengthscale of 6.1 weeks. This component applies until Jan 1942 and from Jul 1945 onwards.

This component explains 8.8% of the data's variance on its own; with the components above it, 100.0%. Adding it changes the cross-validated MAE by 0.0%.

*Extrapolation.* This component is assumed to continue smoothly but is also assumed to be stationary so its distribution will return to the prior. The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of 6.1 weeks.

![Component 6](component_6.svg)

## Model checking

Each statistic compares draws of a component's posterior with draws of its prior. A small p-value (bold below 0.05) points at a property of the data that the component's prior does not reproduce.

| # | QQ max | QQ min | ACF min | ACF min lag | Periodogram max | Periodogram peak | MMD |
|---|---|---|---|---|---|---|---|
| 1 | 0.310 | 0.310 | 0.842 | 0.808 | 0.674 | 0.994 | **0.000** |
| 2 | 0.722 | 0.900 | 0.314 | 0.332 | 0.950 | 0.650 | **0.000** |
| 3 | 0.560 | 0.808 | 0.460 | 0.948 | 0.484 | 0.630 | **0.000** |
| 4 | 0.710 | 0.728 | 0.974 | 0.958 | 0.958 | 0.990 | **0.000** |
| 5 | 0.780 | 0.982 | 0.990 | 0.978 | 0.898 | 0.964 | **0.000** |
| 6 | 0.938 | 0.794 | 0.904 | 0.964 | 0.892 | 0.916 | **0.000** |

QQ: quantile-quantile gaps. ACF: autocorrelation function. MMD: maximum mean discrepancy two-sample test.
