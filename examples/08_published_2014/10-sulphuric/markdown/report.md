# An automatic report for the dataset: 10-sulphuric

*Produced by `autostat` (automatic-statistician-revisited), a Python 3 port of the Automatic Bayesian Covariance Discovery system of Lloyd et al. (AAAI 2014).*

![The data](data.svg)

## Executive summary

The structure search selected `CP(WN + Per, CP(C + WN, WN) + CW(SE, SE)) + Per x SE + SE` (Bayesian information criterion 3816.3; negative log likelihood 1834.5 on 462 points). It decomposes into 9 additive components, listed by how much each reduces extrapolation error:

1. A very smooth function.
2. A constant. This function applies from 1964 until 1990.
3. An approximately periodic function with a period of 1.0 years.
4. A smooth function. This function applies from 1969 until 1977.
5. A smooth function. This function applies from 1964 until 1969 and from 1977 onwards.
6. A periodic function with a period of 2.6 years. This function applies until 1964.
7. Uncorrelated noise. This function applies until 1964.
8. Uncorrelated noise. This function applies from 1964 until 1990.
9. Uncorrelated noise. This function applies from 1990 onwards.

| # | R² cumulative (%) | Residual variance explained (%) | Cross-validated MAE | MAE reduction (%) |
|---|---|---|---|---|
| data | | | 131 | |
| 1 | 34.6 | 34.6 | 37.7 | 71.3 |
| 2 | 61.8 | 41.7 | 18.8 | 50.1 |
| 3 | 71.9 | 26.4 | 16.3 | 13.3 |
| 4 | 90.5 | 66.3 | 15.7 | 3.9 |
| 5 | 93.1 | 27.6 | 15.5 | 0.8 |
| 6 | 93.3 | 2.6 | 15.4 | 0.7 |
| 7 | 93.5 | 3.2 | 15.4 | 0.0 |
| 8 | 99.8 | 96.9 | 15.4 | 0.0 |
| 9 | 100.0 | 100.0 | 15.4 | 0.0 |

MAE: mean absolute error over ten contiguous held-out blocks.

## The formula found

The structure search is a symbolic regression: it searches a grammar of expressions over kernels and keeps the one with the lowest Bayesian information criterion. This is the expression it kept, with every fitted parameter in place; $t$ is in years.

$$y(t) = f_{1}(t) + f_{2}(t) + f_{3}(t) + f_{4}(t) + f_{5}(t) + f_{6}(t) + f_{7}(t) + f_{8}(t) + f_{9}(t)$$

Each component is a Gaussian process with its own covariance:

$$f_{1}(t) \sim \mathrm{GP}\!\left(0,\ 4513\,\exp\!\left(-\frac{(t - t')^2}{2 \cdot 27.23^2}\right)\right)$$

$$f_{2}(t) \sim \mathrm{GP}\!\left(0,\ w_{1}(t)\,w_{1}(t')\left[(1 - w_{2}(t))(1 - w_{2}(t'))\left[3864\right]\right]\right)$$

$$f_{3}(t) \sim \mathrm{GP}\!\left(0,\ 153.8\,\frac{\exp\!\left(0.9068\cos\frac{2\pi (t - t')}{1.003}\right) - I_0(0.9068)}{e^{0.9068} - I_0(0.9068)} \cdot \exp\!\left(-\frac{(t - t')^2}{2 \cdot 14.65^2}\right)\right)$$

$$f_{4}(t) \sim \mathrm{GP}\!\left(0,\ w_{1}(t)\,w_{1}(t')\left[w_{3}(t)\,w_{3}(t')\left[1573\,\exp\!\left(-\frac{(t - t')^2}{2 \cdot 0.2653^2}\right)\right]\right]\right)$$

$$f_{5}(t) \sim \mathrm{GP}\!\left(0,\ w_{1}(t)\,w_{1}(t')\left[(1 - w_{3}(t))(1 - w_{3}(t'))\left[71.21\,\exp\!\left(-\frac{(t - t')^2}{2 \cdot 0.6018^2}\right)\right]\right]\right)$$

$$f_{6}(t) \sim \mathrm{GP}\!\left(0,\ (1 - w_{1}(t))(1 - w_{1}(t'))\left[13.65\,\frac{\exp\!\left(1.942\cos\frac{2\pi (t - t')}{2.643}\right) - I_0(1.942)}{e^{1.942} - I_0(1.942)}\right]\right)$$

$$f_{7}(t) \sim \mathrm{GP}\!\left(0,\ (1 - w_{1}(t))(1 - w_{1}(t'))\left[20.54\,\delta_{t t'}\right]\right)$$

$$f_{8}(t) \sim \mathrm{GP}\!\left(0,\ w_{1}(t)\,w_{1}(t')\left[(1 - w_{2}(t))(1 - w_{2}(t'))\left[194.3\,\delta_{t t'}\right]\right]\right)$$

$$f_{9}(t) \sim \mathrm{GP}\!\left(0,\ w_{1}(t)\,w_{1}(t')\left[w_{2}(t)\,w_{2}(t')\left[53.82\,\delta_{t t'}\right]\right]\right)$$

The changes of regime weigh the components with:

$$w_{1}(t) = \tfrac{1}{2}\left(1 + \tanh\left(0.5771\,(t - 1964)\right)\right)$$

$$w_{2}(t) = \tfrac{1}{2}\left(1 + \tanh\left(5.252\,(t - 1991)\right)\right)$$

$$w_{3}(t) = \tfrac{1}{4}\left(1 + \tanh\left(1.254\,(t - 1970)\right)\right)\left(1 - \tanh\left(1.254\,(t - 1977)\right)\right)$$

These components have a closed form, read from their posterior mean:

$$f_{1}(t) \approx u - \frac{8.151 \left(u - 28.08\right) \left(u + 12.61\right)}{u + 34.58 + \frac{932.0}{u}} + 72.26,\quad u = t - 1956$$

$$f_{6}(t) \approx (1 - w_{1}(t))\,\left(3.403\cos\!\left(\frac{2\pi \cdot (t - 1956)}{2.643} + 1.729\right) + 3.126\cos\!\left(\frac{2\pi \cdot 2 \cdot (t - 1956)}{2.643} - 0.01766\right) + 1.84\cos\!\left(\frac{2\pi \cdot 3 \cdot (t - 1956)}{2.643} + 0.1857\right) + 0.2446\cos\!\left(\frac{2\pi \cdot 4 \cdot (t - 1956)}{2.643} + 1.02\right) + 0.009958\cos\!\left(\frac{2\pi \cdot 5 \cdot (t - 1956)}{2.643} - 1.496\right) + 0.006456\cos\!\left(\frac{2\pi \cdot 6 \cdot (t - 1956)}{2.643} + 2.375\right)\right)$$

A periodic kernel is exactly a sum of cosines, so a periodic component is a Fourier series in its period; it is written up to the harmonic after which the rest weighs less than 0.1 % of the component, hence the approximate sign.

The other components are smooth, quasi-periodic or noise without a short formula: a draw from such a process has no finite closed form; its covariance above is the formula.

Closed form of each component:

| # | component | closed form |
|---|---|---|
| 1 | A very smooth function | symbolic regression (PySR) on the posterior mean, largest gap 0.54 % of the component |
| 2 | A constant. This function applies from 1964 until 1990 | none: the covariance is the formula |
| 3 | An approximately periodic function with a period of 1.0 years | none: the covariance is the formula |
| 4 | A smooth function. This function applies from 1969 until 1977 | none: the covariance is the formula |
| 5 | A smooth function. This function applies from 1964 until 1969 and from 1977 onwards | none: the covariance is the formula |
| 6 | A periodic function with a period of 2.6 years. This function applies until 1964 | Fourier series, 6 harmonics, largest gap 0.049 % of the component |
| 7 | Uncorrelated noise. This function applies until 1964 | none: the covariance is the formula |
| 8 | Uncorrelated noise. This function applies from 1964 until 1990 | none: the covariance is the formula |
| 9 | Uncorrelated noise. This function applies from 1990 onwards | none: the covariance is the formula |

## Search path and stopping criterion

The 2014 configuration searches a fixed number of levels and keeps the lowest BIC; it has no stopping rule. The table follows the winner of each level. A change of structure adds or removes terms; a level that keeps the structure only refits its parameters. BIC differences read on the scale of Kass and Raftery (1995): under 2 barely worth mentioning, 2 to 6 positive, 6 to 10 strong, over 10 very strong.

| level | BIC | change | structure | components | closed | params |
|---|---|---|---|---|---|---|
| 1 | 4222.0 |  | WN + SE | 2 | 0 | 3 |
| 2 | 4033.0 | -189.0 | WN + Per x SE + SE | 3 | 0 | 7 |
| 3 | 3933.1 | -100.0 | CP(WN + SE, WN + SE) + Per x SE | 5 | 0 | 12 |
| 4 | 3873.1 | -60.0 | CP(WN + SE, WN + SE) + Per x SE + SE | 6 | 0 | 14 |
| 5 | 3854.1 | -19.0 | CP(WN + Per, WN + SE) + Per x SE + SE | 6 | 1 | 15 |
| 6 | 3843.8 | -10.3 | CP(WN + Per, CP(WN, WN) + SE) + Per x SE + SE | 7 | 1 | 18 |
| 7 | 3834.9 | -8.9 | CP(WN + Per, CP(C + WN, WN) + SE) + Per x SE + SE | 8 | 1 | 19 |
| 8 | 3825.6 | -9.4 | CP(WN + Per, CP(C + WN, WN) + CW(SE, SE)) + Per x SE + SE | 9 | 1 | 24 |
| 9 | 3819.2 | -6.3 | CP(WN + Per, CP(C + WN, WN) + CW(SE, SE)) + Per x SE + SE | 9 | 1 | 24 |
| 10 (kept) | 3816.3 | -3.0 | CP(WN + Per, CP(C + WN, WN) + CW(SE, SE)) + Per x SE + SE | 9 | 1 | 24 |

![BIC per search level](bic.svg)

- The search ran 10 levels (the published 2014 search: ten levels, no stopping rule) and keeps the model of level 10, CP(WN + Per, CP(C + WN, WN) + CW(SE, SE)) + Per x SE + SE, BIC 3816.3.
- That structure was first reached at level 8; the levels after it only refitted its parameters, lowering BIC by 9.3 in all.
- The best different structure, CP(WN + Per, CP(C + WN, WN) + SE) + Per x SE + SE (level 7), is 18.7 above it: very strong evidence for the kept one on the Kass and Raftery scale.
- The last level lowered BIC by 3.0 through its parameters only: the structure had settled, its fit had not quite.
- A rule stopping after 1 level(s) without a fall above 0.1 would not have stopped within these 10 levels.
- A rule stopping after 2 level(s) without a fall above 0.1 would not have stopped within these 10 levels.
- The kept expression has 24 parameters for 462 points: its BIC is a fit term 2 NLL = 3669.0 plus a complexity term p ln n = 147.3, each extra parameter costing 6.14; it splits into 9 components, 1 of them with a closed form.

![Data and full posterior](fit.svg)

## Detailed discussion of the components

### Component 1: A very smooth function

This component is a very smooth function.

This component explains 34.6% of the data's variance on its own; with the components above it, 34.6%. Adding it changes the cross-validated MAE by -71.3%.

*Extrapolation.* This component is assumed to continue very smoothly but is also assumed to be stationary so its distribution will eventually return to the prior. The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of 27.2 years.

![Component 1](component_1.svg)

### Component 2: A constant. This function applies from 1964 until 1990

This component is constant. This component applies from 1964 until 1990.

This component explains 54.7% of the data's variance on its own; with the components above it, 61.8%. Adding it changes the cross-validated MAE by -50.1%.

*Extrapolation.* This component is assumed to stop before the end of the data and will therefore be extrapolated as zero.

![Component 2](component_2.svg)

### Component 3: An approximately periodic function with a period of 1.0 years

This component is approximately periodic with a period of 1.0 years. Across periods the shape of this function varies smoothly with a typical lengthscale of 14.7 years. The shape of this function within each period has a typical lengthscale of 2.0 months.

This component explains 9.6% of the data's variance on its own; with the components above it, 71.9%. Adding it changes the cross-validated MAE by -13.3%.

*Extrapolation.* This component is assumed to continue to be approximately periodic. The shape of the function is assumed to vary smoothly between periods but will return to the prior. The prior is entirely uncertain about the phase of the periodic function. Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the function.

![Component 3](component_3.svg)

### Component 4: A smooth function. This function applies from 1969 until 1977

This component is a smooth function with a typical lengthscale of 3.2 months. This component applies from 1969 until 1977.

This component explains 7.0% of the data's variance on its own; with the components above it, 90.5%. Adding it changes the cross-validated MAE by -3.9%.

*Extrapolation.* This component is assumed to stop before the end of the data and will therefore be extrapolated as zero.

![Component 4](component_4.svg)

### Component 5: A smooth function. This function applies from 1964 until 1969 and from 1977 onwards

This component is a smooth function with a typical lengthscale of 7.2 months. This component applies from 1964 until 1969 and from 1977 onwards.

This component explains 6.7% of the data's variance on its own; with the components above it, 93.1%. Adding it changes the cross-validated MAE by -0.8%.

*Extrapolation.* This component is assumed to continue smoothly but is also assumed to be stationary so its distribution will return to the prior. The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of 7.2 months.

![Component 5](component_5.svg)

### Component 6: A periodic function with a period of 2.6 years. This function applies until 1964

This component is periodic with a period of 2.6 years. The shape of this function within each period has a typical lengthscale of 3.6 months. This component applies until 1964.

This component explains 0.4% of the data's variance on its own; with the components above it, 93.3%. Adding it changes the cross-validated MAE by -0.7%.

*Extrapolation.* This component is assumed to stop before the end of the data and will therefore be extrapolated as zero.

![Component 6](component_6.svg)

### Component 7: Uncorrelated noise. This function applies until 1964

This component models uncorrelated noise. This component applies until 1964.

This component explains 0.3% of the data's variance on its own; with the components above it, 93.5%. Adding it changes the cross-validated MAE by -0.0%.

*Extrapolation.* This component is assumed to stop before the end of the data and will therefore be extrapolated as zero.

![Component 7](component_7.svg)

### Component 8: Uncorrelated noise. This function applies from 1964 until 1990

This component models uncorrelated noise. This component applies from 1964 until 1990.

This component explains 9.6% of the data's variance on its own; with the components above it, 99.8%. Adding it changes the cross-validated MAE by -0.0%.

*Extrapolation.* This component is assumed to stop before the end of the data and will therefore be extrapolated as zero.

![Component 8](component_8.svg)

### Component 9: Uncorrelated noise. This function applies from 1990 onwards

This component models uncorrelated noise. This component applies from 1990 onwards.

This component explains 0.2% of the data's variance on its own; with the components above it, 100.0%. Adding it changes the cross-validated MAE by -0.0%.

*Extrapolation.* This component assumes the uncorrelated noise will continue indefinitely.

![Component 9](component_9.svg)

## Model checking

Each statistic compares draws of a component's posterior with draws of its prior. A small p-value (bold below 0.05) points at a property of the data that the component's prior does not reproduce.

| # | QQ max | QQ min | ACF min | ACF min lag | Periodogram max | Periodogram peak | MMD |
|---|---|---|---|---|---|---|---|
| 1 | 0.098 | 0.190 | 0.118 | 0.126 | 0.222 | 0.742 | **0.000** |
| 2 | 0.480 | 0.386 | 0.974 | 0.992 | 0.770 | 0.990 | **0.000** |
| 3 | 0.146 | 0.894 | 0.918 | 0.964 | 0.504 | 0.684 | **0.000** |
| 4 | 0.150 | 0.368 | 0.218 | 0.956 | 0.194 | 0.718 | **0.002** |
| 5 | 0.584 | 0.426 | 0.718 | 0.932 | 0.556 | 0.900 | **0.004** |
| 6 | 0.268 | 0.942 | 0.338 | 0.832 | 0.998 | 0.962 | 0.133 |
| 7 | 0.856 | 0.812 | 0.962 | 0.976 | 0.918 | 0.980 | **0.037** |
| 8 | 0.370 | 0.222 | 0.962 | 0.968 | 0.996 | 0.972 | **0.000** |
| 9 | 0.702 | 0.922 | 0.994 | 0.998 | 0.978 | 0.964 | 0.115 |

QQ: quantile-quantile gaps. ACF: autocorrelation function. MMD: maximum mean discrepancy two-sample test.
