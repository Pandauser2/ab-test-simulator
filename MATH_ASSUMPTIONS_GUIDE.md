# A/B Simulator Math Guide

This document explains the key assumptions and formulas used in the simulator.

For standard definitions, see:
- [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
- [Central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)
- [p-value](https://en.wikipedia.org/wiki/P-value)
- [Statistical power](https://en.wikipedia.org/wiki/Power_of_a_test)
- [False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate)
- [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)

## 1) Core modeling assumptions

- Observations are modeled as normal:
  - \(X \sim \mathcal{N}(\mu, \sigma^2)\)
- Users/trials are i.i.d. between groups.
- Treatment effect is stable over the experiment window (no time drift).
- Frequentist calculations assume fixed-horizon testing (no optional stopping correction).

## 2) Fast simulation trick (performance)

To avoid sampling all individuals at large \(n\), the simulator draws group means directly:

- \(\bar{X} \sim \mathcal{N}\left(\mu, \sigma^2/n\right)\)

This is mathematically equivalent under the normal model and much faster.

## 3) Frequentist test used in code

The code computes:

- Standard error:
  - \(SE = \sqrt{\sigma_A^2/n_A + \sigma_B^2/n_B}\)
- Test statistic:
  - \(t = (\bar{X}_B - \bar{X}_A)/SE\)
- Two-sided p-value:
  - \(p = 2 \cdot (1 - \Phi(|t|))\)

Notes:
- `tCDF` currently returns normal CDF (`normalCDF`), so this behaves as a z-approximation.
- In UI text this is described as a frequentist z-style inference flow.
- Related references:
  - [Z-test](https://en.wikipedia.org/wiki/Z-test)
  - [Student's t-test](https://en.wikipedia.org/wiki/Student%27s_t-test)

## 4) Required sample size (power planning)

The simulator computes required sample size from alpha/power using inverse normal (`probit`):

- \(z_{\alpha/2} = -\Phi^{-1}(\alpha/2)\)
- \(z_{\beta} = -\Phi^{-1}(1-\text{power})\)
- Per arm:
  - \(n_{\text{per arm}} = \left\lceil 2\left(\frac{(z_{\alpha/2}+z_{\beta})\sigma}{\delta}\right)^2 \right\rceil\)
- Total:
  - \(N_{\text{total}} = 2 \cdot n_{\text{per arm}}\)

Default call in app:
- \(\alpha = 0.05\), power \(= 0.80\)
- Related references:
  - [Significance level](https://en.wikipedia.org/wiki/Statistical_significance)
  - [Inverse transform / quantile function](https://en.wikipedia.org/wiki/Quantile_function)

## 5) Bayesian update used in code

The simulator uses a normal-normal style closed-form update with two prior controls:

- Prior mean slider: `priorMean`
- Prior variance slider: `priorVar`
- Prior strength slider: `priorStrength` (adds precision scaled by \(\sigma^2\))

Implementation:

- \(\sigma^2 = \text{sigma}^2\)
- Prior precision:
  - \(P_0 = 1/\text{priorVar} + \text{priorStrength}/\sigma^2\)
- Posterior variance:
  - \(\sigma^2_{\text{post}} = 1 / (P_0 + n/\sigma^2)\)
- Posterior mean:
  - \(\mu_{\text{post}} = \sigma^2_{\text{post}} \cdot \left(\frac{\mu_0}{\text{priorVar}} + \frac{\text{priorStrength}\cdot\mu_0}{\sigma^2} + \frac{n\bar{X}}{\sigma^2}\right)\)

The app then estimates \(P(B>A)\) by Monte Carlo sampling from the two posterior normals.
- Related references:
  - [Conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
  - [Posterior probability](https://en.wikipedia.org/wiki/Posterior_probability)

## 6) "Faster Decision" in scorecard

`FASTER DECISION` is based on the power chart crossing rule:

- Find first sample size where frequentist power >= 80%
- Find first sample size where Bayesian decision rate >= 80%
- Smaller crossing sample size = faster method
- If neither crosses, display `NONE`

## 7) FDR interpretation in this simulator

FDR values shown are:

- Frequentist: \(FP/(FP+TP)\)
- Bayesian false confidence analog: \(BayesFP/(BayesFP+BayesTP)\)

Important:
- In this simulation, null/effect cases are sampled in equal counts.
- So displayed FDR corresponds to an implicit 50% true-effect prevalence setup.

## 8) Practical reading tips

- `Required n` is an analytic estimate under current assumptions.
- Power/FDR curves are Monte Carlo estimates (finite-trial noise is expected).
- If assumptions are violated (heavy tails, dependence, strong time effects), results can be biased.
