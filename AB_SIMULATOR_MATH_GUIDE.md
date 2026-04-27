# A/B Simulator Math Guide

This guide explains what each part of the simulator is trying to do, why it matters, and how it is computed.

For standard definitions, see:
- [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
- [Central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)
- [p-value](https://en.wikipedia.org/wiki/P-value)
- [Statistical power](https://en.wikipedia.org/wiki/Power_of_a_test)
- [False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate)
- [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)

## 1) Core Modeling Assumptions

The simulator needs a simple data-generating model so you can quickly stress-test experiment designs. These assumptions make simulation fast and interpretable, but they are approximations.

Assumptions used:
- Observations are modeled as normal:
  - \(X \sim \mathcal{N}(\mu, \sigma^2)\)
- Users/trials are i.i.d. between groups.
- Treatment effect is stable over time (no drift/seasonality in model).
- Frequentist interpretation assumes fixed-horizon testing.

## 2) Fast Simulation Trick (Performance)

At large sample sizes, generating every individual data point is slow. The simulator samples the group mean directly so it stays responsive.

Equivalent shortcut:
- Instead of drawing all \(n\) observations, draw:
  - \(\bar{X} \sim \mathcal{N}\left(\mu, \sigma^2/n\right)\)

Under the normal model, this is mathematically equivalent for mean-based inference.

## 3) Frequentist Test Used in Code

This estimates whether observed A/B differences are likely under a null of no effect.

Implemented formulas:
- Standard error:
  - \(SE = \sqrt{\sigma_A^2/n_A + \sigma_B^2/n_B}\)
- Test statistic:
  - \(t = (\bar{X}_B - \bar{X}_A)/SE\)
- Two-sided p-value:
  - \(p = 2 \cdot (1 - \Phi(|t|))\)

Notes:
- `tCDF` currently uses normal CDF, so behavior is z-approximation.
- References:
  - [Z-test](https://en.wikipedia.org/wiki/Z-test)
  - [Student's t-test](https://en.wikipedia.org/wiki/Student%27s_t-test)

## 4) Required Sample Size (Power Planning)

This gives an upfront sample-size target so your test has a good chance to detect the minimum effect you care about.

Implemented formulas:
- Convert alpha/power to critical values via inverse normal (`probit`):
  - \(z_{\alpha/2} = -\Phi^{-1}(\alpha/2)\)
  - \(z_{\beta} = -\Phi^{-1}(1-\text{power})\)
- Per arm:
  - \(n_{\text{per arm}} = \left\lceil 2\left(\frac{(z_{\alpha/2}+z_{\beta})\sigma}{\delta}\right)^2 \right\rceil\)
- Total:
  - \(N_{\text{total}} = 2 \cdot n_{\text{per arm}}\)

Default call in app:
- \(\alpha = 0.05\), power \(= 0.80\)

References:
- [Significance level](https://en.wikipedia.org/wiki/Statistical_significance)
- [Quantile function (inverse CDF)](https://en.wikipedia.org/wiki/Quantile_function)

## 5) Bayesian Update Used in Code

This updates prior belief with observed data to get posterior uncertainty and posterior mean, then estimates \(P(B>A)\).

Implemented formulas:
- Inputs:
  - Prior mean slider: `priorMean`
  - Prior variance slider: `priorVar`
  - Prior strength slider: `priorStrength`
- Let \(\sigma^2 = \text{sigma}^2\)
- Prior precision:
  - \(P_0 = 1/\text{priorVar} + \text{priorStrength}/\sigma^2\)
- Posterior variance:
  - \(\sigma^2_{\text{post}} = 1 / (P_0 + n/\sigma^2)\)
- Posterior mean:
  - \(\mu_{\text{post}} = \sigma^2_{\text{post}} \cdot \left(\frac{\mu_0}{\text{priorVar}} + \frac{\text{priorStrength}\cdot\mu_0}{\sigma^2} + \frac{n\bar{X}}{\sigma^2}\right)\)

Then \(P(B>A)\) is estimated by Monte Carlo draws from the two posterior normals.

References:
- [Conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
- [Posterior probability](https://en.wikipedia.org/wiki/Posterior_probability)

## 6) "Faster Decision" in the Scorecard

You want to know which method reaches a reliable decision threshold sooner, not just which has a higher endpoint value.

Implemented rule:
- Find first sample size where frequentist power >= 80%
- Find first sample size where Bayesian decision rate >= 80%
- Smaller crossing sample size is "faster"
- If neither crosses, show `NONE`

## 7) FDR Interpretation in This Simulator

FDR tells you how many positive calls are expected to be false in this simulation setup.

Computed here:
- Frequentist:
  - \(FDR = FP/(FP+TP)\)
- Bayesian false-confidence analog:
  - \(BayesFDR = BayesFP/(BayesFP+BayesTP)\)

Important:
- Null and effect cases are simulated in equal counts.
- So displayed FDR corresponds to an implicit 50% true-effect prevalence setting.

## 8) Practical Reading Tips

Different panels use different types of quantities (analytic vs simulation estimates). Reading them correctly avoids confusion.

How to interpret:
- `Required n` is an analytic planning estimate under current assumptions.
- Power/FDR curves are Monte Carlo estimates and can vary slightly run-to-run.
- If assumptions fail (heavy tails, dependence, temporal effects), conclusions can be biased.
