"""
Core statistical comparison: for one series of per-game counts, fit an
intercept-only Poisson and an intercept-only Negative Binomial, test for
overdispersion, and compare both models in-sample (AIC on the training
season) and out-of-sample (negative log-likelihood on the test games).

This module has no notion of "which player/stat" — pipeline.py calls
fit_and_compare() once per series and stitches the results together.
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats


def _poisson_loglik(counts: np.ndarray, lam: float) -> float:
    return float(sp_stats.poisson.logpmf(counts, lam).sum())


def fit_poisson(train_counts: np.ndarray) -> dict:
    lam = float(np.mean(train_counts))
    ll = _poisson_loglik(train_counts, lam)
    k = 1  # one free parameter: lambda
    return {
        "lambda": lam,
        "loglik": ll,
        "aic": 2 * k - 2 * ll,
        "bic": k * np.log(len(train_counts)) - 2 * ll,
        "k": k,
    }


def fit_negbin(train_counts: np.ndarray) -> dict:
    """
    NB2 parametrization (variance = mu + alpha * mu^2), intercept-only.
    Fit by MLE via statsmodels; `alpha` is the overdispersion parameter
    (alpha -> 0 recovers Poisson).
    """
    X = np.ones((len(train_counts), 1))
    model = sm.NegativeBinomial(train_counts, X, loglike_method="nb2")
    res = model.fit(disp=0, maxiter=200)

    mu = float(np.exp(res.params[0]))  # log link on the intercept
    alpha = float(res.params[-1])
    ll = float(res.llf)
    k = 2  # mu and alpha
    return {
        "mu": mu,
        "alpha": alpha,
        "loglik": ll,
        "aic": 2 * k - 2 * ll,
        "bic": k * np.log(len(train_counts)) - 2 * ll,
        "k": k,
    }


def negbin_logpmf(counts: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """log-pmf of the NB2(mu, alpha) distribution, via scipy's (n, p) form."""
    alpha = max(alpha, 1e-8)  # guard against a near-zero alpha at the boundary
    r = 1.0 / alpha
    p = r / (r + mu)
    return sp_stats.nbinom.logpmf(counts, r, p)


def dispersion_test(train_counts: np.ndarray) -> dict:
    """
    Variance-to-mean ratio (always reported) plus the classic Poisson
    dispersion chi-square test: sum((y - mean)^2) / mean ~ chi2(n-1) under
    H0: data are Poisson. A small p-value is evidence of overdispersion.
    """
    n = len(train_counts)
    mean = float(np.mean(train_counts))
    var = float(np.var(train_counts, ddof=1))
    vmr = var / mean if mean > 0 else np.nan

    if mean > 0:
        stat = float(np.sum((train_counts - mean) ** 2) / mean)
        pvalue = float(1 - sp_stats.chi2.cdf(stat, df=n - 1))
    else:
        stat, pvalue = np.nan, np.nan

    return {"vmr": vmr, "chi2_stat": stat, "chi2_pvalue": pvalue}


def lrt_poisson_negbin(ll_poisson: float, ll_negbin: float) -> tuple[float, float]:
    """
    Likelihood-ratio test of Poisson (nested, alpha=0) vs. Negative Binomial.
    alpha=0 sits on the boundary of the parameter space, so the usual chi2(1)
    reference is wrong; under Self & Liang (1987) the correct null
    distribution is a 50:50 mixture of chi2(0) (a point mass at 0) and
    chi2(1), which halves the naive chi2(1) p-value.
    """
    lr_stat = max(2 * (ll_negbin - ll_poisson), 0.0)
    pvalue = 0.5 * (1 - sp_stats.chi2.cdf(lr_stat, df=1))
    return lr_stat, pvalue


def fit_and_compare(train_counts, test_counts, player: str, stat: str) -> dict:
    """
    Fits both models on `train_counts`, and returns one record with every
    diagnostic needed for the summary table, plus the fitted parameters
    (private, underscore-prefixed keys) needed to redraw the fitted PMFs
    later in viz.py.
    """
    train_counts = np.asarray(train_counts, dtype=float)
    test_counts = np.asarray(test_counts, dtype=float)

    poisson_fit = fit_poisson(train_counts)
    negbin_fit = fit_negbin(train_counts)
    disp = dispersion_test(train_counts)
    lr_stat, lr_pvalue = lrt_poisson_negbin(poisson_fit["loglik"], negbin_fit["loglik"])

    winner_train = "NegBin" if negbin_fit["aic"] < poisson_fit["aic"] else "Poisson"

    winner_test = None
    test_nll_poisson = np.nan
    test_nll_negbin = np.nan
    if len(test_counts) > 0:
        test_nll_poisson = float(-_poisson_loglik(test_counts, poisson_fit["lambda"]))
        test_nll_negbin = float(-negbin_logpmf(test_counts, negbin_fit["mu"], negbin_fit["alpha"]).sum())
        winner_test = "NegBin" if test_nll_negbin < test_nll_poisson else "Poisson"

    return {
        "player": player,
        "stat": stat,
        "n_train": len(train_counts),
        "n_test": len(test_counts),
        "mean_train": float(train_counts.mean()),
        "var_train": float(train_counts.var(ddof=1)),
        "vmr": disp["vmr"],
        "dispersion_chi2_pvalue": disp["chi2_pvalue"],
        "poisson_lambda": poisson_fit["lambda"],
        "poisson_aic": poisson_fit["aic"],
        "negbin_mu": negbin_fit["mu"],
        "negbin_alpha": negbin_fit["alpha"],
        "negbin_aic": negbin_fit["aic"],
        "aic_diff_poisson_minus_negbin": poisson_fit["aic"] - negbin_fit["aic"],
        "lrt_stat": lr_stat,
        "lrt_pvalue": lr_pvalue,
        "winner_train_aic": winner_train,
        "test_nll_poisson": test_nll_poisson,
        "test_nll_negbin": test_nll_negbin,
        "winner_test": winner_test,
        "_train_counts": train_counts,
        "_test_counts": test_counts,
    }
