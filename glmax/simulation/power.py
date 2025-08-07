from scipy.stats import binomtest
import seaborn as sns
import pandas as pd
import numpy as np


def power_binomial(n_samples, failure_rate, failure_threshold,
                   alternative="less", p_threshold=0.05,
                   plot=True, n_sims=100000):
    """
    Power analysis for whether a failure rate is
    significantly less or greater or different than a threshold.
    """
    results = {}
    n_samples_post_dropout = [i * (1 - dropout) for i in n_samples]
    for n in n_samples:
        results[n] = []
        rng = np.random.default_rng(seed=None)
        for i in np.arange(n_sims):
            fail = sum(rng.binomial(1, failure_rate, n))  # failures
            if p_threshold == "counts":  # significance = failure < threshold
                p_value = 0 if fail < failure_threshold * n else 1
            else:  # significance based on actual p-value statistic
                p_value = binomtest(fail, n, p=failure_threshold,
                                    alternative=alternative).pvalue
            results[n] += [pd.Series([p_value, fail], index=["p", "fails"])]
        results[n] = pd.concat(results[n], keys=np.arange(1, n_sims + 1),
                               names=["Simulation"])
    results = pd.concat(results, keys=n_samples, names=["N"]).unstack(-1)
    if dropout != 0:
        results.loc[:, "N_Post_Dropout"] = results.reset_index().replace(
            {"N": dict(zip(n_samples, n_samples_post_dropout))})["N"]
    results.loc[:, "significant"] = results["p"] == 0 if (
        p_threshold == "counts") else results["p"] < p_threshold
    power = results.groupby("N").mean()["significant"].to_frame("Power")
    power = power.assign(**{"Failure Rate": failure_rate}).assign(**{
        "Failure Threshold": failure_threshold, "P Threshold": p_threshold})
    if plot is True:
        fig = sns.relplot(x="N", y="Power", data=power, kind="line")
        fig.set_axis_labels("N" if dropout == 0 else (
            f"N (Post-{dropout * 100}% Dropout)"), "Power")
    return power, results


def power_binomial_sensitivity(n_samples, failure_rate, failure_threshold,
                               alternative="less", p_threshold=0.05,
                               dropout=0, plot=True,
                               ylim=None, n_sims=100000):
    """Run power analysis for multiple failure rates and thresholds."""
    failure_threshold = [failure_threshold] if isinstance(
        failure_threshold, (int, float)) else failure_threshold
    results = pd.concat([pd.concat([power_binomial(
        n_samples, r, f, alternative=alternative, p_threshold=p_threshold,
        plot=False, n_sims=n_sims)[1] for f in failure_threshold],
                                 keys=failure_threshold,
                                 names=["Failure Threshold"])
                       for r in failure_rate], keys=failure_rate,
                      names=["Failure Rate"])
    results.loc[:, "significant"] = results["p"] == 0 if (
        p_threshold == "counts") else results["p"] < p_threshold
    power = (results.groupby(["Failure Rate", "Failure Threshold", "N"]).mean(
        )["significant"]).to_frame("Power")
    if plot is True:
        fig = sns.relplot(
            x="N", y="Power", data=power, col="Failure Threshold",
            hue="Failure Rate", kind="line", palette="tab20")
        fig.set(ylim=ylim if ylim else (0, 1))
        fig.set_axis_labels("N" if dropout == 0 else (
            f"N (Post-{dropout * 100}% Dropout)"), "Power")
    power = power[power.columns.difference(power.index.names)]
    return power, results
