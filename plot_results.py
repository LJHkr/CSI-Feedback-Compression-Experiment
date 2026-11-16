"""Plot the CSI feedback experiment results and seed-level confidence intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
T95 = 2.262  # Two-sided 95% Student-t interval, 10 independent seeds.
MATCHED_RHO = 0.85
SCAN_BUDGET = 64
METHODS = ["full_uniform", "pca_equal", "pca_unequal"]
LABELS = ["Full uniform", "PCA equal", "PCA unequal"]
COLORS = ["#777777", "#3976B9", "#CC6633"]
METRICS = ["nmse", "nmse_db", "rate", "rate_loss", "beam_agreement", "sharing_bits"]
REPORT_COUNTS = np.array([10, 100, 1000, 10000])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    half = T95 * values.std(ddof=1) / np.sqrt(len(values))
    return mean, half


def summarize(data, keys, sweep):
    rows = []
    for condition, group in data.groupby(keys):
        for metric in METRICS:
            mean, half = interval(group[metric])
            row = dict(zip(keys, condition))
            row.update(sweep=sweep, metric=metric, n=len(group), mean=mean,
                       ci95_half=half, ci95_low=mean-half, ci95_high=mean+half)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(data, keys, sweep, baselines=METHODS[:2],
                       column="method", target="pca_unequal"):
    rows = []
    for metric in ["nmse", "nmse_db", "rate", "rate_loss", "beam_agreement"]:
        wide = data.pivot(index=["seed"] + keys, columns=column, values=metric)
        for baseline in baselines:
            delta = (wide[target] - wide[baseline]).rename("delta").reset_index()
            for condition, group in delta.groupby(keys):
                mean, half = interval(group["delta"])
                row = dict(zip(keys, condition))
                row.update(sweep=sweep, contrast=target + " - " + baseline,
                           metric=metric, n=len(group), mean_difference=mean,
                           ci95_half=half, ci95_low=mean-half, ci95_high=mean+half)
                rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(RESULTS / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def sweep_plot(summary, x, xlabel, title, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, metric, ylabel in zip(axes, ["nmse_db", "rate_loss"],
                                 ["NMSE (dB)", "Rate loss (bit/s/Hz)"]):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            if method not in summary.method.unique():
                continue
            points = summary[(summary.method == method) & (summary.metric == metric)].sort_values(x)
            ax.errorbar(points[x], points["mean"], yerr=points.ci95_half,
                        color=color, label=label, marker="o", capsize=3, linewidth=1.6)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(sorted(summary[x].unique()))
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)
    fig.suptitle(title + "\nMean and 95% confidence interval over 10 seeds", fontsize=11)
    save_figure(fig, name)


def allocation_plot(allocations, runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    component = np.arange(1, 17)
    for j, budget in enumerate([32, 64, 128]):
        row = allocations[(allocations.seed == 0) & (allocations.budget == budget)
                          & (allocations.method == "pca_unequal")].iloc[0]
        bits = np.fromstring(row.real_bits, dtype=int, sep=" ")
        axes[0].bar(component + (j-1)*0.25, bits, width=0.25,
                    label=str(budget) + " bits", color=["#B8CCE2", "#5285B7", "#D58B60"][j])
    axes[0].set(xlabel="Complex PCA component", ylabel="Bits per real or imaginary part",
                title="Unequal allocation (seed 0)")
    axes[0].set_xticks(component)
    axes[0].legend(fontsize=9)
    matched = runs[(runs.test_rho == MATCHED_RHO) & (runs.budget == SCAN_BUDGET)]
    for method, label, color in zip(METHODS, LABELS, COLORS):
        cost = matched[matched.method == method].sharing_bits.mean()
        axes[1].semilogx(REPORT_COUNTS, cost/REPORT_COUNTS, marker="o", color=color, label=label)
    axes[1].set(xlabel="Number of CSI reports", ylabel="Downlink bits per report",
                title="Shared information cost / number of reports")
    axes[1].set_xticks(REPORT_COUNTS, ["10", "100", "1,000", "10,000"])
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    save_figure(fig, "allocation_and_sharing")


def objective_plot(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    x = np.arange(2)
    for ax, metric, ylabel in zip(axes, ["nmse_db", "rate_loss"],
                                 ["NMSE (dB)", "Rate loss (bit/s/Hz)"]):
        for j, selector in enumerate(["nmse", "rate"]):
            points = summary[(summary.test_rho == MATCHED_RHO) & (summary.metric == metric)
                             & (summary.selector == selector)].set_index("method").loc[METHODS[1:]]
            label = ["Selected by validation NMSE", "Selected by validation rate loss"][j]
            ax.errorbar(x + (j-0.5)*0.16, points["mean"], yerr=points.ci95_half,
                        fmt="o", capsize=4, color=COLORS[j+1], label=label)
        ax.set_xticks(x, LABELS[1:])
        ax.set(ylabel=ylabel, xlim=(-0.4, 1.4))
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Validation selection objective (64 feedback bits, rho = 0.85)\n"
                 "Mean and 95% confidence interval over 10 seeds", fontsize=11)
    save_figure(fig, "selection_objective")


def main():
    runs = pd.read_csv(RESULTS / "run_results.csv")
    dimensions = pd.read_csv(RESULTS / "dimension_results.csv")
    allocations = pd.read_csv(RESULTS / "allocations.csv")
    budget_summary = summarize(runs, ["test_rho", "budget", "method"], "budget")
    dimension_summary = summarize(dimensions, ["test_rho", "budget", "k", "method"], "dimension")
    pd.concat([budget_summary, dimension_summary], ignore_index=True).to_csv(RESULTS / "summary.csv", index=False)
    pairs = [paired_differences(runs, ["test_rho", "budget"], "budget"),
             paired_differences(dimensions, ["test_rho", "budget", "k"], "dimension", ["pca_equal"])]
    pd.concat(pairs, ignore_index=True).to_csv(RESULTS / "paired_differences.csv", index=False)
    sweep_plot(budget_summary[budget_summary.test_rho == MATCHED_RHO], "budget",
               "Feedback payload budget (bits)", "Matched channel statistics (rho = 0.85)", "budget")
    sweep_plot(dimension_summary, "k", "Retained complex PCA dimensions",
               "Dimension scan (64 feedback bits, rho = 0.85)", "dimensions")
    sweep_plot(budget_summary[budget_summary.budget == SCAN_BUDGET], "test_rho",
               "Test channel correlation (rho)", "Environment mismatch (train rho = 0.85, 64 bits)", "mismatch")
    allocation_plot(allocations, runs)
    objectives = pd.read_csv(RESULTS / "objective_results.csv")
    objective_summary = summarize(objectives, ["test_rho", "budget", "method", "selector"], "objective")
    objective_summary.to_csv(RESULTS / "objective_summary.csv", index=False)
    objective_pairs = paired_differences(objectives, ["test_rho", "budget", "method"],
                                         "objective", ["nmse"], column="selector", target="rate")
    objective_pairs.to_csv(RESULTS / "objective_paired.csv", index=False)
    objective_plot(objective_summary)


if __name__ == "__main__":
    main()
