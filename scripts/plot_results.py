"""Generate publication-ready figures for this experiment.

Creates:
1. figures/accuracy_vs_k.png (Accuracy curve across k=1, 3, 5, 8 vs greedy with 95% CIs)
2. figures/delta_bootstrap.png (Paired difference bootstrap distribution with margin thresholds)
3. figures/parse_failure_rate.png (Parse failure rate comparison: earlier prompt vs clean prompt)
4. figures/table.png (Main results table rendered as an image for Substack)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set clean publication styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#adb5bd"
plt.rcParams["axes.linewidth"] = 0.8

def plot_all():
    summary_file = RESULTS_DIR / "summary.json"
    if not summary_file.exists():
        print("summary.json not found yet. Run analyze.py first.")
        return

    summary = json.loads(summary_file.read_text())
    greedy = summary["greedy"]
    curves = summary["curves"]
    n_q = summary["n_questions"]
    verdict = summary["verdict"]

    # ==============================================================
    # 1. figures/accuracy_vs_k.png
    # ==============================================================
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    
    k_vals = [1, 3, 5, 8]
    accs = [curves[str(k)]["accuracy"] * 100 for k in k_vals]
    cis = [curves[str(k)]["ci95"] for k in k_vals]
    yerr_lower = [accs[i] - cis[i][0] * 100 for i in range(len(k_vals))]
    yerr_upper = [cis[i][1] * 100 - accs[i] for i in range(len(k_vals))]

    # Plot Greedy baseline
    g_acc = greedy["accuracy"] * 100
    g_ci = [greedy["ci95"][0] * 100, greedy["ci95"][1] * 100]
    ax.axhline(g_acc, color="#c92a2a", linestyle="--", linewidth=1.5, label=f"Greedy Baseline: {g_acc:.1f}%")
    ax.axhspan(g_ci[0], g_ci[1], color="#ffe3e3", alpha=0.5, label="Greedy 95% Bootstrap CI")

    # Plot SC curve
    ax.errorbar(
        k_vals, accs, yerr=[yerr_lower, yerr_upper],
        fmt="-o", color="#1971c2", ecolor="#1971c2", elinewidth=2,
        capsize=5, capthick=1.5, markersize=7, linewidth=2,
        label="Majority Vote (SC@k)"
    )

    for k, acc in zip(k_vals, accs):
        ax.annotate(
            f"{acc:.1f}%", (k, acc),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9, fontweight="bold", color="#1864ab"
        )

    ax.set_title(f"GSM8K Accuracy vs Sample Budget (Qwen3-1.7B, n={n_q} Fresh Items)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Samples in Majority Vote (k)", fontsize=10, labelpad=8)
    ax.set_ylabel("Exact-Match Accuracy (%)", fontsize=10, labelpad=8)
    ax.set_xticks(k_vals)
    ax.set_ylim(min(g_acc, min(accs)) - 4, max(accs) + 5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da", fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_vs_k.png")
    plt.close(fig)
    print("Saved figures/accuracy_vs_k.png")

    # ==============================================================
    # 2. figures/delta_bootstrap.png
    # ==============================================================
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    
    # Reload bootstrap samples from scored.jsonl to plot empirical distribution
    scored_file = RESULTS_DIR / "scored.jsonl"
    scored_lines = [json.loads(l) for l in scored_file.read_text().splitlines() if l.strip()]
    g_arr = np.array([l["greedy_correct"] for l in scored_lines])
    v8_arr = np.array([l["vote_correct"]["8"] for l in scored_lines])
    delta_arr = v8_arr - g_arr
    
    rng = np.random.default_rng(35)
    boot_idx = rng.integers(0, len(delta_arr), size=(10000, len(delta_arr)))
    boot_deltas = delta_arr[boot_idx].mean(axis=1) * 100

    d_mean = curves["8"]["delta"] * 100
    ci_lo = curves["8"]["delta_ci95"][0] * 100
    ci_hi = curves["8"]["delta_ci95"][1] * 100

    counts, bins, patches = ax.hist(boot_deltas, bins=35, color="#74c0fc", edgecolor="#1971c2", alpha=0.7, density=True)
    
    # Vertical markers
    ax.axvline(0, color="#495057", linestyle="-", linewidth=1.5, label="Null (Δ = 0)")
    ax.axvline(2.0, color="#f59f00", linestyle=":", linewidth=1.5, label="Lower Bound Gate (+2.0 pp)")
    ax.axvline(5.0, color="#2b8a3e", linestyle="--", linewidth=1.8, label="Superiority Target (+5.0 pp)")
    
    # 95% CI bracket
    ax.errorbar([d_mean], [max(counts) * 0.85], xerr=[[d_mean - ci_lo], [ci_hi - d_mean]],
                fmt="s", color="#d9480f", ecolor="#d9480f", elinewidth=2.5, capsize=6, capthick=2,
                label=f"Observed Δ = {d_mean:+.2f} pp [95% CI {ci_lo:.2f}, {ci_hi:.2f}]")

    ax.set_title(f"Bootstrap Distribution of Paired Gain: SC@8 vs Greedy ({verdict})", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Paired Accuracy Difference (Percentage Points)", fontsize=10, labelpad=8)
    ax.set_ylabel("Probability Density", fontsize=10, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da", fontsize=8.5, loc="upper right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delta_bootstrap.png")
    plt.close(fig)
    print("Saved figures/delta_bootstrap.png")

    # ==============================================================
    # 3. figures/parse_failure_rate.png
    # ==============================================================
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    
    categories = ["Earlier run (Flawed Prompt)\n'Answer: NUMBER'", "This run (Clean Prompt)\n'The answer is [number]'"]
    greedy_fails = [55.75, greedy["parse_fail_rate"] * 100]
    stoch_fails = [55.80, summary["stochastic_overall"]["parse_fail_rate"] * 100]

    x = np.arange(len(categories))
    width = 0.32

    rects1 = ax.bar(x - width/2, greedy_fails, width, label="Greedy Decoding (T=0)", color="#fa5252", edgecolor="#c92a2a")
    rects2 = ax.bar(x + width/2, stoch_fails, width, label="Stochastic Samples (T=0.7)", color="#4dabf7", edgecolor="#1971c2")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(2.0, color="#d9480f", linestyle="--", linewidth=1.2, label="Pre-Registered Pilot Gate (< 2.0%)")
    ax.set_title("Parse Failure Rate: Earlier Prompt Artifact vs Clean Extractor", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Parse Failure Rate (%)", fontsize=10, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylim(0, 68)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "parse_failure_rate.png")
    plt.close(fig)
    print("Saved figures/parse_failure_rate.png")

    # ==============================================================
    # 4. figures/table.png (Main results table rendered as an image)
    # ==============================================================
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=150)
    ax.axis("off")

    table_data = [
        ["Method / Arm", "Samples (k)", "Correct", "Accuracy", "95% Bootstrap CI", "Paired Δ vs Greedy", "Parse Failures"],
        ["Greedy (T=0)", "1", f"{greedy['correct']}/{n_q}", f"{greedy['accuracy']*100:.2f}%", f"[{greedy['ci95'][0]*100:.1f}, {greedy['ci95'][1]*100:.1f}]%", "Baseline", f"{greedy['parse_fails']}/{n_q} ({greedy['parse_fail_rate']:.1%})"],
        ["Vote@1 (T=0.7)", "1", f"{round(curves['1']['accuracy']*n_q)}/{n_q}", f"{curves['1']['accuracy']*100:.2f}%", f"[{curves['1']['ci95'][0]*100:.1f}, {curves['1']['ci95'][1]*100:.1f}]%", f"{curves['1']['delta']*100:+.2f} pp", f"< 1.0%"],
        ["Vote@3 (T=0.7)", "3", f"{round(curves['3']['accuracy']*n_q)}/{n_q}", f"{curves['3']['accuracy']*100:.2f}%", f"[{curves['3']['ci95'][0]*100:.1f}, {curves['3']['ci95'][1]*100:.1f}]%", f"{curves['3']['delta']*100:+.2f} pp", f"< 1.0%"],
        ["Vote@5 (T=0.7)", "5", f"{round(curves['5']['accuracy']*n_q)}/{n_q}", f"{curves['5']['accuracy']*100:.2f}%", f"[{curves['5']['ci95'][0]*100:.1f}, {curves['5']['ci95'][1]*100:.1f}]%", f"{curves['5']['delta']*100:+.2f} pp", f"< 1.0%"],
        ["Vote@8 (T=0.7)", "8", f"{round(curves['8']['accuracy']*n_q)}/{n_q}", f"{curves['8']['accuracy']*100:.2f}%", f"[{curves['8']['ci95'][0]*100:.1f}, {curves['8']['ci95'][1]*100:.1f}]%", f"{curves['8']['delta']*100:+.2f} pp", f"< 1.0%"],
    ]

    col_widths = [0.22, 0.12, 0.12, 0.13, 0.19, 0.20, 0.18]
    table = ax.table(
        cellText=table_data,
        colWidths=col_widths,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.05, 1.6)

    # Style header row
    for j in range(len(table_data[0])):
        cell = table[0, j]
        cell.set_facecolor("#1971c2")
        cell.set_text_props(color="white", weight="bold")
    
    # Style alternating rows
    for i in range(1, len(table_data)):
        bg_color = "#f8f9fa" if i % 2 == 1 else "#ffffff"
        if i == len(table_data) - 1:  # Vote@8 primary row
            bg_color = "#e7f5ff"
        for j in range(len(table_data[0])):
            cell = table[i, j]
            cell.set_facecolor(bg_color)
            if i == len(table_data) - 1:
                cell.set_text_props(weight="bold")

    ax.set_title(f"Confirmatory Evaluation: Qwen3-1.7B on GSM8K (n={n_q})\nVerdict: {verdict}",
                 fontsize=11, fontweight="bold", pad=15)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "table.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/table.png")

if __name__ == "__main__":
    plot_all()
