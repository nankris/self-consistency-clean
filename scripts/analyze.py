"""Confirmatory analysis for the clean self-consistency experiment.

Evaluates GSM8K greedy decoding vs Self-Consistency majority voting
at k in {1, 3, 5, 8} samples on 500 fresh test questions.
Performs 10,000 paired bootstrap resamples, calculates CIs,
and tests pre-registered decision rules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / "scripts"))
from parser import canon, parse_answer, vote

def run_analysis():
    # 1. Load batches
    rows = []
    for f in sorted(RESULTS_DIR.glob("b*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    assert len(rows) >= 500, f"Expected at least 500 rows, found {len(rows)}"
    rows = rows[:500]
    n = len(rows)
    print(f"Loaded {n} confirmatory questions.")

    # 2. Extract answers and check correctness
    gold_answers = [canon(r["gold"]) for r in rows]
    
    greedy_parsed = [parse_answer(r["greedy"]["text"]) for r in rows]
    greedy_correct = np.array([p == g for p, g in zip(greedy_parsed, gold_answers)], dtype=int)
    greedy_parse_fails = sum(p is None for p in greedy_parsed)

    # Samples: 8 samples per question
    samples_parsed = []
    total_stoch_samples = 0
    stoch_parse_fails = 0
    for r in rows:
        row_p = [parse_answer(s["text"]) for s in r["samples"]]
        samples_parsed.append(row_p)
        total_stoch_samples += len(row_p)
        stoch_parse_fails += sum(p is None for p in row_p)

    # Compute votes for k in [1, 3, 5, 8]
    k_values = [1, 3, 5, 8]
    vote_results = {}
    for k in k_values:
        v_list = [vote(row_p, k) for row_p in samples_parsed]
        v_correct = np.array([v == g for v, g in zip(v_list, gold_answers)], dtype=int)
        v_fails = sum(v is None for v in v_list)
        vote_results[k] = {
            "votes": v_list,
            "correct": v_correct,
            "parse_fails": v_fails,
            "accuracy": float(v_correct.mean()),
            "count": int(v_correct.sum())
        }

    # 3. Bootstrap CI (10,000 paired resamples)
    rng = np.random.default_rng(35)
    boot_indices = rng.integers(0, n, size=(10000, n))

    greedy_boot = greedy_correct[boot_indices].mean(axis=1)
    greedy_ci = np.quantile(greedy_boot, [0.025, 0.975])

    for k in k_values:
        v_corr = vote_results[k]["correct"]
        v_boot = v_corr[boot_indices].mean(axis=1)
        vote_results[k]["ci95"] = np.quantile(v_boot, [0.025, 0.975]).tolist()
        
        # Paired delta: Vote@k - Greedy
        delta = v_corr - greedy_correct
        delta_boot = delta[boot_indices].mean(axis=1)
        vote_results[k]["delta_mean"] = float(delta.mean())
        vote_results[k]["delta_ci95"] = np.quantile(delta_boot, [0.025, 0.975]).tolist()
        vote_results[k]["delta_ci90"] = np.quantile(delta_boot, [0.05, 0.95]).tolist()
        
        # McNemar test
        b = int(((v_corr == 1) & (greedy_correct == 0)).sum())
        c = int(((v_corr == 0) & (greedy_correct == 1)).sum())
        vote_results[k]["mcnemar_b"] = b
        vote_results[k]["mcnemar_c"] = c
        vote_results[k]["mcnemar_p"] = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0

    # 4. Decision Rule on Primary Endpoint (SC@8 vs Greedy)
    primary = vote_results[8]
    delta_8 = primary["delta_mean"]
    ci_8_lo, ci_8_hi = primary["delta_ci95"]
    greedy_fail_rate = greedy_parse_fails / n
    stoch_fail_rate = stoch_parse_fails / total_stoch_samples

    passes_parse_gate = (greedy_fail_rate < 0.02) and (stoch_fail_rate < 0.02)
    
    if delta_8 >= 0.05 and ci_8_lo > 0.02 and passes_parse_gate:
        verdict = "PROVEN"
        verdict_detail = f"Superiority proven: SC@8 delivers {delta_8*100:+.2f} pp gain (95% CI [{ci_8_lo*100:.2f}, {ci_8_hi*100:.2f}] pp), exceeding the +5.0 pp target with CI lower bound > +2.0 pp and parse failures < 2%."
    elif ci_8_hi <= 0.05 or ci_8_lo <= 0.0:
        verdict = "DISPROVEN"
        verdict_detail = f"Disproven: SC@8 gain {delta_8*100:+.2f} pp (95% CI [{ci_8_lo*100:.2f}, {ci_8_hi*100:.2f}] pp) fails to reliably clear the pre-registered +5.0 pp threshold."
    else:
        verdict = "INCONCLUSIVE"
        verdict_detail = f"Inconclusive: point estimate {delta_8*100:+.2f} pp overlaps margin or bounds."

    # 5. Token counts and generation stats
    greedy_tokens = [r["greedy"]["tokens"] for r in rows]
    sample_tokens = [s["tokens"] for r in rows for s in r["samples"]]

    summary = {
        "n_questions": n,
        "model": "Qwen/Qwen3-1.7B",
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "greedy": {
            "correct": int(greedy_correct.sum()),
            "accuracy": float(greedy_correct.mean()),
            "ci95": greedy_ci.tolist(),
            "parse_fails": greedy_parse_fails,
            "parse_fail_rate": greedy_fail_rate,
            "mean_tokens": float(np.mean(greedy_tokens))
        },
        "stochastic_overall": {
            "total_samples": total_stoch_samples,
            "parse_fails": stoch_parse_fails,
            "parse_fail_rate": stoch_fail_rate,
            "mean_tokens": float(np.mean(sample_tokens))
        },
        "curves": {
            k: {
                "accuracy": vote_results[k]["accuracy"],
                "ci95": vote_results[k]["ci95"],
                "delta": vote_results[k]["delta_mean"],
                "delta_ci95": vote_results[k]["delta_ci95"],
                "delta_ci90": vote_results[k]["delta_ci90"],
                "vote_only": vote_results[k]["mcnemar_b"],
                "greedy_only": vote_results[k]["mcnemar_c"],
                "mcnemar_p": vote_results[k]["mcnemar_p"]
            }
            for k in k_values
        }
    }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # 6. Save scored jsonl
    with (RESULTS_DIR / "scored.jsonl").open("w") as f:
        for i in range(n):
            item = {
                "confirmatory_id": rows[i]["confirmatory_id"],
                "gsm8k_test_idx": rows[i]["gsm8k_test_idx"],
                "gold": gold_answers[i],
                "greedy_parsed": greedy_parsed[i],
                "greedy_correct": int(greedy_correct[i]),
                "votes": {k: vote_results[k]["votes"][i] for k in k_values},
                "vote_correct": {k: int(vote_results[k]["correct"][i]) for k in k_values}
            }
            f.write(json.dumps(item) + "\n")

    return summary

if __name__ == "__main__":
    run_analysis()
