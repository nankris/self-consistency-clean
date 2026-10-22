"""Prepare fresh GSM8K splits for pilot and confirmatory runs.

Guarantees zero overlap with the questions used in the earlier runs of this
study (the self-consistency-voting repo). Point PRIOR_RUNS_DIR at a directory
holding one sub-folder per earlier run, each with results/questions.json.
"""
import json
import os
from pathlib import Path
from datasets import load_dataset

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading GSM8K test and train datasets...")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    ds_train = load_dataset("openai/gsm8k", "main", split="train")

    used_questions = set()
    used_indices = set()

    prior_dir = Path(os.environ.get("PRIOR_RUNS_DIR", "prior_runs"))
    prior_files = sorted(prior_dir.glob("*/results/questions.json"))
    if len(prior_files) != 4:
        raise SystemExit(f"Expected the 4 earlier runs' results/questions.json under {prior_dir}, found {len(prior_files)}; set PRIOR_RUNS_DIR to a self-consistency-voting checkout.")
    for q_file in prior_files:
        if q_file.exists():
            data = json.loads(q_file.read_text())
            for item in data:
                used_questions.add(item["question"].strip())
                if "pos" in item:
                    used_indices.add(item["pos"])
                if "index" in item:
                    used_indices.add(item["index"])
                if "idx" in item:
                    used_indices.add(item["idx"])

    # Match against ds_test
    matched_test_indices = set()
    for idx, item in enumerate(ds_test):
        if item["question"].strip() in used_questions:
            matched_test_indices.add(idx)

    print(f"Total GSM8K test set items: {len(ds_test)}")
    print(f"Total questions used in prior runs: {len(used_questions)}")
    print(f"Matched test items previously used: {len(matched_test_indices)}")

    fresh_test_indices = [i for i in range(len(ds_test)) if i not in matched_test_indices]
    print(f"Available fresh test items: {len(fresh_test_indices)}")
    assert len(fresh_test_indices) >= 500, f"Expected at least 500 fresh items, found {len(fresh_test_indices)}"

    # Select exactly 500 fresh test items for confirmatory evaluation
    confirmatory_indices = fresh_test_indices[:500]
    confirmatory_data = []
    for rank, idx in enumerate(confirmatory_indices):
        item = ds_test[idx]
        gold_raw = item["answer"].split("####")[-1].strip()
        confirmatory_data.append({
            "confirmatory_id": rank,
            "gsm8k_test_idx": idx,
            "question": item["question"].strip(),
            "answer": item["answer"].strip(),
            "gold": gold_raw
        })

    # Pilot: first 50 items from GSM8K train split
    pilot_data = []
    for idx in range(50):
        item = ds_train[idx]
        gold_raw = item["answer"].split("####")[-1].strip()
        pilot_data.append({
            "pilot_id": idx,
            "gsm8k_train_idx": idx,
            "question": item["question"].strip(),
            "answer": item["answer"].strip(),
            "gold": gold_raw
        })

    (RESULTS_DIR / "confirmatory_questions.json").write_text(json.dumps(confirmatory_data, indent=2))
    (RESULTS_DIR / "pilot_questions.json").write_text(json.dumps(pilot_data, indent=2))
    
    # Save provenance
    provenance = {
        "total_test_items": len(ds_test),
        "prior_used_count": len(matched_test_indices),
        "available_fresh_count": len(fresh_test_indices),
        "confirmatory_count": len(confirmatory_data),
        "confirmatory_test_indices": confirmatory_indices,
        "pilot_count": len(pilot_data),
        "pilot_split": "train[0:50]"
    }
    (RESULTS_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Successfully saved {len(confirmatory_data)} fresh confirmatory questions and {len(pilot_data)} pilot questions.")

if __name__ == "__main__":
    main()
