"""Local monitor script for Experiment 035b.

Polls Colab runner via fast colab download of logs and progress artifacts,
streams status, and mirrors results back to local results/ incrementally.
"""
from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SESSION = "b10-035b"

os.environ["NO_PROXY"] = "colab.research.google.com,colab.dev,.colab.dev,googleapis.com,google.com"
os.environ["no_proxy"] = "colab.research.google.com,colab.dev,.colab.dev,googleapis.com,google.com"

def download(remote_path: str, local_path: Path) -> bool:
    cmd = ["colab", "download", "-s", SESSION, remote_path, str(local_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0, env=os.environ)
    return res.returncode == 0

def main():
    print(f"[{time.strftime('%X')}] Starting robust file-based monitor for session {SESSION}...", flush=True)
    downloaded_batches = set()
    last_log_size = 0
    pilot_announced = False

    while True:
        try:
            # 1. Download runner.log
            log_file = RESULTS_DIR / "runner.log"
            if download("/content/b10-035b/runner.log", log_file):
                current_size = log_file.stat().st_size
                if current_size > last_log_size:
                    lines = log_file.read_text(errors="replace").splitlines()
                    for line in lines[-10:]:
                        print(f"[{time.strftime('%X')}] [VM] {line}", flush=True)
                    last_log_size = current_size

            # 2. Check pilot summary
            if not pilot_announced:
                p_file = RESULTS_DIR / "pilot_summary.json"
                if download("/content/b10-035b/pilot_summary.json", p_file):
                    try:
                        p_data = json.loads(p_file.read_text())
                        print(f"\n==========================================", flush=True)
                        print(f">>> PILOT GATE RESULT:", flush=True)
                        print(f"    Passed: {p_data['passed']}", flush=True)
                        print(f"    Greedy parse fail rate: {p_data['greedy_parse_fail_rate']:.1%}", flush=True)
                        print(f"    Stochastic parse fail rate: {p_data['stochastic_parse_fail_rate']:.1%}", flush=True)
                        print(f"    Duration: {p_data['pilot_duration_seconds']:.1f}s", flush=True)
                        print(f"==========================================\n", flush=True)
                        pilot_announced = True
                    except Exception:
                        pass

            # 3. Check progress & download batches
            for b_idx in range(0, 500, 50):
                b_name = f"b{b_idx:04d}.jsonl"
                b_path = RESULTS_DIR / b_name
                if b_name not in downloaded_batches:
                    if download(f"/content/b10-035b/{b_name}", b_path):
                        lines = [l for l in b_path.read_text().splitlines() if l.strip()]
                        if len(lines) >= 50:
                            downloaded_batches.add(b_name)
                            print(f">>> Completed & mirrored batch {b_name} ({len(lines)} items, {b_path.stat().st_size} bytes)", flush=True)
                        elif len(lines) > 0:
                            print(f">>> In-progress batch {b_name} ({len(lines)}/50 items)", flush=True)

            # 4. Check completion
            comp_file = RESULTS_DIR / "complete.txt"
            if download("/content/b10-035b/complete.txt", comp_file):
                print(f"\n[{time.strftime('%X')}] REMOTE RUN IS FULLY COMPLETE!", flush=True)
                break

            # 5. Check error
            err_file = RESULTS_DIR / "error.txt"
            if download("/content/b10-035b/error.txt", err_file):
                print(f"\n[{time.strftime('%X')}] ERROR on remote VM:", flush=True)
                print(err_file.read_text(), flush=True)
                break

        except Exception as e:
            print(f"[{time.strftime('%X')}] Monitor poll warning: {type(e).__name__}: {e}", flush=True)

        time.sleep(15)

    # Mirror remaining artifacts
    print(f"\n[{time.strftime('%X')}] Downloading final metadata and all batches...", flush=True)
    for extra in ["metadata.json", "pilot_results.json", "progress.json"]:
        download(f"/content/b10-035b/{extra}", RESULTS_DIR / extra)
        
    for b_idx in range(0, 500, 50):
        b_name = f"b{b_idx:04d}.jsonl"
        download(f"/content/b10-035b/{b_name}", RESULTS_DIR / b_name)

    print(f"[{time.strftime('%X')}] All artifacts mirrored successfully to results/!", flush=True)

if __name__ == "__main__":
    main()
