"""Local orchestrator for the clean self-consistency run.

Manages remote execution on Colab VM session 'sc-clean'.
Ensures reboot-proofing: runs inference as detached background process on Colab,
periodically polls progress, and mirrors results back to results/ incrementally.
"""
from __future__ import annotations

import os
import sys
import time
import json
import base64
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
SESSION = "sc-clean"

# Ensure NO_PROXY is set so colab CLI can connect to Colab VM endpoints
os.environ["NO_PROXY"] = "colab.research.google.com,colab.dev,.colab.dev,googleapis.com,google.com"
os.environ["no_proxy"] = "colab.research.google.com,colab.dev,.colab.dev,googleapis.com,google.com"

def colab_exec(code: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    cmd = ["colab", "exec", "-s", SESSION, "--timeout", str(timeout)]
    return subprocess.run(cmd, input=code, text=True, capture_output=True, env=os.environ)

def upload_file(local_path: Path, remote_path: str):
    print(f"Uploading {local_path.name} to Colab VM at {remote_path}...", flush=True)
    b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    code = f"""
import base64, os
os.makedirs(os.path.dirname('{remote_path}'), exist_ok=True)
data = base64.b64decode('{b64}')
with open('{remote_path}', 'wb') as f:
    f.write(data)
"""
    res = colab_exec(code, timeout=60.0)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to upload {local_path}: {res.stderr}")

def main():
    print("=== Step 1: Check Colab session GPU ===", flush=True)
    res = colab_exec("import torch; print('GPU:', torch.cuda.get_device_name(0))", timeout=60.0)
    if res.returncode != 0:
        print("Error checking session:", res.stderr, flush=True)
        sys.exit(1)
    print(res.stdout.strip(), flush=True)

    print("\n=== Step 2: Upload experiment files ===", flush=True)
    upload_file(SCRIPTS_DIR / "parser.py", "/content/sc-clean/parser.py")
    upload_file(SCRIPTS_DIR / "colab_runner.py", "/content/sc-clean/colab_runner.py")
    upload_file(RESULTS_DIR / "pilot_questions.json", "/content/sc-clean/pilot_questions.json")
    upload_file(RESULTS_DIR / "confirmatory_questions.json", "/content/sc-clean/confirmatory_questions.json")

    print("\n=== Step 3: Launch detached runner on Colab VM ===", flush=True)
    launch_code = """
import subprocess, os, sys
from pathlib import Path
R = Path('/content/sc-clean')
pid_file = R / 'runner.pid'
alive = False
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        if stat[2] != 'Z':
            alive = True
    except (ProcessLookupError, FileNotFoundError, ValueError):
        pass

if not alive:
    log_file = (R / 'runner.log').open('a')
    p = subprocess.Popen(
        ['python3', '-u', str(R / 'colab_runner.py')],
        cwd=str(R),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    pid_file.write_text(str(p.pid))
    print('LAUNCHED_PID:', p.pid)
else:
    print('ALREADY_RUNNING_PID:', pid_file.read_text().strip())
"""
    res = colab_exec(launch_code, timeout=60.0)
    print("Launch response:", res.stdout.strip(), flush=True)

    print("\n=== Step 4: Monitor progress & mirror results incrementally ===", flush=True)
    t_start = time.time()
    downloaded_batches = set()
    last_log_line = ""

    while True:
        poll_code = """
import json, os
from pathlib import Path
R = Path('/content/sc-clean')
pid_file = R / 'runner.pid'
alive = False
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        if stat[2] != 'Z':
            alive = True
    except Exception:
        pass

status = {
    'alive': alive,
    'complete': (R / 'complete.txt').exists(),
    'error': (R / 'error.txt').exists(),
    'pilot_done': (R / 'pilot_summary.json').exists(),
    'progress': json.loads((R / 'progress.json').read_text()) if (R / 'progress.json').exists() else None,
    'files': [p.name for p in sorted(R.glob('b*.jsonl'))]
}
if (R / 'error.txt').exists():
    status['error_msg'] = (R / 'error.txt').read_text()
if (R / 'runner.log').exists():
    lines = (R / 'runner.log').read_text().splitlines()
    status['tail'] = lines[-10:] if lines else []
print(json.dumps(status))
"""
        res = colab_exec(poll_code, timeout=45.0)
        if res.returncode == 0:
            try:
                status = json.loads(res.stdout.strip())
                if status.get("error"):
                    print("\n[!] Error detected on remote runner:", flush=True)
                    print(status.get("error_msg", "Unknown error"), flush=True)
                    break

                tail = status.get("tail", [])
                for line in tail:
                    if line != last_log_line:
                        print(f"[{time.strftime('%X')}] {line}", flush=True)
                        last_log_line = line

                # If pilot just finished and not mirrored yet
                if status.get("pilot_done") and not (RESULTS_DIR / "pilot_summary.json").exists():
                    p_code = "import base64; print(base64.b64encode(open('/content/sc-clean/pilot_summary.json', 'rb').read()).decode('ascii'))"
                    p_res = colab_exec(p_code, timeout=30.0)
                    if p_res.returncode == 0:
                        (RESULTS_DIR / "pilot_summary.json").write_bytes(base64.b64decode(p_res.stdout.strip()))
                        print(">>> Mirrored pilot_summary.json to local results/", flush=True)

                # Mirror batch files
                for f_name in status.get("files", []):
                    dl_code = f"import base64; print(base64.b64encode(open('/content/sc-clean/{f_name}', 'rb').read()).decode('ascii'))"
                    dl_res = colab_exec(dl_code, timeout=30.0)
                    if dl_res.returncode == 0 and dl_res.stdout.strip():
                        (RESULTS_DIR / f_name).write_bytes(base64.b64decode(dl_res.stdout.strip()))
                        if f_name not in downloaded_batches:
                            downloaded_batches.add(f_name)
                            print(f">>> Mirrored {f_name} to local results/", flush=True)

                if status.get("complete"):
                    print("\n[+] Confirmatory run is complete on remote VM!", flush=True)
                    break

                if not status.get("alive") and not status.get("complete"):
                    print("\n[!] Runner process exited unexpectedly without complete.txt!", flush=True)
                    break

            except Exception as e:
                print("Polling parse error:", e, flush=True)
        else:
            print("Poll request failed:", res.stderr, flush=True)

        time.sleep(15)

    print("\n=== Step 5: Mirror all artifacts from VM to results/ ===", flush=True)
    download_files = [
        "metadata.json", "pilot_summary.json", "pilot_results.json",
        "progress.json", "runner.log", "complete.txt"
    ]
    list_code = "import json; from pathlib import Path; print(json.dumps([p.name for p in Path('/content/sc-clean').glob('*.jsonl')]))"
    res = colab_exec(list_code, timeout=30.0)
    if res.returncode == 0:
        jsonl_files = json.loads(res.stdout.strip())
        download_files.extend(jsonl_files)

    for fname in sorted(set(download_files)):
        fetch_code = f"""
import base64, os
from pathlib import Path
p = Path('/content/sc-clean/{fname}')
if p.exists():
    print(base64.b64encode(p.read_bytes()).decode('ascii'))
else:
    print('NOT_FOUND')
"""
        res = colab_exec(fetch_code, timeout=60.0)
        out = res.stdout.strip()
        if out and out != "NOT_FOUND":
            (RESULTS_DIR / fname).write_bytes(base64.b64decode(out))
            print(f"Downloaded {fname} ({os.path.getsize(RESULTS_DIR / fname)} bytes)", flush=True)

    print("\nAll artifacts mirrored successfully to results/!", flush=True)

if __name__ == "__main__":
    main()
