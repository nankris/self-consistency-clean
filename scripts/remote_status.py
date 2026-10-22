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
    status['tail'] = lines[-15:] if lines else []
print(json.dumps(status))
