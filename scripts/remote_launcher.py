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
    except Exception:
        pass

if not alive:
    log_file = open(R / 'runner.log', 'a')
    p = subprocess.Popen(
        ['python3', '-u', str(R / 'colab_runner.py')],
        cwd=str(R),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    pid_file.write_text(str(p.pid))
    print(f'LAUNCHED_RUNNER_PID: {p.pid}')
else:
    print(f'ALREADY_RUNNING_PID: {pid_file.read_text().strip()}')
