#!/usr/bin/env python3
# tools/start_cluster.py
"""
Spin up the Ray cluster, rsync the repo, and launch the driver.
"""
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLUSTER_YAML = ROOT / "tools" / "cluster.yaml"


def sh(cmd: str):
    print(">", cmd)
    subprocess.check_call(cmd, shell=True)


def main():
    sh(f"ray up {CLUSTER_YAML} -y")
    sh(f"ray rsync_up {CLUSTER_YAML} {ROOT} /home/ubuntu/app -r")
    time.sleep(5)
    sh(
        f'ray exec {CLUSTER_YAML} "cd /home/ubuntu/app && python -m brawl.gym.drivers.async_driver benchmark=webarena" --screen'
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
