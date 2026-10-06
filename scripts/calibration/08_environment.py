"""
08_environment.py
=================
Records the software environment and repository state of a pipeline run to
results/calibration/run_log/environment.json (called first by `make all`).
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKGS = ["numpy", "scipy", "pandas", "pyarrow", "h5py", "statsmodels", "patsy", "zetapy", "scikit-learn",
        "matplotlib", "pytest"]


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    env = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, platform=platform.platform(), machine=platform.machine(),
        packages={p: (metadata.version(p) if _has(p) else None) for p in PKGS},
        git_commit=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        data_source="s3://allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/ (NWB read with h5py)",
    )
    pinned = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            k, v = line.split("==")
            pinned[k.strip()] = v.strip()
    env["mismatches_vs_requirements"] = {k: (env["packages"].get(k), v) for k, v in pinned.items()
                                         if env["packages"].get(k) != v}
    out = ROOT / "results/calibration/run_log"
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))


def _has(p):
    try:
        metadata.version(p)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
