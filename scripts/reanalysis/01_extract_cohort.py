"""
01_extract_cohort.py
====================
Stream-and-purge extraction of all 28 cohort sessions from the public Allen S3
bucket. Each NWB is downloaded (with one session prefetched), reduced with
src/reanalysis/extract_nwb.py to data/derived/session_<id>.pkl.gz, and deleted.

    python scripts/reanalysis/01_extract_cohort.py [--nwb-dir /path] [--keep]
"""

import argparse
import gzip
import logging
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reanalysis.extract_nwb import process_session  # noqa: E402

URL = ("https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/"
       "ecephys-cache/session_{sid}/session_{sid}.nwb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("extract")


def download(sid: int, nwb_dir: Path) -> Path:
    out = nwb_dir / f"session_{sid}.nwb"
    if out.exists() and out.stat().st_size > 1e8:
        return out
    tmp = out.with_suffix(".part")
    for attempt in range(5):
        r = subprocess.run(["curl", "-sS", "--fail", "-o", str(tmp), URL.format(sid=sid)])
        if r.returncode == 0:
            tmp.rename(out)
            return out
        time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed for {sid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb-dir", default="/home/user/nwb")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    nwb_dir = Path(args.nwb_dir)
    nwb_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT / "results/cohort/full_28_specimen_manifest.csv")
    sids = [int(s) for s in manifest["session_id"]]
    todo = [s for s in sids if not (out_dir / f"session_{s}.pkl.gz").exists()]
    log.info("%d sessions to extract", len(todo))

    prefetch = {}

    def fetch(s):
        try:
            prefetch[s] = download(s, nwb_dir)
        except Exception as e:  # noqa: BLE001
            prefetch[s] = e

    th = None
    if todo:
        th = threading.Thread(target=fetch, args=(todo[0],))
        th.start()
    for i, sid in enumerate(todo):
        th.join()
        path = prefetch.pop(sid)
        if i + 1 < len(todo):
            th = threading.Thread(target=fetch, args=(todo[i + 1],))
            th.start()
        if isinstance(path, Exception):
            log.error("skip %d: %s", sid, path)
            continue
        t0 = time.time()
        rec = process_session(str(path), sid)
        with gzip.open(out_dir / f"session_{sid}.pkl.gz", "wb") as fh:
            pickle.dump(rec, fh, protocol=4)
        log.info("[%d/%d] session %d: %d units, %d candidates, %d CCG pairs, %.0f s",
                 i + 1, len(todo), sid, len(rec["units"]), len(rec["candidates"]), len(rec["ccg"]), time.time() - t0)
        if not args.keep:
            path.unlink()


if __name__ == "__main__":
    main()
