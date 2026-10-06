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


def _content_length(url: str) -> int:
    r = subprocess.run(["curl", "-sSI", "--fail", url], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1])
    raise RuntimeError(f"no content-length for {url}")


def download(sid: int, nwb_dir: Path, n_streams: int = 8) -> Path:
    """Segmented parallel download (S3 throttles single connections); each segment
    and the assembled file are checked against Content-Length."""
    out = nwb_dir / f"session_{sid}.nwb"
    url = URL.format(sid=sid)
    size = _content_length(url)
    if out.exists() and out.stat().st_size == size:
        return out
    seg = -(-size // n_streams)
    parts = [nwb_dir / f"session_{sid}.part{i}" for i in range(n_streams)]
    ranges = [(i * seg, min(size, (i + 1) * seg) - 1) for i in range(n_streams)]
    for attempt in range(5):
        todo = [i for i, (a, b) in enumerate(ranges) if not (parts[i].exists() and parts[i].stat().st_size == b - a + 1)]
        if not todo:
            break
        procs = [subprocess.Popen(["curl", "-sS", "--fail", "-r", f"{ranges[i][0]}-{ranges[i][1]}", "-o", str(parts[i]), url])
                 for i in todo]
        for pr in procs:
            pr.wait()
        time.sleep(2 ** attempt if attempt else 0)
    if any(not (parts[i].exists() and parts[i].stat().st_size == b - a + 1) for i, (a, b) in enumerate(ranges)):
        raise RuntimeError(f"download failed for {sid}")
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fo:
        for pth in parts:
            with open(pth, "rb") as fi:
                while True:
                    buf = fi.read(1 << 24)
                    if not buf:
                        break
                    fo.write(buf)
            pth.unlink()
    if tmp.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {sid}")
    tmp.rename(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb-dir", default="/home/user/nwb")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--min-version", type=int, default=2,
                    help="re-extract sessions whose record is older than this extract_version")
    args = ap.parse_args()
    nwb_dir = Path(args.nwb_dir)
    nwb_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT / "results/cohort/full_28_specimen_manifest.csv")
    sids = [int(s) for s in manifest["session_id"]]
    def needs(s):
        f = out_dir / f"session_{s}.pkl.gz"
        if not f.exists():
            return True
        with gzip.open(f, "rb") as fh:
            return pickle.load(fh).get("extract_version", 1) < args.min_version
    todo = [s for s in sids if needs(s)]
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
