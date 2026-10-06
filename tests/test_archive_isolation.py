"""
The live pipeline (src/, scripts/) must not depend on anything in archive/,
which holds superseded material that was not computed from data
(see archive/README.md).

Fails if any Python file under src/ or scripts/ imports an `archive` module, or
if any file of the current pipeline (src/reanalysis, scripts/reanalysis,
scripts/calibration) refers to archive/ or to an output that was moved there.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = [ROOT / "src", ROOT / "scripts"]
# the current data-derived pipeline (Makefile targets); must not read archived outputs
PIPELINE = [ROOT / "src" / "reanalysis", ROOT / "scripts" / "reanalysis", ROOT / "scripts" / "calibration"]
ARCHIVED_OUTPUTS = (
    "archive/",
    "results/neuroscience_study",
    "results/secondary_stimulation",
    "results/controls",
    "results/figures",
    "results/sham_control",
    "manuscript/figures",
    "manuscript/tables",
    "manuscript/main.tex",
    "manuscript/manuscript_tcbb.tex",
)


def _py_files(bases=LIVE):
    for base in bases:
        yield from sorted(base.rglob("*.py"))


def test_no_archive_imports():
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "archive" or n.startswith("archive.") for n in names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"live code imports from archive/: {offenders}"


def test_no_reads_of_archived_outputs():
    offenders = []
    for path in _py_files(PIPELINE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(tok in node.value for tok in ARCHIVED_OUTPUTS):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value[:80]!r}")
    assert not offenders, f"live code references archived material: {offenders}"


def test_archive_readme_present():
    assert (ROOT / "archive" / "README.md").exists()
