"""
The live pipeline (src/, scripts/) must not depend on anything in archive/,
which holds superseded material that was not computed from data
(see archive/README.md).

Fails if any Python file under src/ or scripts/ imports an `archive` module or a
`src.*` / `scripts.*` module that no longer exists (e.g. one moved to archive/), or
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
    # legacy ML / feature-extraction layer audit (archive/README.md, second section)
    "responsiveness_methods",
    "results/cross_validation",
    "results/SCIENTIFIC_AUDIT",
    "results/feature_ablation.csv",
    "results/computational_benchmark",
    "Table_audit_loffo_ablation",
    "model_comparison_28spec",
    "per_specimen_results_28spec",
    "per_session_results_28spec",
    "leakage_audit_28spec.csv",
    "label_circularity_28spec",
    "feature_ablation_28spec",
    "graph_ablation_28spec",
    "graph_shuffle_control_28spec",
    "evidence_score_robustness_28spec",
    "final_ml_summary",
    "hyperparameter_log",
    "full_28_execution_manifest",
    "data_provenance.md",
    "label_dependency_graph",
    "temporal_window_audit",
    "preprocessing_leakage_audit",
    "hyperparameter_selection_audit",
    "nested_cv_audit",
    "gnn_leakage_audit",
    "feature_dependency_audit",
    "feature_redundancy_report",
    "duplicate_unit_audit",
    "FINAL_LEAKAGE_AUDIT_REPORT",
    "fig_audit8_graph_leakage",
    "full_28_cohort_report",
    "full_28_cohort_summary",
    "full_cohort_analysis_summary",
    "reports/leakage_audit",
    "reports/robustness_audit",
    "reports/statistical_audit",
    "reports/tcbb_readiness",
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


def _module_exists(name):
    rel = Path(*name.split("."))
    return (ROOT / rel).with_suffix(".py").exists() or (ROOT / rel).is_dir()


def test_live_project_imports_resolve():
    """A live src.* / scripts.* import must not point at a module that was archived."""
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for n in names:
                if n.split(".")[0] in ("src", "scripts") and not _module_exists(n):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {n}")
    assert not offenders, f"live code imports modules that do not exist: {offenders}"


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
