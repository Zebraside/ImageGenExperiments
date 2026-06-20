"""Offline tests for the imagegen.evaluate orchestrator's pure helpers.

The FID/KID numeric path needs Inception weights + a GPU and is exercised by the
real eval run; here we cover the run selection and the ranking/markdown rendering
that turn metric dicts into the report table.
"""

from imagegen.evaluate import RUNS, render_markdown, resolve_runs


def test_resolve_runs_all_and_filter():
    assert resolve_runs(None) == RUNS  # no filter -> every run, declared order
    only = resolve_runs("exp2_lora,base")
    keys = [r["key"] for r in only]
    assert keys == ["base", "exp2_lora"]  # filtered but order preserved


def test_resolve_runs_ignores_unknown_keys():
    assert resolve_runs("does_not_exist") == []


def test_render_markdown_ranks_by_fid_and_deltas_vs_base():
    results = {
        "base": {"label": "Base", "fid": 100.0, "kid": 0.10, "kid_std": 0.01, "loss": 0.2},
        "exp2_lora": {"label": "LoRA", "fid": 60.0, "kid": 0.05, "kid_std": 0.005, "loss": 0.15},
        "exp5_lora_lpips": {"label": "LPIPS", "fid": 80.0, "kid": 0.08, "kid_std": 0.004, "loss": 0.16},
    }
    md = render_markdown(results)
    lines = [ln for ln in md.splitlines() if ln.startswith("| ")]
    # header + 3 data rows; ranked best-FID first => LoRA, LPIPS, base
    data = [ln for ln in lines if "`" in ln]
    assert "`exp2_lora`" in data[0]
    assert "`exp5_lora_lpips`" in data[1]
    assert "`base`" in data[2]
    # ΔFID vs base: LoRA improves by 40, base is the reference
    assert "-40.00" in data[0]
    assert "0.00 (base)" in data[2]


def test_render_markdown_handles_missing_metrics():
    results = {
        "base": {"label": "Base", "fid": 100.0, "kid": 0.10, "kid_std": 0.01, "loss": 0.2},
        "broken": {"label": "Broken", "fid": None, "error": "boom"},
    }
    md = render_markdown(results)
    assert "—" in md  # missing FID/KID rendered as em dash
    assert "`broken`" in md  # still listed (sorted last)
