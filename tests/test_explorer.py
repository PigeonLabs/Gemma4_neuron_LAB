"""Evidence identity, filter regressions, and isolated UI job gating.

AppTest copies live-readonly catalog evidence into pytest's temporary root when
checking state files. No test writes to the real worker queue or calls a GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit import dataframe_util


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "explorer/app.py"
spec = importlib.util.spec_from_file_location("neuron_explorer", APP)
explorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explorer)


def actual_frozen():
    path = ROOT / "configs/hypotheses.json"
    if not path.exists():
        pytest.skip("Frozen research candidates have not been created")
    value = json.loads(path.read_text())
    if not value.get("candidates"):
        pytest.skip("No frozen research candidates")
    return value


def element(items, label):
    return next(item for item in items if item.label == label)


def isolated_app(tmp_path, smoke_status, worker_status):
    frozen = actual_frozen()
    for path, value in (
        ("configs/hypotheses.json", frozen),
        ("logs/summary/smoke.json", {"status": smoke_status}),
        ("runs/worker.json", {"status": worker_status}),
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    source = APP.read_text().replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(tmp_path)!r})")
    app = AppTest.from_string(source, default_timeout=15).run()
    assert not app.exception
    return app


def test_revision_separation_and_latest_catalog_status():
    # Small identity fixtures test schema behavior, not model measurements.
    discovery = [{"layer": 2, "neuron": 7, "model_revision": "r1", "status": "candidate"},
                 {"layer": 2, "neuron": 7, "model_revision": "r2", "status": "candidate"}]
    frozen = {"candidates": [{"id": "frozen", "layer": 2, "neuron": 7, "status": "candidate"}]}
    catalog = [{"layer": 2, "neuron": 7, "model_revision": "r1", "status": "ambiguous"}]
    rows = explorer.merge_catalog(discovery, frozen, catalog, "r1")
    assert len(rows) == 2
    first = next(row for row in rows if row["model_revision"] == "r1")
    second = next(row for row in rows if row["model_revision"] == "r2")
    assert first["id"] == "frozen" and first["status"] == "ambiguous"
    assert second["status"] == "candidate" and "id" not in second
    assert len(first["_sources"]) == 3 and len(second["_sources"]) == 1


def test_missing_corrupt_and_oversized_json_are_not_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr(explorer, "ROOT", tmp_path)
    errors = []
    assert explorer.read_json(tmp_path / "missing.json", errors) is None
    assert errors == []
    broken = tmp_path / "broken.json"
    broken.write_text('{"unfinished":')
    assert explorer.read_json(broken, errors) is None
    assert len(errors) == 1
    monkeypatch.setattr(explorer, "MAX_JSON_BYTES", 4)
    assert explorer.read_json(broken, errors) is None
    assert "표시 한도" in errors[-1]
    assert explorer.records(None) == []


def test_actual_catalog_four_filters_and_selected_coordinate():
    frozen = actual_frozen()
    candidates = explorer.read_json(ROOT / "catalog/public/candidates.json")
    neurons = explorer.read_json(ROOT / "catalog/public/neurons.json")
    rows = explorer.merge_catalog(candidates, frozen, neurons, frozen["model_revision"])
    target = next(row for row in rows if row.get("id") == frozen["candidates"][0]["id"])
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    app.multiselect[0].set_value([target["layer"]]).run()
    app.multiselect[1].set_value([target["neuron"]]).run()
    app.multiselect[2].set_value([explorer.category(target)]).run()
    app.multiselect[3].set_value([explorer.validation_status(target)]).run()
    assert not app.exception
    frame = app.dataframe[0].value
    assert len(frame) == 1
    assert frame.iloc[0]["layer"] == target["layer"]
    assert frame.iloc[0]["neuron"] == target["neuron"]
    picker = element(app.selectbox, "뉴런 상세")
    assert len(picker.options) == 1
    assert f"L{target['layer']} / N{target['neuron']}" in picker.options[0]
    if target.get("related_neurons"):
        related_frame = next(item.value for item in app.dataframe if "pearson_r" in item.value.columns)
        assert len(related_frame) == len(target["related_neurons"])
        assert set(related_frame["id"]) == {r["id"] for r in target["related_neurons"]}


def test_rendered_chart_data_matches_measured_summary():
    actual_frozen()
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    if not any(box.label == "Baseline 대비 지표" for box in app.selectbox):
        pytest.skip("Intervention aggregates have not been measured")
    element(app.selectbox, "Baseline 대비 지표").set_value("js_divergence").run()
    assert not app.exception
    stage = element(app.selectbox, "개입 run").value
    cid = element(app.selectbox, "개입 후보").value
    summary = json.loads((ROOT / f"logs/summary/{stage}.json").read_text())
    expected = {row["condition"]: row["metrics"]["js_divergence"]["mean"]
                for row in summary["aggregates"] if row["candidate_id"] == cid}
    charts = app.get("vega_lite_chart")
    effects = next(item for item in charts if '"field": "condition"' in item.proto.spec)
    chart_spec = json.loads(effects.proto.spec)
    payload = effects.proto.data.data
    if not payload:
        dataset_name = chart_spec["data"]["name"]
        payload = next(item.data.data for item in effects.proto.datasets if item.name == dataset_name)
    data = dataframe_util.convert_arrow_bytes_to_pandas_df(payload)
    assert dict(zip(data["condition"], data["mean"])) == pytest.approx(expected)
    assert any("표시 지표: js_divergence" in caption.value for caption in app.caption)


@pytest.mark.parametrize("smoke,worker,enabled", [
    ("PASS", "READY", True), ("PASS", "BUSY", False),
    ("FAIL", "READY", False), (None, "READY", False),
])
def test_live_is_enabled_only_after_smoke_pass_and_ready_worker(tmp_path, smoke, worker, enabled):
    app = isolated_app(tmp_path, smoke, worker)
    assert element(app.button, "Live steering 요청").disabled is not enabled
    assert not list((tmp_path / "runs").glob("jobs/*.pending.json"))


def test_live_submit_writes_only_isolated_bounded_job(tmp_path):
    app = isolated_app(tmp_path, "PASS", "READY")
    element(app.text_area, "프롬프트").set_value("Compute 4 + 9. Reply with only the integer.")
    app.slider[0].set_value(-1.0)
    element(app.number_input, "최대 생성 token").set_value(12)
    element(app.button, "Live steering 요청").click().run()
    assert not app.exception
    jobs = list((tmp_path / "runs/jobs").glob("*_live.pending.json"))
    assert len(jobs) == 1
    job = json.loads(jobs[0].read_text())
    assert job == {"stage": "live", "prompt": "Compute 4 + 9. Reply with only the integer.",
                   "candidate_id": actual_frozen()["candidates"][0]["id"],
                   "sigma": -1.0, "max_new_tokens": 12}
    assert not list((tmp_path / "runs/jobs").glob("*.tmp.json"))
