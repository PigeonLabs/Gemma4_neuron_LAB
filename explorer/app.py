"""Local Streamlit research console; reads evidence and queues bounded live jobs.

This process never imports the model runtime, runs commands, or calls an API.
The single GPU worker owns model access and validates every submitted job.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 24 * 1024 * 1024


def read_json(path: Path, errors: list[str] | None = None):
    if not path.exists():
        return None
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("24 MiB 표시 한도를 초과했습니다")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if errors is not None:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
        return None


def records(value, keys=("candidates", "neurons")) -> list[dict]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in keys:
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    return []


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def identity(row: dict, default_revision: str) -> tuple | None:
    layer, neuron = row.get("layer"), row.get("neuron")
    if not isinstance(layer, int) or not isinstance(neuron, int):
        return None
    revision = row.get("model_revision", row.get("revision", default_revision))
    return (str(revision), layer, neuron)


def category(row: dict) -> str:
    return str(row.get("target_category", row.get("category", row.get("best_category", "Unknown"))))


def validation_status(row: dict) -> str:
    status = row.get("validation_status", row.get("status", "unknown"))
    if isinstance(status, dict):
        status = status.get("status", "unknown")
    return str(status)


def label(row: dict) -> str:
    cid = row.get("id", row.get("candidate_id"))
    coordinate = f"L{row.get('layer', '?')} / N{row.get('neuron', '?')}"
    return f"{cid + ' · ' if isinstance(cid, str) else ''}{coordinate} · {category(row)}"


def merge_catalog(candidates, frozen, neurons, default_revision):
    merged = {}
    # A later public catalog can update the frozen hypothesis's validation state.
    for source, value in (("catalog/public/candidates.json", candidates),
                          ("configs/hypotheses.json", frozen),
                          ("catalog/public/neurons.json", neurons)):
        for raw in records(value):
            key = identity(raw, default_revision)
            if key is None:
                continue
            old = merged.get(key, {})
            row = {**old, **raw, "model_revision": key[0]}
            row["_sources"] = [*old.get("_sources", []), source]
            merged[key] = row
    return sorted(merged.values(), key=lambda r: (r["layer"], r["neuron"], r["model_revision"]))


def filter_rows(rows, selected_layers, selected_neurons, selected_categories, selected_statuses):
    return [row for row in rows
            if (not selected_layers or row.get("layer") in selected_layers)
            and (not selected_neurons or row.get("neuron") in selected_neurons)
            and (not selected_categories or category(row) in selected_categories)
            and (not selected_statuses or validation_status(row) in selected_statuses)]


def scalar_table(row: dict):
    nested = row.get("activation_statistics", {})
    values = {**nested, **row} if isinstance(nested, dict) else row
    fields = ["mean", "std", "min", "max", "frequency", "q05", "q50", "q95",
              "prompt_mean", "cross_prompt_std", "selectivity_gap", "selectivity_z",
              "quantile_sample_count"]
    table = [{"statistic": key, "value": values[key]} for key in fields if finite(values.get(key))]
    if table:
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("이 항목에 공개된 활성 요약 수치가 없습니다.")


def candidate_panel(rows):
    if not rows:
        st.info("현재 필터에 해당하는 측정 후보가 없습니다. 수집 완료 후 catalog가 표시됩니다.")
        return
    st.dataframe([{"layer": r["layer"], "neuron": r["neuron"], "category": category(r),
                   "validation_status": validation_status(r), "selectivity_z": r.get("selectivity_z"),
                   "run_id": r.get("run_id")} for r in rows], hide_index=True, width="stretch")
    index = st.selectbox("뉴런 상세", range(len(rows)), format_func=lambda i: label(rows[i]), key="detail_neuron")
    row = rows[index]
    st.caption(f"Revision: {row['model_revision']} · 출처: {', '.join(row['_sources'])}")
    left, right = st.columns(2)
    with left:
        st.markdown("**관측 통계**")
        scalar_table(row)
    with right:
        st.markdown("**표본 활성 히스토그램**")
        sample = row.get("sampled_activations")
        sample = [float(x) for x in sample if finite(x)] if isinstance(sample, list) else []
        if sample:
            count, edges = np.histogram(sample, bins=min(16, max(3, int(math.sqrt(len(sample))))))
            centers = (edges[:-1] + edges[1:]) / 2
            st.bar_chart({"activation": centers, "count": count}, x="activation", y="count")
            st.caption(f"실제 보존한 token 표본 {len(sample)}개. 전체 활성분포의 근사 표본이며 tail은 불안정할 수 있습니다.")
        else:
            st.info("보존한 token 표본이 없어 히스토그램을 표시하지 않습니다.")
    means = row.get("category_means")
    if isinstance(means, dict):
        points = [{"category": c, "mean": v} for c, v in means.items() if finite(v)]
        if points:
            st.markdown("**범주별 prompt 균형 평균**")
            st.bar_chart(points, x="category", y="mean")
    for key, title in (("top_contexts", "상위 활성 문맥"), ("low_contexts", "하위 활성 문맥")):
        contexts = row.get(key)
        if isinstance(contexts, list) and contexts:
            with st.expander(title):
                st.dataframe(contexts, hide_index=True, width="stretch")
                st.caption("Prompt별 극단에서 뽑은 사례입니다. 전체 token의 순위를 그대로 나열한 목록과 다릅니다.")
    if row.get("related_neurons"):
        with st.expander("관련 뉴런"):
            related = row["related_neurons"]
            if isinstance(related, list) and all(isinstance(r, dict) for r in related):
                fields = ("id", "layer", "neuron", "pearson_r", "n_category_means", "run_ids")
                st.dataframe([{k: r.get(k) for k in fields} for r in related],
                             hide_index=True, width="stretch")
                st.caption("실제로 측정한 범주 평균 활성 profile의 관찰적 유사도입니다. "
                           "뉴런 사이의 연결·상호작용·같은 의미 기능을 입증하지 않습니다.")
            else:
                st.write(related)
    with st.expander("원본 catalog 항목"):
        st.json({k: v for k, v in row.items() if not k.startswith("_")})


def effects_panel(summaries, visible_ids, filters_active):
    available = {name: value for name, value in summaries.items()
                 if isinstance(value, dict) and isinstance(value.get("aggregates"), list) and value["aggregates"]}
    if not available:
        st.info("완료된 개입 집계가 없습니다.")
        return
    stage = st.selectbox("개입 run", list(available), key="effect_stage")
    run = available[stage]
    st.caption(f"Run: {run.get('run_id', 'Unknown')} · split: {run.get('split', 'Unknown')} · status: {run.get('status', 'Unknown')}")
    group = [r for r in run["aggregates"] if isinstance(r, dict)
             and (not filters_active or r.get("candidate_id") in visible_ids)]
    ids = sorted({str(r["candidate_id"]) for r in group if r.get("candidate_id") is not None})
    if not ids:
        st.info("현재 필터에 해당하는 개입 결과가 없습니다.")
        return
    cid = st.selectbox("개입 후보", ids, key="effect_candidate")
    selected = [r for r in group if str(r.get("candidate_id")) == cid]
    names = sorted({k for r in selected for k, v in r.get("metrics", {}).items()
                    if isinstance(v, dict) and finite(v.get("mean"))})
    if not names:
        st.info("이 후보에 표시 가능한 집계 지표가 없습니다.")
        return
    metric = st.selectbox("Baseline 대비 지표", names, key="effect_metric")
    table = []
    for row in selected:
        value = row.get("metrics", {}).get(metric)
        if isinstance(value, dict) and finite(value.get("mean")):
            table.append({"condition": row.get("condition"), "mean": value["mean"],
                          "ci_low": value.get("ci_low"), "ci_high": value.get("ci_high"),
                          "n_prompts": row.get("n_prompts", value.get("n")),
                          "n_interventions": row.get("n_interventions"),
                          "generation_n": row.get("generation_n"),
                          "generation_changed_fraction": row.get("generation_changed_fraction"),
                          "collapse_proxy_fraction": row.get("collapse_proxy_fraction")})
    if table:
        st.caption(f"표시 지표: {metric} · 조건별 평균과 신뢰구간")
        st.bar_chart(table, x="condition", y="mean")
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Baseline 대비 next-token 분포/개입 norm 또는 생성 변화의 집계입니다. 행동의 의미·정확도 점수가 아닙니다. CI가 있으면 아래 run에 명시된 방법을 따릅니다.")
    for field in ("confidence_interval", "control_matching", "position_policy", "quality_scope"):
        if run.get(field):
            st.write(f"{field}: {run[field]}")


def hypotheses_panel(frozen_rows, summaries, rows, default_revision):
    visible = {identity(row, default_revision) for row in rows}
    selected = [row for row in frozen_rows if identity(row, default_revision) in visible]
    if not selected:
        st.info("현재 필터에 해당하는 동결 가설이 없습니다.")
    for row in selected:
        with st.expander(label(row), expanded=len(selected) == 1):
            found = False
            for field in ("hypothesis", "hypotheses", "hypothesis_text", "competing_explanations",
                          "alternative_explanations", "alternatives", "falsification", "prediction"):
                if row.get(field) is not None:
                    st.markdown(f"**{field}**")
                    st.write(row[field])
                    found = True
            if not found:
                st.caption("이 항목에 별도의 가설 설명 필드가 없습니다.")
            st.json(row)
    ids = {str(r.get("id", r.get("candidate_id"))) for r in rows
           if r.get("id", r.get("candidate_id")) is not None}
    correlations = [{"stage": name, "split": run.get("split"), **entry}
                    for name, run in summaries.items() if isinstance(run, dict)
                    for entry in run.get("correlation", []) if isinstance(entry, dict)
                    and str(entry.get("candidate_id")) in ids]
    if correlations:
        st.markdown("**분할별 활성 상관 검증**")
        st.dataframe(correlations, hide_index=True, width="stretch")
        st.caption("AUC는 범주와 활성의 관찰적 분리도입니다. permutation p-value는 별도 보정이 명시되지 않으면 다중비교 보정값이 아닙니다.")
    st.caption("candidate/correlated 등은 저장된 검증 상태를 표시합니다. 설명 모델의 문장이나 logits 변화만으로 기능을 확정하지 않습니다.")


def run_panel(summaries, errors):
    if not summaries:
        st.info("공개 run 요약이 없습니다.")
    else:
        table = []
        for name, run in summaries.items():
            if not isinstance(run, dict):
                continue
            entry = {"summary": name, "run_id": run.get("run_id"), "status": run.get("status"),
                     "elapsed_seconds": run.get("elapsed_seconds"),
                     "prompts": run.get("prompts", run.get("prompt_count")),
                     "intervention_count": run.get("intervention_count")}
            peak = run.get("peak_reserved_bytes")
            entry["peak_reserved_GB"] = peak / 1e9 if finite(peak) else None
            table.append(entry)
        st.dataframe(table, hide_index=True, width="stretch")
        selected = st.selectbox("요약 원본", list(summaries), key="summary_detail")
        st.json(summaries[selected], expanded=False)
    jobs_dir = ROOT / "runs/jobs"
    paths = sorted(jobs_dir.glob("*.json"), reverse=True)[:30] if jobs_dir.exists() else []
    jobs = []
    for path in paths:
        if not any(part in path.name for part in (".pending.", ".running.", ".done.", ".failed.")):
            continue
        job = read_json(path, errors)
        if isinstance(job, dict):
            jobs.append({"file": path.name, "stage": job.get("stage"),
                         "state": path.name.split(".")[-2],
                         "status": job.get("result", {}).get("status") if isinstance(job.get("result"), dict) else None,
                         "error": job.get("error")})
    if jobs:
        st.markdown("**최근 로컬 작업**")
        st.dataframe(jobs, hide_index=True, width="stretch")
    if errors:
        with st.expander("자료 읽기 오류"):
            for error in errors:
                st.write(error)


def readiness():
    smoke = read_json(ROOT / "logs/summary/smoke.json")
    worker = read_json(ROOT / "runs/worker.json")
    smoke_ok = isinstance(smoke, dict) and smoke.get("status") == "PASS"
    worker_ok = isinstance(worker, dict) and worker.get("status") == "READY"
    return smoke_ok and worker_ok, smoke, worker


def generation_text(value):
    if isinstance(value, dict):
        return value.get("text")
    return value if isinstance(value, str) else None


def live_job_panel():
    key = st.session_state.get("live_job_key")
    if not key:
        return
    found = next((ROOT / "runs/jobs" / f"{key}.{state}.json"
                  for state in ("done", "failed", "running", "pending")
                  if (ROOT / "runs/jobs" / f"{key}.{state}.json").exists()), None)
    if found is None:
        st.info("작업 상태 파일을 기다리고 있습니다.")
        return
    job = read_json(found)
    if not isinstance(job, dict):
        st.info("작업 결과가 저장되는 중입니다. 잠시 후 새로고침하세요.")
        return
    state = found.name.split(".")[-2]
    st.caption(f"로컬 작업: {found.name}")
    if state in {"pending", "running"}:
        st.info("대기 중" if state == "pending" else "GPU worker에서 실행 중")
        return
    if state == "failed" or job.get("error"):
        st.error(str(job.get("error", "Worker가 작업 실패를 기록했습니다.")))
        return
    result = job.get("result", {})
    if not isinstance(result, dict):
        st.warning("Worker가 예상한 JSON 객체 형식의 결과를 반환하지 않았습니다.")
        return
    baseline = generation_text(result.get("baseline", result.get("baseline_text")))
    changed = generation_text(result.get("intervention", result.get("intervention_text")))
    left, right = st.columns(2)
    with left:
        st.markdown("**Baseline**")
        st.text(baseline if baseline is not None else "결과에 baseline text가 없습니다.")
    with right:
        st.markdown("**Intervention**")
        st.text(changed if changed is not None else "결과에 intervention text가 없습니다.")
    with st.expander("Live 결과 원본"):
        st.json(result)
    st.caption("Live 입력은 탐색용입니다. 독립 validation/test의 증거로 자동 편입되지 않습니다.")


def live_panel(frozen_rows):
    allowed, smoke, worker = readiness()
    options = [row for row in frozen_rows if isinstance(row.get("id"), str) and row["id"]]
    st.write("동결한 후보의 마지막 forward token에 제한된 additive steering을 요청합니다.")
    st.caption(f"Smoke: {smoke.get('status', 'NOT_EVALUATED') if isinstance(smoke, dict) else 'NOT_EVALUATED'} · Worker: {worker.get('status', 'UNAVAILABLE') if isinstance(worker, dict) else 'UNAVAILABLE'}")
    if not allowed:
        st.info("Live steering을 사용할 수 없습니다. Smoke PASS와 로컬 worker READY가 모두 필요합니다.")
    if not options:
        st.info("동결된 후보 목록이 없어 요청할 수 없습니다.")
    enabled = allowed and bool(options)
    with st.form("live_steering"):
        picked = st.selectbox("동결 후보", range(len(options)), format_func=lambda i: label(options[i]),
                              disabled=not enabled) if options else None
        prompt = st.text_area("프롬프트", max_chars=4000, disabled=not enabled,
                              help="모델 입력 한도는 chat template을 포함한 256 token입니다. 초과 입력은 worker가 거부합니다.")
        sigma = st.slider("강도 (discovery token std 단위)", -2.0, 2.0, 1.0, 0.25, disabled=not enabled)
        max_tokens = st.number_input("최대 생성 token", min_value=1, max_value=24, value=16,
                                     step=1, disabled=not enabled)
        submitted = st.form_submit_button("Live steering 요청", disabled=not enabled)
    if submitted:
        current_ready, _, _ = readiness()
        if not current_ready:
            st.error("Worker 상태가 바뀌었습니다. READY 상태에서 다시 요청하세요.")
        elif not prompt.strip():
            st.warning("프롬프트를 입력하세요.")
        else:
            # Re-read the frozen list before writing; no caller-selected file paths.
            latest = read_json(ROOT / "configs/hypotheses.json")
            valid_ids = {r.get("id") for r in records(latest) if isinstance(r.get("id"), str)}
            cid = options[picked]["id"]
            if cid not in valid_ids:
                st.error("동결 후보 목록이 바뀌었습니다. 페이지를 새로고침하세요.")
            else:
                key = f"{time.time_ns()}_live"
                jobs = ROOT / "runs/jobs"
                jobs.mkdir(parents=True, exist_ok=True)
                payload = {"stage": "live", "prompt": prompt.strip(), "candidate_id": cid,
                           "sigma": float(sigma), "max_new_tokens": int(max_tokens)}
                temporary = jobs / f"{key}.tmp.json"
                try:
                    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
                    temporary.replace(jobs / f"{key}.pending.json")
                except OSError as exc:
                    st.error(f"작업 파일을 저장하지 못했습니다: {exc}")
                else:
                    st.session_state["live_job_key"] = key
                    st.success("로컬 GPU worker 대기열에 요청했습니다.")
    if hasattr(st, "fragment"):
        st.fragment(run_every=2)(live_job_panel)()
    else:
        st.button("Live 결과 새로고침", key="live_refresh")
        live_job_panel()


def main():
    st.set_page_config(page_title="Gemma 4 Neuron Lab · Research Console", layout="wide")
    st.title("Gemma 4 Neuron Lab")
    st.caption("로컬 연구 콘솔 · 측정된 활성·개입·검증 기록")
    errors = []
    summaries = {p.stem: value for p in sorted((ROOT / "logs/summary").glob("*.json"))
                 if (value := read_json(p, errors)) is not None}
    frozen = read_json(ROOT / "configs/hypotheses.json", errors)
    frozen_rows = records(frozen)
    model = summaries.get("model_load", {})
    revision = (frozen.get("model_revision") if isinstance(frozen, dict) else None) or model.get("revision", "Unknown")
    rows = merge_catalog(read_json(ROOT / "catalog/public/candidates.json", errors), frozen,
                         read_json(ROOT / "catalog/public/neurons.json", errors), revision)
    with st.sidebar:
        st.button("전체 자료 새로고침")
        st.caption(f"Model revision: {revision}")
        layers = st.multiselect("Layer", sorted({r["layer"] for r in rows}))
        neurons = st.multiselect("Neuron", sorted({r["neuron"] for r in rows}))
        categories = st.multiselect("Category", sorted({category(r) for r in rows}))
        statuses = st.multiselect("Validation status", sorted({validation_status(r) for r in rows}))
        st.caption("선택하지 않은 필터는 전체 항목을 표시합니다.")
    visible = filter_rows(rows, layers, neurons, categories, statuses)
    ids = {str(r.get("id", r.get("candidate_id"))) for r in visible
           if r.get("id", r.get("candidate_id")) is not None}
    tabs = st.tabs(["뉴런 탐색", "개입 효과", "가설·검증", "실행 기록", "Live steering"])
    with tabs[0]:
        candidate_panel(visible)
    with tabs[1]:
        effects_panel(summaries, ids, bool(layers or neurons or categories or statuses))
    with tabs[2]:
        hypotheses_panel(frozen_rows, summaries, visible, revision)
    with tabs[3]:
        run_panel(summaries, errors)
    with tabs[4]:
        live_panel(frozen_rows)
    st.caption("공개 요약의 PASS는 해당 기술·실행 검사의 상태입니다. 뉴런의 의미 기능이 검증됐다는 뜻은 아닙니다.")


if __name__ == "__main__":
    main()
