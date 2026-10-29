#!/usr/bin/env python3
"""Build the Korean research homepage from recorded measurements.

Execution PASS, scientific hypothesis status, and remote publication status are
reported separately. Missing measurements remain NOT_EVALUATED. No clock value,
example neuron, fallback metric, or inferred semantic result is generated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/PigeonLabs/Gemma4_neuron_LAB"
STAGES = ("model_load", "smoke", "pilot", "survey", "steering", "validation", "expanded", "utility", "synthesis", "publication")
CONDITIONS = (
    ("baseline", "Baseline"),
    ("add_+1sd", "Additive +1SD"),
    ("add_-1sd", "Negative −1SD"),
    ("zero_ablation", "Zero Ablation"),
    ("activation_restoration", "Activation Restoration"),
    ("random_neuron_projected", "Random Neuron · projected"),
    ("random_direction_projected", "Random Direction · projected"),
)
CP_LABELS = {
    "CP0": "환경·저장소·참조 감사", "CP1": "BF16 모델·런타임 아키텍처",
    "CP2": "훅·Smoke Test", "CP3": "활성 통계·시각화", "CP4": "Steering·무작위 대조군",
    "CP5": "독립 가설 검증", "CP6": "확대 탐색·카탈로그", "CP7": "반례·연구 후보 종합",
    "CP8": "최종 기록·공개 정리",
}
FIGURE_LABELS = {
    "layer_activation_variability": ("계층별 활성 변동", "뉴런 간 p10–p90은 관측 분포의 폭이며 신뢰구간이 아닙니다."),
    "category_activation_heatmap": ("계층·범주별 활성 패턴", "프롬프트별 가중치를 같게 한 범주 평균의 표준화된 절대 편차입니다."),
    "candidate_selectivity": ("후보 뉴런 선택성", "선택된 discovery 후보의 기술 통계이며, 유의성 검정의 z-score나 인과 증거가 아닙니다."),
    "candidate_activation_histograms": ("후보 뉴런 활성 분포", "제한된 토큰 reservoir의 근사 히스토그램이며, 각 패널의 활성 축 범위가 다릅니다."),
}


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def checked_number(value, name: str, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: missing or nonfinite measured value")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name}: measured value outside valid range")
    return float(value)


def integer(value, name: str) -> int:
    checked_number(value, name, 0)
    if int(value) != value:
        raise ValueError(f"{name}: expected an integer count")
    return int(value)


def num(value, digits=4) -> str:
    return f"{checked_number(value, 'displayed metric'):.{digits}g}"


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()


def link(label: str, path: str) -> str:
    return f"[{cell(label)}]({path})"


def summary_path(stage: str) -> str:
    return f"logs/summary/{stage}.json"


def status(summary: dict | None) -> str:
    value = summary.get("status") if summary else None
    return str(value) if value in {"PASS", "FAIL", "RUNNING", "NOT_EVALUATED"} else "NOT_EVALUATED"


def successful(summaries: dict, *stages: str):
    for stage in stages:
        if status(summaries.get(stage)) == "PASS":
            return stage, summaries[stage]
    return None, None


def evidence(value, root: Path) -> str:
    if isinstance(value, list):
        return " · ".join(evidence(v, root) for v in value)
    if isinstance(value, dict):
        path = value.get("path") or value.get("url")
        if path:
            return link(value.get("label", "측정 근거"), str(path))
        return cell(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if not value:
        return ""
    value = str(value)
    if value.startswith(("https://", "http://")) or (root / value.split("#", 1)[0]).is_file():
        return link("측정 근거", value)
    return cell(value)


def catalog_rows(root: Path) -> list[dict]:
    source = load(root / "catalog/public/neurons.json", [])
    if isinstance(source, dict):
        source = source.get("neurons", source.get("entries", []))
    if not isinstance(source, list) or any(not isinstance(row, dict) for row in source):
        raise ValueError("Public neuron catalog must contain a list of measured neuron entries")
    return source


def latest_date(summaries: dict) -> str | None:
    dates = []
    for summary in summaries.values():
        if not isinstance(summary, dict):
            continue
        run_id = summary.get("run_id", "")
        if re.match(r"\d{8}T\d{6}_", str(run_id)):
            dates.append(f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}")
    for checkpoint in summaries.get("publication", {}).get("checkpoints", []):
        day = checkpoint.get("date_kst", "")
        if re.match(r"\d{4}-\d{2}-\d{2}", str(day)):
            dates.append(day[:10])
    return max(dates) if dates else None


def metric_cell(metric: dict, name: str, count: int, percentage=False) -> str:
    mean = checked_number(metric.get("mean"), name, 0, 1 if percentage else None)
    low = checked_number(metric.get("ci_low"), name + " lower CI", 0, 1 if percentage else None)
    high = checked_number(metric.get("ci_high"), name + " upper CI", 0, 1 if percentage else None)
    if low > high or integer(metric.get("n"), name + " n") != count:
        raise ValueError(f"{name}: interval order or prompt denominator is inconsistent")
    if percentage:
        return f"{100 * mean:.1f}%"
    return f"{mean:.4g} [{low:.4g}, {high:.4g}]"


def result_rows(summary: dict, candidate_id="H001") -> dict:
    rows = {}
    for row in summary.get("aggregates", []):
        if row.get("candidate_id") != candidate_id:
            continue
        condition = row.get("condition")
        if condition in rows:
            raise ValueError(f"Duplicate aggregate for {candidate_id}/{condition}")
        rows[condition] = row
    return rows


def round2_ef_lines(root: Path) -> list[str]:
    s = load(root / "logs/summary/round2_ef_analysis.json", {})
    if s.get("status") != "PASS" or s.get("analysis_phase") != "confirmation":
        return []
    conf = s["phases"]["confirmation"]
    e, f = conf["E"], conf["F"]["groups"]["translate"]
    p, damage, interaction = e["primary"], f["stats"]["damage"], f["stats"]["interaction"]
    labels = {"NOT_SUPPORTED": "등록 기준 미충족", "NOT_IDENTIFIABLE": "식별 불가", "SUPPORTED_WITHIN_REGISTERED_SCOPE": "등록 범위 내 지지",
              "CONDITIONAL_COMPENSATION_EVIDENCE_REQUIRES_CONTROL_GEOMETRY_REVIEW": "조건부 보상 증거 · geometry 검토 필요"}
    return ["", "## Round 2: 추천 5번·6번 실제 실행", "",
        f"새 연구로 **{integer(s['total_condition_rows'], 'R2 condition rows'):,}개 조건 행**, 검증·보정을 포함해 **{integer(s['total_forward_calls'], 'R2 forward calls'):,}회 실제 BF16 forward**를 수행했습니다. "
        "GPT Pro Deep research·Claude Opus 5.5 Extra Research의 설계 검토를 통합하고, calibration 파라미터를 commit한 뒤 별도 확증을 실행했습니다.", "",
        "| 실험 | 확증 family | 1차 결과 | 연구 판정 |", "|---|---:|---|---|",
        f"| 5 입력 취약도 예측 | {e['n_families']} | ΔR² {num(p['delta_r2'])} [{num(p['delta_r2_ci_low'])}, {num(p['delta_r2_ci_high'])}] | {labels[e['hypothesis_status']]} |",
        f"| 6 H001→H003 복구 | {conf['F']['n_source_families']} | 손상 {num(damage['mean'])} [{num(damage['ci_low'])}, {num(damage['ci_high'])}] nats; I={num(interaction['mean'])} | {labels[f['hypothesis_status']]} |", "",
        "5번은 새 입력에서 무작위 대조 반응을 측정하고 실제 개입 geometry에 조건화한 예측입니다. 6번은 손상 gate가 통과한 경우에만 복구 비율을 해석합니다. "
        "괄호는 95% family bootstrap CI이며 실행 PASS와 가설 지지는 별개입니다. "
        "[실측 보고서](reports/ROUND2_EF_RESULTS_KO.md) · [등록 설계](docs/ROUND2_EF_PREREGISTRATION.md) · [요약 JSON](logs/summary/round2_ef_analysis.json)", "",
        "![R2-E 확증 예측](figures/round2_e_predictions.svg)", "",
        "64개 생성 family에서 각 두 선택지 순서를 평균했습니다. 같은 축에서 고정된 기본·확장 예측식을 비교합니다.", "",
        "![R2-F 손상·복구 대비](figures/round2_f_factorial.svg)", "",
        "각 mode의 64개 생성 family, 점은 평균·선은 95% family CI입니다. 세부 대조와 한계는 보고서에 기록했습니다."]


def render(root: Path = ROOT) -> str:
    summaries = {stage: load(root / summary_path(stage), {}) for stage in STAGES}
    if any(not isinstance(value, dict) for value in summaries.values()):
        raise ValueError("Measurement summary must be a JSON object")
    catalog = catalog_rows(root)
    quality_note = ("생성 품질은 빈 출력·반복 proxy와 별도 소규모 정답 평가로 점검했으며, 대표적인 내용 품질 평가는 더 필요합니다. "
                    if status(summaries["utility"]) == "PASS" else
                    "생성 품질의 빈 출력·반복 proxy와 대표 행동 정확도를 구별하며, 정답 기반 보존 평가는 아직 미평가입니다. ")
    activation_stage, activation = successful(summaries, "survey", "pilot")
    causal_stage, causal = successful(summaries, "expanded", "validation", "steering")
    causal_rows = result_rows(causal) if causal else {}
    model, smoke = summaries["model_load"], summaries["smoke"]
    checkpoints = summaries["publication"].get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError("Publication checkpoints must be a list")
    published = {row.get("checkpoint"): row for row in checkpoints if isinstance(row, dict)}
    out = ["<!-- Generated from measured summary files by scripts/build_readme.py. -->",
           "# Gemma 4 Neuron Lab", "", "*Autonomous Discovery, Steering, and Causal Analysis of LLM Neurons*", "",
           "## Project Overview", "",
           "Gemma 4 12B의 MLP 뉴런을 BF16 활성값 수집, steering 개입, 무작위 대조군과 독립 검증으로 연구합니다. "
           "뉴런은 `GELU(gate(x)) × up(x)` 이후, `down_proj` 이전의 중간 활성 좌표로 정의합니다. "
           "Python·PyTorch·Transformers, streaming 통계, SQLite·Parquet를 사용합니다. "
           "[연구 지침](MASTER.md)과 [방법론](docs/METHODOLOGY.md)에 재현 조건과 한계를 기록합니다.", "",
           "## Research Status", ""]
    day = latest_date(summaries)
    if day:
        out += [f"기록 기준: **{day} KST**. 아래 `PASS`는 해당 실행·기술 검증 상태입니다. "
                "뉴런 기능·인과성의 검증 상태는 카탈로그에서 별도로 표시합니다.", ""]
    out += ["| 단계 | 실행·기록 상태 | 근거 |", "|---|---|---|"]
    state_rows = [
        ("CP0", "PASS" if published.get("CP0", {}).get("remote_verified") is True else "NOT_EVALUATED", "docs/REFERENCE_AUDIT.md"),
        ("CP1", status(model), summary_path("model_load")),
        ("CP2", status(smoke), summary_path("smoke")),
        ("CP3", ("FAIL · survey" if status(summaries["survey"]) == "FAIL" else
                 ("PASS · 전체 계층" if activation_stage == "survey" else "부분 측정 · Pilot") if activation else
                 status(summaries["pilot"])),
         summary_path("survey" if status(summaries["survey"]) == "FAIL" else activation_stage or "pilot")),
        ("CP4", status(summaries["steering"]), summary_path("steering")),
        ("CP5", status(summaries["validation"]), summary_path("validation")),
        ("CP6", status(summaries["expanded"]) + (" · 카탈로그 있음" if catalog else ""), summary_path("expanded")),
        ("CP7", status(summaries["synthesis"]), summary_path("synthesis")),
        ("CP8", "게시 확인" if published.get("CP8", {}).get("remote_verified") is True else "NOT_EVALUATED", "reports/RESEARCH_SUMMARY_KO.md"),
    ]
    for cp, state, source in state_rows:
        source_link = link("기록", source) if (root / source).is_file() else "미측정·미기록"
        out.append(f"| {cp} {CP_LABELS[cp]} | {state} | {source_link} |")
    out += round2_ef_lines(root)
    out += ["", "## Key Findings", ""]
    findings = summaries["synthesis"].get("findings", [])
    if findings:
        if not isinstance(findings, list):
            raise ValueError("Synthesis findings must be a list")
        for finding in findings[:5]:
            if not isinstance(finding, dict) or not finding.get("text"):
                raise ValueError("Each synthesis finding requires a recorded text statement")
            ref = evidence(finding.get("evidence"), root)
            out.append(f"- {cell(finding['text'])}" + (f" {ref}" if ref else ""))
    else:
        findings_count = 0
        if status(model) == "PASS":
            layers = integer(model.get("layers"), "model layers")
            width = integer(model.get("intermediate_size"), "model intermediate width")
            neurons = integer(model.get("neuron_count"), "model neuron count")
            if layers * width != neurons:
                raise ValueError("Runtime layer and neuron dimensions do not reconcile")
            out.append(f"- 실제 BF16 모델은 **{layers}계층 × {width:,}좌표 = {neurons:,}개 MLP 뉴런**입니다. "
                       f"{link('런타임 아키텍처', summary_path('model_load'))}")
            findings_count += 1
        if status(smoke) == "PASS":
            passed, total = integer(smoke.get("passed"), "smoke passed"), integer(smoke.get("total"), "smoke total")
            if passed != total or total < 1:
                raise ValueError("Smoke PASS count does not match its recorded total")
            delta = num(smoke.get("max_read_only_logit_difference"))
            out.append(f"- Smoke Test **{passed}/{total}** 검증을 통과했고, read-only 훅의 최대 logits 차이는 **{delta}**였습니다. "
                       f"{link('기술 검증', summary_path('smoke'))}")
            findings_count += 1
        if activation:
            out.append(f"- {'전체 계층 survey' if activation_stage == 'survey' else '선택 계층 pilot'}에서 "
                       f"**{integer(activation.get('prompts'), 'activation prompts')}개 프롬프트·{integer(activation.get('categories'), 'categories')}개 범주**, "
                       f"계층당 **{integer(activation.get('tokens_per_layer'), 'tokens per layer'):,}개 user-content 토큰**을 측정했습니다. "
                       f"후보 선택성은 discovery 상관 통계이며 기능은 아직 확정하지 않습니다. {link('활성 통계', summary_path(activation_stage))}")
            findings_count += 1
        if not findings_count:
            out.append("검증된 측정 결과는 아직 없습니다. 현재 환경·계측 파이프라인을 준비하고 있습니다.")
    out += ["", "## Experiment Results", ""]
    if causal and causal_rows:
        split = causal.get("split", "Unknown")
        out += [f"**H001 · {cell(causal_stage)} / {cell(split)} 분할**의 실제 집계입니다. "
                f"{link('전체 조건·후보 결과', summary_path(causal_stage))}", "",
                "| 조건 | n 프롬프트 / 개입 수 | 평균 KL · 95% CI (nats) | 다음 토큰 top-1 변화 |",
                "|---|---:|---:|---:|"]
        for condition, label in CONDITIONS:
            row = causal_rows.get(condition)
            if row is None:
                out.append(f"| {label} | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |")
                continue
            n = integer(row.get("n_prompts"), "aggregate prompt count")
            interventions = integer(row.get("n_interventions"), "aggregate intervention count")
            kl = metric_cell(row.get("metrics", {}).get("kl_base_to_changed", {}), "KL", n)
            top1 = metric_cell(row.get("metrics", {}).get("top1_changed", {}), "top1 change", n, percentage=True)
            out.append(f"| {label} | {n} / {interventions} | {kl} | {top1} |")
        out += ["", "95% CI는 프롬프트 단위 percentile bootstrap입니다. 무작위 대조군의 seed는 프롬프트 안에서 평균하며, "
                "대조군은 **+1SD 조건에서만** 측정했습니다. `projected`는 nominal down-projection L2를 맞춘 조건이며 실제 BF16 교란 norm도 별도로 기록합니다.", "",
                "활성 통계는 **user-content 토큰**을 수집합니다. Steering은 **입력의 마지막 chat-wrapper 토큰**에 개입하며, "
                "생성에서는 매 forward의 마지막 토큰에 개입합니다. 두 위치의 측정 범위를 구분해야 합니다. "
                "Restoration은 같은 입력의 clean 활성값으로 손상 좌표를 되돌리는 기술 검증입니다."]
    else:
        out.append("Baseline, 양·음 steering, zero ablation, restoration, random neuron/direction control의 비교 결과는 "
                   "**NOT_EVALUATED**입니다. 실행된 활성 통계와 개입 효과를 구분합니다.")
    out += ["", "## Visualizations", ""]
    figure_meta = load(root / "figures/metadata.json", {})
    figure_count = 0
    if figure_meta and activation:
        recorded_source = figure_meta.get("source", "")
        source_path = root / recorded_source
        source_matches = (recorded_source == summary_path(activation_stage) and source_path.is_file()
                          and hashlib.sha256(source_path.read_bytes()).hexdigest() == figure_meta.get("source_sha256"))
        if source_matches:
            for name, chart in figure_meta.get("charts", {}).items():
                if chart.get("status") != "GENERATED":
                    continue
                figure_path = chart.get("svg", "")
                actual_path = root / figure_path
                if not figure_path.startswith("figures/") or not actual_path.is_file():
                    continue
                if hashlib.sha256(actual_path.read_bytes()).hexdigest() != chart.get("svg_sha256"):
                    raise ValueError(f"Figure digest differs from measured provenance: {figure_path}")
                title, note = FIGURE_LABELS.get(name, (name.replace("_", " "), cell(chart.get("caption", ""))))
                out += [f"![{title}]({figure_path})", "",
                        f"Discovery n={integer(activation['prompts'], 'figure prompts')}, "
                        f"계층당 {integer(activation['tokens_per_layer'], 'figure tokens'):,}개 user-content 토큰. {note}", ""]
                figure_count += 1
    steering_meta = load(root / "figures/steering_metadata.json", {})
    for name in ("steering_dose_H001", "heldout_auc", "counterexample_activation"):
        chart = steering_meta.get("charts", {}).get(name, {})
        if chart.get("status") != "GENERATED":
            continue
        path = chart.get("svg", "")
        actual_path = root / path
        if not path.startswith("figures/") or not actual_path.is_file():
            raise ValueError("Missing measured steering figure: " + path)
        if hashlib.sha256(actual_path.read_bytes()).hexdigest() != chart.get("svg_sha256"):
            raise ValueError("Steering figure digest differs: " + path)
        for source in chart.get("sources", []):
            source_path = root / source["source"]
            if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != source["source_sha256"]:
                raise ValueError("Steering figure source is stale: " + path)
        out += [f"![{name.replace('_',' ')}]({path})", "", cell(chart.get("caption", "")), ""]
        figure_count += 1
    if not figure_count:
        out += ["최신 실제 측정 요약에 대응하는 그림은 아직 생성되지 않았습니다. 가상 그래프는 게시하지 않습니다.", ""]
    out += ["[그림 출처·표본 수·해시](figures/metadata.json)" if figure_count else "실제 측정 후 `scripts/make_figures.py`로 그림을 재생성합니다.", "",
            "## Interesting Neurons", ""]
    if catalog:
        out += ["가설과 검증 상태는 측정 카탈로그의 기록을 그대로 표시합니다.", "",
                "| Layer | Neuron | 기능 가설 | 상태 | 측정 효과·근거 |", "|---:|---:|---|---|---|"]
        for row in catalog[:8]:
            layer, neuron = row.get("layer", row.get("layer_id")), row.get("neuron", row.get("neuron_id"))
            integer(layer, "catalog layer")
            integer(neuron, "catalog neuron")
            hypothesis = row.get("hypothesis", row.get("hypothesized_function", "Unknown"))
            if isinstance(hypothesis, dict):
                hypothesis = hypothesis.get("text", hypothesis.get("statement", "Unknown"))
            validation_status = row.get("validation_status", row.get("status", "unknown"))
            effect = row.get("effect_summary", row.get("effect", ""))
            if not isinstance(effect, str):
                effect = ""
            own_rows = result_rows(causal, row.get("id", "")) if causal and row.get("id") else {}
            if not effect and "add_+1sd" in own_rows:
                effect = "+1SD KL=" + num(own_rows["add_+1sd"]["metrics"]["kl_base_to_changed"]["mean"])
            source = link("카탈로그", "catalog/public/neurons.json")
            out.append(f"| {layer} | {neuron} | {cell(hypothesis)} | {cell(validation_status)} | {cell(effect)} {source} |")
        counts = Counter(str(row.get("validation_status", row.get("status", "unknown"))) for row in catalog)
        out += ["", "카탈로그 상태: " + " · ".join(f"`{k}` {v}" for k, v in sorted(counts.items())) + "."]
    elif activation and activation.get("top_candidates"):
        out += ["다음은 활성 통계로 선택된 후보입니다. 기능 이름은 **Unknown**이며 인과 효과는 아직 확정하지 않습니다.", "",
                "| Layer | Neuron | Hypothesis | 상태 | 근거 |", "|---:|---:|---|---|---|"]
        for row in activation["top_candidates"][:5]:
            out.append(f"| {integer(row['layer'], 'candidate layer')} | {integer(row['neuron'], 'candidate neuron')} | Unknown | candidate | "
                       f"{link('discovery 통계', summary_path(activation_stage))} |")
    else:
        out.append("실제 측정된 뉴런 후보·기능 가설은 아직 없습니다.")
    out += ["", "## Research Timeline", ""]
    if checkpoints:
        out += ["| KST 날짜 | 체크포인트 | Commit | 원격 확인 |", "|---|---|---|---|"]
        for checkpoint in checkpoints:
            cp = cell(checkpoint.get("checkpoint", "Unknown"))
            sha = checkpoint.get("commit", "")
            commit = link(sha[:8], REPOSITORY + "/commit/" + sha) if re.fullmatch(r"[a-f0-9]{40}", str(sha)) else "미기록"
            verified = "확인됨" if checkpoint.get("remote_verified") is True else "미확인"
            out.append(f"| {cell(checkpoint.get('date_kst', '미기록'))} | {cp} | {commit} | {verified} |")
    else:
        out.append("GitHub 업로드 체크포인트는 아직 확인되지 않았습니다.")
    out += ["", "## Reproduction", ""]
    env = model.get("environment", {})
    versions = []
    if env.get("python"):
        versions.append("Python " + cell(env["python"]))
    for package in ("torch", "transformers"):
        if env.get("packages", {}).get(package):
            versions.append(package + " " + cell(env["packages"][package]))
    if versions:
        out += ["실제 실행 환경: " + " · ".join(versions) + "." + (" GPU: " + cell(env["gpu"]) + "." if env.get("gpu") else ""), ""]
    revision = model.get("revision") or (activation or {}).get("model_revision")
    if revision:
        out += [f"모델 revision: `{cell(revision)}`. 정확한 의존성은 [lock 파일](requirements.lock.txt), "
                "계산·저장 예산과 실행 조건은 [configs](configs)에 있습니다.", ""]
    out += ["프로젝트 루트에서 환경·고정 revision 다운로드:", "", "```bash",
            "python3 -m venv .venv", ".venv/bin/python -m pip install -r requirements.lock.txt",
            '.venv/bin/python -m pip install -e ".[test,explorer]"',
            'HF_HOME="$PWD/data/huggingface" .venv/bin/python scripts/prepare_model.py', "```", "",
            "커밋된 discovery 요약과 동결 가설을 유지한 채 인과 실험을 재현합니다. 새로운 survey는 별도 등록 연구로 실행합니다.", "", "```bash",
            "PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit smoke"]
    if (root / "configs/hypotheses.json").is_file():
        out += ["PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit steering"]
        if (root / "configs/validation_cases.json").is_file():
            out += ["PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit validation",
                    "PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit expanded"]
    out += ["PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit shutdown",
            'HF_HOME="$PWD/data/huggingface" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src .venv/bin/python -m neuron_lab.cli worker', "```", "",
            "측정 요약에서 그림·README를 재생성하고 수치 동기화를 검사합니다:", "", "```bash",
            ".venv/bin/python scripts/analyze_results.py", ".venv/bin/python scripts/make_figures.py",
            ".venv/bin/python scripts/make_steering_figures.py", ".venv/bin/python scripts/build_readme.py",
            ".venv/bin/python scripts/build_readme.py --check", ".venv/bin/python -m pytest -q", "```", "",
            "가중치·전체 활성 텐서·원본 생성 로그는 로컬에 보존합니다. 공개 요약의 해시와 run ID로 원본 산출물을 연결합니다.", "",
            "## Research Log", ""]
    documents = [("연구 기록", "docs/RESEARCH_LOG.md"), ("참조 구현 감사", "docs/REFERENCE_AUDIT.md"),
                 ("아키텍처", "docs/ARCHITECTURE.md"), ("방법론", "docs/METHODOLOGY.md"),
                 ("실험 계획", "docs/EXPERIMENT_PLAN.md"), ("재현 안내", "docs/REPRODUCTION.md"),
                 ("Explorer", "explorer/README.md"), ("최종 연구 보고서", "reports/RESEARCH_SUMMARY_KO.md"),
                 ("결과 재분석·신규성 중심 Round 2 설계", "reports/NOVELTY_IDEAS_ROUND2_KO.md")]
    present = [link(label, path) for label, path in documents if (root / path).is_file()]
    out += [" · ".join(present) if present else "상세 연구 기록은 아직 생성되지 않았습니다.", "",
            "## Limitations & Future Work", "",
            "자체 작성 소규모 text-only 코퍼스의 결과입니다. Discovery 후보 선택성은 인과적 기능을 뜻하지 않으며, "
            "next-token KL·top-1 변화는 의미적 정확도나 일반 성능 보존을 대신하지 않습니다. "
            f"{quality_note}"
            "무작위 대조군의 여러 seed는 독립 모델 반복 실험이 아닙니다.", "",
            "검증 범위를 넓히려면 개입·관측 토큰 위치를 맞춘 대조, 더 다양한 독립 원본, 의미적 행동 평가와 반복 실험이 필요합니다. "
            "이미지·음성·대규모 장문 일반화는 이 기록에서 검증하지 않았습니다."]
    next_experiments = summaries["synthesis"].get("next_experiments", [])
    if isinstance(next_experiments, list) and next_experiments:
        out += ["", "실제 관찰에서 정리한 다음 실험:", ""]
        for item in next_experiments[:3]:
            text = item.get("text", item.get("title", "")) if isinstance(item, dict) else item
            if text:
                out.append("- " + cell(text))
    return "\n".join(line.rstrip() for line in out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if README differs from the current measured summaries")
    parser.add_argument("--stdout", action="store_true", help="Print the deterministic README without changing any file")
    args = parser.parse_args()
    expected = render()
    destination = ROOT / "README.md"
    if args.check:
        if not destination.is_file() or destination.read_text() != expected:
            print("README.md is not synchronized with recorded measurements. Run scripts/build_readme.py.", file=sys.stderr)
            return 1
        print("README.md matches the recorded measurement summaries and figure provenance.")
        return 0
    if args.stdout:
        print(expected, end="")
        return 0
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(expected)
    temporary.replace(destination)
    print("Updated README.md from recorded measurement summaries.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"README generation rejected inconsistent or unavailable evidence: {exc}", file=sys.stderr)
        sys.exit(1)
