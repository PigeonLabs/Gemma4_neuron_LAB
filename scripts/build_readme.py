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
    receipts = load(root / "logs/summary/round2_ef_publication.json", {}).get("checkpoints", [])
    results_receipt = next((r for r in reversed(receipts) if r.get("checkpoint") == "R2_EF_RESULTS" and r.get("remote_verified") is True), None)
    publication_line = (f"결과 게시 확인: [{results_receipt['remote_sha'][:8]}]({REPOSITORY}/commit/{results_receipt['remote_sha']}). "
        "[R2 게시 기록](logs/summary/round2_ef_publication.json)" if results_receipt else "R2 결과 게시 상태: 미확인.")
    peer = load(root / "logs/summary/round2_ef_peer_review.json", {})
    peer_lines = []
    if peer.get("status") == "COMPLETED":
        ranked = sorted(peer["recommendations"], key=lambda r: r["rank"])
        later = load(root / "logs/summary/round3a_p0.json", {})
        peer_lines = ["실측 결과를 GPT Pro·Claude Opus 5.5 Extra에 전달해 재검토를 완료했습니다. Novelty 우선 후속 추천: "
            + " → ".join(f"{r['rank']}위 {cell(r['title_ko'])}" for r in ranked)
            + (". 이는 실행 전 추천 기록이며, 이후 1번의 실제 진행 상태는 아래 R3-A 결과를 따릅니다. " if later else ". 새 실험은 **NOT_EVALUATED**입니다. ")
            + "[피드백 반영·설계 상세](reports/ROUND3_RECOMMENDATIONS_KO.md)", ""]
    return ["", "## Round 2: 추천 5번·6번 실제 실행", "",
        f"새 연구로 **{integer(s['total_condition_rows'], 'R2 condition rows'):,}개 조건 행**, 검증·보정을 포함해 **{integer(s['total_forward_calls'], 'R2 forward calls'):,}회 실제 BF16 forward**를 수행했습니다. "
        "GPT Pro Deep research·Claude Opus 5.5 Extra Research의 설계 검토를 통합하고, calibration 파라미터를 commit한 뒤 별도 확증을 실행했습니다.", "",
        "| 실험 | 확증 family | 1차 결과 | 연구 판정 |", "|---|---:|---|---|",
        f"| 5 입력 취약도 예측 | {e['n_families']} | ΔR² {num(p['delta_r2'])} [{num(p['delta_r2_ci_low'])}, {num(p['delta_r2_ci_high'])}] | {labels[e['hypothesis_status']]} |",
        f"| 6 H001→H003 복구 | {conf['F']['n_source_families']} | 손상 {num(damage['mean'])} [{num(damage['ci_low'])}, {num(damage['ci_high'])}] nats; I={num(interaction['mean'])} | {labels[f['hypothesis_status']]} |", "",
        "5번은 후보를 먼저 측정해 그 geometry로 대조의 크기를 정한 조건부 예측입니다. 후보 관측 전 예측이나 호출 절감은 입증하지 않았습니다. 6번은 손상 gate가 통과한 경우에만 복구 비율을 해석합니다. "
        "괄호는 95% family bootstrap CI이며 실행 PASS와 가설 지지는 별개입니다. "
        "[실측 보고서](reports/ROUND2_EF_RESULTS_KO.md) · [등록 설계](docs/ROUND2_EF_PREREGISTRATION.md) · [요약 JSON](logs/summary/round2_ef_analysis.json)", "",
        publication_line, "",
        *peer_lines,
        "![R2-E 확증 예측](figures/round2_e_predictions.svg)", "",
        "64개 생성 family에서 각 두 선택지 순서를 평균했습니다. 같은 축에서 고정된 기본·확장 예측식을 비교합니다.", "",
        "![R2-F 손상·복구 대비](figures/round2_f_factorial.svg)", "",
        "각 mode의 64개 생성 family, 점은 평균·선은 95% family CI입니다. 세부 대조와 한계는 보고서에 기록했습니다."]


def round3a_lines(root: Path) -> list[str]:
    s = load(root / "logs/summary/round3a_p0.json", {})
    if s.get("status") != "PASS":
        return []
    q = s["qualification"]
    stopped = q["status"] == "STOPPED_AT_NUMERICAL_QUALIFICATION"
    rows = ["", "## R3-A: 추천 1번의 실제 수치 검증", "",
        f"**{'사전등록한 수치 gate 미통과로 P0에서 중단했습니다.' if stopped else 'P0 수치 검증을 완료했습니다.'}** "
        f"새 {integer(s['n_source_families'], 'P0 families')} family의 {integer(s['n_variants'], 'P0 variants')}변형에서 "
        f"{integer(s['condition_rows'], 'P0 rows')}개 조건 행·{integer(s['forward_calls'], 'P0 forwards')}회 실제 forward를 측정했습니다. "
        "같은 BF16 저장 가중치를 FP32로 올려 비교했으며 훅·복원·nonzero 개입 전달은 통과했습니다.", "",
        "| 후보 | BF16 평균 절댓값 (nats) | FP32 평균 절댓값 (nats) | 부호 효과 Spearman | 수치 eligible |", "|---|---:|---:|---:|---|"]
    rows += [f"| {cell(c['id'])} | {num(c['mean_abs_signed_effect_BF16'])} | {num(c['mean_abs_signed_effect_FP32'])} | {num(c['spearman_signed_effects'])} | {'통과' if c['eligible'] else '미통과'} |" for c in q['candidates']]
    rows += ["", "32변형은 8개 원본의 paired 파생이며 독립 n=32가 아닙니다. 현재 dose·위치·과제에서 정밀도에 안정적인 의미 효과를 확인하지 못했습니다. "
        "뉴런의 기능 부재나 기존 R2 통계의 소급 무효화를 뜻하지 않습니다. Discovery·calibration·confirmation은 **NOT_EVALUATED**입니다.", "",
        "[실측·중단 사유](reports/ROUND3A_RESULTS_KO.md) · [등록 기준](docs/ROUND3A_PREREGISTRATION.md) · [실측 JSON](logs/summary/round3a_p0.json)", "",
        "![R3-A P0 정밀도 비교](figures/round3a_p0_precision.svg)", ""]
    return rows


def round4_lines(root: Path) -> list[str]:
    s = load(root / "logs/summary/round3a_peer_review.json", {})
    if s.get("status") != "COMPLETED":
        return []
    return ["", "## P0 이후 GPT·Claude 재검토와 다음 추천", "",
        "두 채팅에 P0 실측 요약을 전달하고 완료 답변과 Claude 수정본을 확인했습니다. 두 모델의 자체 1순위는 달랐습니다. "
        "Novelty를 우선한 최종 추천은 " + " → ".join(f"{r['rank']}위 {cell(r['title_ko'])}" + ("(조건부)" if r['rank']==3 else "") for r in s['recommendations']) + ".", "",
        "모든 후속은 **DRAFT / NOT_EVALUATED**, 신규성은 **UNVERIFIED**입니다. 이번 재검토 중 새 GPU 추론은 하지 않았습니다. "
        "[피드백·선행연구·추천 설계](reports/ROUND4_RECOMMENDATIONS_KO.md) · [초안 예산](configs/round4_plan.yaml)", ""]


def round5_lines(root: Path) -> list[str]:
    s = load(root / "logs/summary/round5_design_peer_review.json", {})
    v = load(root / "logs/summary/round5_design_verification.json", {})
    if s.get("status") != "DESIGN_REVIEW_COMPLETE" or v.get("status") != "PASS":
        return []
    if s.get("actual_R5_GPU_forwards") != 0 or v.get("inference_executed") is not False:
        raise ValueError("R5 design-only homepage requires separate measured-result rendering after execution")
    registration = load(root / "logs/summary/round5_registration.json", {})
    measured = load(root / "logs/summary/round5_discovery.json", {})
    current = []
    if measured:
        outcome = measured.get("analysis", {}).get("gate_outcome", measured.get("gate_outcome", "UNKNOWN"))
        current = ["", f"**R5 실제 실행:** {integer(measured['actual_forward_calls'], 'R5 actual calls'):,} forward, "
            f"후보 {integer(measured.get('candidate_count_evaluated', 0), 'R5 candidates')}개 평가, "
            f"판정 **{cell(outcome)}**. 피드백 회복·query-time 모집·norm replay는 **NOT_EVALUATED**입니다.",
            "[실측 결과와 해석](reports/ROUND5_RESULTS_KO.md) · [실측 요약 JSON](logs/summary/round5_discovery.json)", ""]
    elif registration:
        current = ["", "**R5 실행 등록 완료:** 새 technical/discovery 입력과 gate를 고정했습니다. "
            "모델 결과는 아직 NOT_EVALUATED이며 독립 calibration/confirmation은 열지 않았습니다.", ""]
    return ["", "## 선택한 새 연구 R5: 지속 차단 아래 피드백에 따른 기능적 모집", "",
        "사용자가 새 제안 2번을 선택했습니다. GPT Pro Deep Research·Claude Opus 5.5 Extra Research의 새 보고서와 교차 비판을 반영해 "
        "**동일한 문제·참조답에서 오류 status 이력만 바꿀 때, 남은 좌표 집합에 대한 질의 시점의 인과적 의존성이 달라지는가**를 설계했습니다. "
        "위 R4 순위는 당시 추천 기록입니다.", "",
        f"설계 기록 시점에는 **설계·CPU 검증 완료**, 실제 R5 GPU forward **{integer(s['actual_R5_GPU_forwards'], 'R5 measured forwards')}회**, "
        "손상·회복·모집 결과는 **NOT_EVALUATED**, 전체 확증은 **NOT_REGISTERED**입니다. "
        f"CPU 예시 최대 {integer(v['max_example_input_tokens'], 'R5 example tokens')}토큰; "
        f"계획상 {integer(v['independent_budget_calculation'], 'R5 planned calls'):,} forward는 미래 예산이며 실측이 아닙니다.", "",
        "[협업 설계·판정·중단 기준](reports/ROUND5_DESIGN_KO.md) · [실행 예산](configs/round5_plan.yaml) · "
        "[실제 검토 기록](logs/summary/round5_design_peer_review.json)", ""] + current


def round6_execution_lines(root: Path) -> list[str]:
    s = load(root / "logs/summary/round6_execution_results.json", {})
    if not s:
        return []
    if s.get("status") != "EXECUTED_AND_STOPPED_AT_REGISTERED_PREREQUISITES":
        raise ValueError("R6 execution needs an explicit rendering for its measured outcome")
    if sum(r["total_actual_forwards"] for r in s["studies"]) != s["total_actual_forwards"]:
        raise ValueError("R6 execution forward counts do not reconcile")
    labels = {"conflict_direct": "충돌·직접", "conflict_compose": "충돌·합성", "direct_only": "직접 단독",
              "composition_only": "합성 단독", "code_only": "코드", "agreement": "양쪽·일치",
              "conflict1": "양쪽·충돌1", "conflict2": "양쪽·충돌2", "composition": "합성"}
    out = ["", "## R6 세 실험 실제 실행: 준비도 단계 중단", "",
        f"A·B·C 및 별도 과제 A2에서 실제 **{integer(s['total_actual_forwards'], 'R6 total forwards')}회** forward를 수행했습니다. "
        f"과제 측정 {integer(s['scientific_forward_calls'], 'R6 task forwards')}회 + BF16/FP32 기술 검증 "
        f"{integer(s['technical_forward_calls'], 'R6 technical forwards')}회입니다. 기술 검증은 모두 통과했으나 "
        "네 실행 모두 사전 과제 준비도에서 멈췄습니다. **인과적 병목·백업·중간값 재사용은 NOT_IDENTIFIABLE**, "
        "native 후보 탐색·calibration·confirmation은 **NOT_EVALUATED**입니다.", "",
        "| 실행 | 준비도 실측 | 상태 |", "|---|---|---|"]
    for r in s["studies"]:
        measured = load(root / r["summary_path"], {})
        if measured.get("actual_forward_calls") != r["scientific_forward_calls"] or measured.get("outcome", {}).get("gate") != r["gate"]:
            raise ValueError("R6 execution homepage disagrees with its run summary")
        cells = "; ".join(f"{labels[k]} {v['correct']}/{v['n']} ({100*v['accuracy']:.2f}%)" for k,v in r["readiness"].items())
        out.append(f"| {cell(r['title_ko'])} | {cells} | `{cell(r['gate'])}` |")
    out += ["", "기준: 과제 85% 이상, 코드 95% 이상. 각 실행은 8개의 독립 nonce 세계에서 반복 질의·코드북 변형을 측정했습니다. "
        "정확도는 네 코드 중 다음 토큰 argmax이며, full-vocabulary top1도 별도 검산했습니다. A 실패 후 A2는 새 연구로 등록했고 추가 과제 변경은 하지 않았습니다. "
        "이 결과는 병목 부재나 중간값 미사용의 증거가 아닙니다.", "",
        "![R6 사전 과제 준비도](figures/round6_task_readiness.svg)", "",
        "[실측 결과·해석](reports/ROUND6_EXECUTION_RESULTS_KO.md) · [집계 JSON](logs/summary/round6_execution_results.json) · "
        "[원시 행 독립 재검산](logs/summary/round6_execution_replay.json) · [과제 변경 등록](reports/ROUND6_EXPLICIT_REVISION_KO.md)", ""]
    return out


def round6_lines(root: Path) -> list[str]:
    s = load(root / "logs/summary/round6_peer_review.json", {})
    d = load(root / "logs/summary/round6_margin_diagnostics.json", {})
    if not s:
        return []
    if s.get("status") != "REVIEW_COMPLETE" or s.get("new_GPU_forwards") != 0 or d.get("new_GPU_forwards") != 0:
        raise ValueError("R6 recommendations need a separate rendering after actual model execution")
    group = next(x for x in d["cells"] if x["arm"] == "L15_K128_composition" and x["witness"] == "dual_route")
    ranks = " → ".join(f"{r['rank']}위 {cell(r['title_ko'])}" + ("(조건부)" if r['conditional'] else "") for r in s["recommendations"])
    return ["", "## R5 실측 재검토와 실제 병목을 찾기 위한 R6 추천", "",
        "GPT Pro·Claude Opus 5.5 Extra에 실측 요약을 전달하고 완료 답변·추가 margin 진단의 수정·교차 비판을 읽었습니다. "
        "최종 추천: " + ranks + ".", "",
        f"기존 발견 자료의 CPU 사후 진단에서는 정확도 손상 0pp여도 한 후보의 양쪽 표 평균 절대 margin 변화가 "
        f"**{num(group['mean_absolute_margin_change_nats'], 4)} nats**였습니다. 이는 새 확증·의미 병목의 증거가 아닙니다. "
        "R5 중단 판정은 보존합니다. 이 공동 검토 기록 시점의 새 GPU forward는 **0회**이며, 당시 후속 설계는 "
        "**DRAFT / NOT_REGISTERED / NOT_EVALUATED**였습니다. 신규성은 **UNVERIFIED**이며 이후 실행은 아래 별도 기록을 따릅니다.", "",
        "[피드백·추천·중단 기준](reports/ROUND6_RECOMMENDATIONS_KO.md) · [탐색 예산 초안](configs/round6_plan.yaml) · "
        "[공동 검토 기록](logs/summary/round6_peer_review.json)", ""] + round6_execution_lines(root)


def round7_lines(root: Path) -> list[str]:
    p = load(root / "logs/summary/round7_progress.json", {})
    c = load(root / "logs/summary/round7_candidates.json", {})
    if not p or not c:
        return []
    out = ["", "## R7 자율 연구 재설정", "", cell(p["status"]) + ". 기존 실패·미식별 판정을 보존합니다. 아래는 최초 출력 전 선정한 실행 우선순위이며 현재 논문화 판정과 구분합니다.", "",
           "| 선정 당시 우선순위 | 질문 | 당시 P0 실행 판단 |", "|---:|---|---|"]
    for x in c["ranked_candidates"]:
        out.append(f"| {integer(x['rank'], 'R7 rank')} | {cell(x['id'] + ' · ' + x['title'])} | {cell(x['decision'])} |")
    out += ["", "두 기존 연구 채팅의 실제 독립 검토와 적대적 교차검토를 완료했습니다. GPT의 Pro 웹 원전 검토는 별도 Deep Research 완료 보고서와 구분합니다. 신규성은 미확정이며 P0는 탐색입니다.", ""]
    measured = []
    for letter in 'ABCDE':
        q = load(root / f"logs/summary/round7_{letter}_P0.json", {})
        if q:
            o = q.get('outcome') or {}
            measured.append((letter, q, o))
    if measured:
        out += ["| 실제 P0 | science forward | 별도 technical forward | 결과 |", "|---|---:|---:|---|"]
        for letter,q,o in measured:
            out.append(f"| {letter} | {integer(q['actual_forward_calls'], 'R7 science')} | {integer(q['technical_forward_calls_separate'], 'R7 technical')} | {cell(o.get('classification','FAIL'))} · {cell(o.get('gate','FAIL'))} |")
        out += ["", "P1/P2는 각 등록과 실제 실행 요약이 없는 한 NOT_EVALUATED입니다.", ""]
        aggregate = load(root / 'logs/summary/round7_execution_results.json', {})
        if aggregate:
            total = sum(q['actual_forward_calls'] + q['technical_forward_calls_separate'] for _,q,_ in measured)
            if total != aggregate['total_actual_forwards']:
                raise ValueError('R7 aggregate forward count is stale')
            meta = load(root / 'figures/round7_P0_results_metadata.json', {})
            source = root / 'logs/summary/round7_execution_results.json'
            figure = root / 'figures/round7_P0_results.svg'
            if meta.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest() or meta.get('figure_sha256') != hashlib.sha256(figure.read_bytes()).hexdigest():
                raise ValueError('R7 figure provenance does not match measured summary')
            out += [f"R7 실제 합계 **{integer(total, 'R7 total')}회** = science budget {integer(aggregate['scientific_forwards'], 'R7 science total')}회 + 별도 technical {integer(aggregate['technical_forwards'], 'R7 technical total')}회. 등록 P0 판정과 기제의 식별·신규성을 구분합니다.", "",
                    "![R7 실제 P0 결과와 중단](figures/round7_P0_results.svg)", "",
                    "[R7 실측 보고서](reports/ROUND7_EXECUTION_RESULTS_KO.md) · [등록 판정의 해석·추가 사실 감사](reports/ROUND7_INTERPRETATION_AUDIT_KO.md) · [원시 행 재검산](logs/summary/round7_replay.json)", ""]
    else:
        out += ["새 R7 모델 forward는 아직 0회입니다. A의 사전등록은 완료했으며 실행은 NOT_EVALUATED입니다.", ""]
    post = load(root / 'logs/summary/round7_post_results_peer_review.json', {})
    screening = load(root / 'logs/summary/round7_next_candidate_screening.json', {})
    if post and screening:
        if post.get('status') != 'ACTUAL_INDEPENDENT_AND_ADVERSARIAL_CROSS_REVIEWS_COMPLETED':
            raise ValueError('R7 post-result reviews are incomplete')
        out += ["실측 이후에도 두 채팅의 독립 심사와 적대적 교차검토를 완료했습니다. GPT는 다음 1순위의 조건부 Go를 Hold로 낮췄고 Claude는 자신의 후속 세 제안을 철회했습니다. 사실 오류는 원시 행과 토큰 위치로 검산해 정정했습니다.", "",
                f"R7 종결 당시 다음 제안 다섯 개의 신규성 심사: **Go {integer(screening['Go_count'], 'next Go')} · Hold {integer(screening['Hold_count'], 'next Hold')} · No-Go {integer(screening['No_Go_count'], 'next No-Go')}**. 당시에도 다섯 개의 실행 가능한 새 연구를 선정한 상태는 아니었습니다. 후속 최종 판단은 아래 선정 감사에 기록합니다. R8 모델 실행은 NOT_EVALUATED입니다.", "",
                "| 당시 검토 순위 | 다음 질문 | 당시 판단 |", "|---:|---|---|"]
        for item in screening['items']:
            out.append(f"| {integer(item['rank'], 'screening rank')} | {cell(item['title'])} | {cell(item['decision'])} |")
        out += ["", "[사후 교차검토·다음 후보의 9개 평가 항목](reports/ROUND7_REVIEW_AND_NEXT_KO.md) · [누적 논문 후보 순위](reports/PAPER_CANDIDATE_RANKING_KO.md)", ""]
    out += ["[다섯 후보·선행 대비](reports/ROUND7_CANDIDATES_KO.md) · [이전 증거 재감사](reports/ROUND7_BASELINE_AUDIT_KO.md) · [진행 JSON](logs/summary/round7_progress.json) · [설계 당시 공동 검토](logs/summary/round7_design_peer_review.json)", ""]
    return out


def round8_selection_lines(root: Path) -> list[str]:
    s = load(root / 'logs/summary/round8_selection_results.json', {})
    if not s:
        return []
    peer = load(root / s['peer_review_path'], {})
    if peer.get('status') != 'ACTUAL_INDEPENDENT_CROSS_AND_FINAL_FACTUAL_REVIEWS_COMPLETED':
        raise ValueError('Final selection review is incomplete')
    reuse = load(root / 'logs/summary/round8_reuse_identifiability_audit.json', {})
    if reuse.get('status') != 'PASS' or s['new_Gemma_model_forwards'] != 0:
        raise ValueError('Selection audit must remain distinct from an actual model round')
    title = '## 이전 탐색(R8)의 종료 기록' if (root / 'logs/summary/round9_progress.json').exists() else '## 후속 선정 감사와 자율 탐색 종료'
    out = ['', title, '',
           '**논문으로 발전시킬 가치가 검증된 새 주제를 찾지 못했습니다.** 두 Hold와 새 제안을 실제 독립·교차 심사하고, 식별 반례를 CPU로 검산한 뒤 사용자 목표의 유효 질문 부재·반복 식별불가 종료 규칙을 적용했습니다. 과거 실측은 보존합니다.', '',
           f"후속 제안 {integer(s['screened_proposals'], 'screened proposals')}개는 모두 현재 No-Go이며, 실행 가능한 새 연구 선정은 {integer(s['selected_viable_new_model_studies'], 'new model studies')}개입니다. 이 선정 감사의 실제 Gemma forward는 **{integer(s['new_Gemma_model_forwards'], 'selection model forwards')}회**입니다. R8을 실제 모델 실험 라운드로 세지 않습니다.", '',
           f"합성 CPU countermodel은 원안 {integer(reuse['total_observables_compared'], 'synthetic original observations')}개, 수정안 {integer(reuse['same_input_different_table_repair']['observables_compared'], 'synthetic repair observations')}개 관측을 비교했습니다. 결과 재사용과 입력/표 저장 후 재계산이 같은 제안 효과를 만들 수 있었습니다. 실제 LLM 결과나 새로운 정리가 아닙니다.", '',
           '새 명제가 모든 미래에 없다는 뜻이 아닙니다. 경쟁 기제가 다른 결과를 예측하는 새 대비와 선행을 넘는 기여가 확보되어야 별도 연구를 재개할 수 있습니다.', '',
           '[최종 탐색 결과·다섯 제안·종료 근거](reports/AUTONOMOUS_DISCOVERY_OUTCOME_KO.md) · [실제 후속 공동 심사](logs/summary/round8_peer_review.json) · [누적 후보 순위](reports/PAPER_CANDIDATE_RANKING_KO.md)', '']
    return out


def round9_lines(root: Path) -> list[str]:
    s = load(root / 'logs/summary/round9_progress.json', {})
    if not s:
        return []
    final = load(root / 'logs/summary/round9_final_decision.json', {})
    out = ['', '## R9 자율 연구 실측과 현재 판단', '',
            ('새 다섯 P0와 실측 후 독립·교차 심사를 마쳤지만 논문 가치가 검증된 새 주제를 찾지 못했습니다. 목표 §4E·§8의 실패 탐색 종료 분기를 적용합니다. 이전 결과는 보존하며 모든 가능한 연구질문의 소진을 주장하지 않습니다.' if final else '이전 R8 실패 종료와 모든 실측은 보존하고 새로운 다섯 질문을 검증합니다. 아직 새 논문 주제를 찾았다는 주장은 없습니다.'), '',
            f"현재 상태: **{s['status']}**. 새로 선정한 모델 연구 {integer(s['model_studies_selected'], 'R9 selected studies')}개, 재개 이후 실제 Gemma forward **{integer(s['new_Gemma_model_forwards'], 'R9 actual forwards')}회**입니다. P0는 {s['new_P0']}, P1/P2는 {s['new_P1']}/{s['new_P2']}입니다.", '',
            '실제 GPT Pro 원전 검토와 Claude Opus5.5 Extra Research의 독립 답변 및 양쪽 적대적 교차답변을 수집했습니다. GPT의 별도 Deep Research 요청은 오류였으므로 완료 보고서로 세지 않습니다. 전송·원전 접근 한계와 의견 차이는 심사 기록에 보존합니다.', '',
            '[현재 상태와 한계](reports/ROUND9_REOPENING_KO.md) · [다섯 질문 사전등록](reports/ROUND9_P0_PREREG_KO.md) · [P0 예산·판정](configs/round9_P0.yaml) · [심사 기록](logs/summary/round9_peer_review.json) · [원전 확인 범위](logs/summary/round9_literature_audit.json)', '']
    measured = {k: load(root / f'logs/summary/round9_P0_{k}.json', {}) for k in 'ABCDE'}
    if any(measured.values()):
        out += ['[R9 실제 결과와 설계 한계](reports/ROUND9_EXECUTION_RESULTS_KO.md) · [원시 행·호출 재검산](logs/summary/round9_P0_replay.json) · [사후 형식·입력 진단](logs/summary/round9_readout_diagnostics.json)', '']
        out += ['| R9 질문 | 실제 과학 forward | 실행 gate | P0 분류 |', '|---|---:|---|---|']
        for k, v in measured.items():
            o = v.get('outcome') or {}
            out += [f"| {k} | {integer(v['actual_forward_calls'], 'R9 science calls') if v else 'NOT_EVALUATED'} | {cell(o.get('gate', 'NOT_EVALUATED'))} | {cell(o.get('classification', 'NOT_EVALUATED'))} |"]
        out += ['', '공유 기술 호출은 질문별 과학 호출에 중복 합산하지 않습니다. P0 Negative·미식별은 일반적인 현상 부재를 입증하지 않습니다.', '']
    if final:
        out += ['A/B/C/E는 이 계측에서 주 질문을 평가하지 못했습니다. D는 등록된 저차 정상·고차만 실패 공동 패턴의 Negative0/8이며, 저차 정상 조건이0개여서 그 조건 안의 고차 오류율은 추론할 수 없습니다. 현재 추가 GPU/P1은 No-Go입니다.', '',
                '[실측 후 공동 심사·후속 제안 선별·종료 근거](reports/ROUND9_REVIEW_AND_NEXT_KO.md) · [해석 유효성 guard](logs/summary/round9_interpretation_guard.json) · [현재 최종 판단](logs/summary/round9_final_decision.json)', '']
    if (root / 'figures/round9_P0_results.svg').exists():
        out += ['![R9 실제 P0와 한계](figures/round9_P0_results.svg)', '']
    return out


def round10_lines(root: Path) -> list[str]:
    s = load(root / 'logs/summary/round10_progress.json', {})
    if not s:
        return []
    out = ['', '## R10 A/B/C/E 계측 수리와 새 데이터 실험', '',
           '사용자의 명시적 재실험 요청에 따라 R9에서 평가하지 못한 네 질문의 계측을 수리합니다. 원래 R9 결과·판정과 D는 보존합니다. 수리 성공, 가설의 결과, 논문 가치는 별도로 평가합니다.', '',
           f"현재 상태: **{s['status']}**. 실제 기술 forward {integer(s['new_technical_forwards'], 'R10 technical calls')}회, calibration {integer(s['new_calibration_forwards'], 'R10 calibration calls')}회, 새 P0 {integer(s['new_P0_forwards'], 'R10 P0 calls')}회입니다. P1/P2는 {s['P1']}/{s['P2']}입니다.", '',
           '실제 GPT Pro와 Claude Opus5.5 Extra Research 독립 검토와 양쪽 교차답변을 반영했습니다. 전체 생성 후 최종 선언 판독, 문자·제시 위치 독립 교차, 기초 anchor, 새 세계·token 입력 분할, 미판독 최악 경계를 적용합니다.', '',
           '[과학 출력 전 사전등록](reports/ROUND10_REPAIR_PREREG_KO.md) · [예산·gates](configs/round10_repair.yaml) · [실제 설계 공동 심사](logs/summary/round10_peer_review.json) · [입력·소스 해시](logs/summary/round10_registration.json)', '']
    if s.get('R10b_registration_path'):
        out += [f"원래 R10은 A 보정 {integer(s['first_attempt_completed_responses'], 'R10 partial responses')}개(유효 {integer(s['first_attempt_valid_responses'], 'R10 partial valid')}, 토큰 상한 {integer(s['first_attempt_capped_responses'], 'R10 partial capped')}) 뒤 추가 공학적 중단으로 종료했습니다. 당시 모든 P0는 미평가였습니다. 원래 소스와 실패를 보존한 별도 R10b에 간결 출력 지시·A 직접 공지 anchor·새 분할·질문별 중단 규칙을 등록했습니다.", '',
                '[R10b 새 사전등록](reports/ROUND10B_REPAIR_PREREG_KO.md) · [R10b 실제 독립·교차 심사](logs/summary/round10b_peer_review.json) · [R10b 예산](configs/round10b_repair.yaml)', '']
    measured = load(root / 'logs/summary/round10_execution_results.json', {})
    if measured:
        out += ['[실측 결과와 한계](reports/ROUND10_REPAIR_RESULTS_KO.md)', '',
                '| 질문 | 새 P0 응답 | 실제 forward | gate | 가설 판정 |', '|---|---:|---:|---|---|']
        for row in measured.get('studies', []):
            out += [f"| {cell(row['study'])} | {integer(row['P0_responses'], 'R10 P0 responses')} | {integer(row['P0_forwards'], 'R10 P0 calls')} | {cell(row['gate'])} | {cell(row['classification'])} |"]
        out += ['', '한 응답은 여러 prefill/decoding forward를 사용합니다. 미판독은 의미상 NO가 아니며, 실제 main 후 넓은 구간은 측정한 Inconclusive입니다.', '']
        if (root / 'figures/round10_readout_results.svg').exists():
            out += ['![R10 최종답 판독과 실행 범위](figures/round10_readout_results.svg)', '']
    return out


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
    execution_day = load(root / "logs/summary/round6_execution_results.json", {}).get("date_kst")
    if execution_day and re.fullmatch(r"\d{4}-\d{2}-\d{2}", execution_day):
        day = max(day, execution_day) if day else execution_day
    if day:
        out += [f"기록 기준: **{day} KST**. 아래 `PASS`는 해당 실행·기술 검증 상태입니다. "
                "뉴런 기능·인과성의 검증 상태는 카탈로그에서 별도로 표시합니다.", ""]
    ranking = load(root / 'logs/summary/paper_candidate_ranking.json', {})
    latest_round = load(root / 'logs/summary/round7_execution_results.json', {})
    if ranking and latest_round:
        out += [f"최근 R7은 다섯 P0에 실제 {integer(latest_round['total_actual_forwards'], 'latest R7 forwards')}회 forward를 수행하고 원시 행 재검산 및 GPT·Claude 사후 교차검토를 마쳤습니다. 현재 논문 Go는 **{integer(ranking['Go_count'], 'paper Go')}개**입니다. [실측과 한계](reports/ROUND7_EXECUTION_RESULTS_KO.md) · [R7 당시 심사·후속 제안](reports/ROUND7_REVIEW_AND_NEXT_KO.md)", ""]
        if (root / 'logs/summary/round10_progress.json').exists():
            out += ['**현재 R10 계측 수리 재실험:** A/B/C/E는 R9에서 평가하지 못한 질문입니다. 사용자의 명시적 요청으로 실제 공동 설계 검토를 거쳐 새 데이터를 등록했습니다. [현재 범위와 판정](reports/ROUND10_REPAIR_PREREG_KO.md).', '']
        elif (root / 'logs/summary/round9_progress.json').exists():
            final = load(root / 'logs/summary/round9_final_decision.json', {})
            out += [('**R9 실측 후 탐색 종료:** 실제 새 다섯 P0와 양쪽 사후 독립·교차 검토를 마쳤지만 논문 가치가 검증된 새 주제는 찾지 못했습니다. 현재 탐색풀에 목표의 허용된 실패 종료 조건을 적용했습니다. [근거와 한계](reports/ROUND9_REVIEW_AND_NEXT_KO.md).' if final else '**자율 연구 재개(R9):** 사용자의 새 활성 목표에 따라 다른 연구 방향의 독립 조사를 진행 중입니다. 이전 R8의 실패 종료는 역사적 기록이며 현재 목표의 완료를 뜻하지 않습니다. [현재 상태](reports/ROUND9_REOPENING_KO.md).'), '']
        elif (root / 'logs/summary/round8_selection_results.json').exists():
            out += ['**자율 탐색 종료:** 후속 독립·교차검토와 식별 반례 감사 뒤에도 논문 가치가 검증된 새 명제를 찾지 못했습니다. [최종 결과와 종료 근거](reports/AUTONOMOUS_DISCOVERY_OUTCOME_KO.md).', '']
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
    out += round3a_lines(root)
    out += round4_lines(root)
    out += round5_lines(root)
    out += round6_lines(root)
    out += round7_lines(root)
    out += round8_selection_lines(root)
    out += round9_lines(root)
    out += round10_lines(root)
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
