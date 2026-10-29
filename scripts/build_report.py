#!/usr/bin/env python3
"""Render a deterministic Korean report from recorded evidence, without model/GPU use.

The default command writes reports/RESEARCH_SUMMARY_KO.md. --stdout only renders;
--check verifies that the existing report exactly matches the current sources.
Missing measurements are never replaced by zeros. Scientific interpretations
come from the recorded synthesis rather than thresholds invented here.
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
DESTINATION = "reports/RESEARCH_SUMMARY_KO.md"
REPOSITORY = "https://github.com/PigeonLabs/Gemma4_neuron_LAB"
STAGES = ("smoke", "pilot", "survey", "steering", "validation", "expanded")
CONDITIONS = (("add_-1sd", "−1 SD"), ("add_+1sd", "+1 SD"),
              ("zero_ablation", "Zero"), ("random_neuron_projected", "Random neuron · projected"),
              ("random_direction_projected", "Random direction · projected"))
ROLES = (("positive_control", "Positive"), ("negative_control", "Negative"),
         ("near_miss", "Near-miss"), ("paraphrase", "Paraphrase"),
         ("out_of_distribution", "OOD"))
NE = "NOT_EVALUATED"


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def load(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_constant) if path.is_file() else default


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()


def numeric(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid recorded numeric value")
    return float(value)


def num(value, digits=5) -> str:
    n = numeric(value)
    return NE if n is None else f"{n:.{digits}g}"


def count(value) -> str:
    n = numeric(value)
    if n is None:
        return NE
    if n < 0 or not n.is_integer():
        raise ValueError("Invalid recorded count")
    return f"{int(n):,}"


def gib(value) -> str:
    n = numeric(value)
    return NE if n is None else f"{n / 1024**3:.3f}"


def state(value) -> str:
    return cell(value.get("status", NE)) if isinstance(value, dict) else NE


def is_pass(value) -> bool:
    return state(value) == "PASS"


def interval(value: dict, mean_key="mean") -> str:
    if not isinstance(value, dict) or value.get(mean_key) is None:
        return NE
    low, high = numeric(value.get("ci_low")), numeric(value.get("ci_high"))
    if low is None or high is None:
        return f"{num(value[mean_key])} [CI: {NE}]"
    if low > high:
        raise ValueError("Reversed recorded confidence interval")
    return f"{num(value[mean_key])} [{num(low)}, {num(high)}]"


def link(root: Path, label: str, relative: str) -> str:
    path, _, anchor = relative.partition("#")
    if not (root / path).is_file():
        return NE
    return f"[{cell(label)}](../{path}" + (f"#{anchor}" if anchor else "") + ")"


def evidence(root: Path, value) -> str:
    if isinstance(value, list):
        return " · ".join(filter(None, (evidence(root, item) for item in value)))
    if isinstance(value, dict):
        path = value.get("path") or value.get("url")
        return evidence(root, path) if path else cell(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if not value:
        return ""
    value = str(value)
    if value.startswith(("https://", "http://")):
        return f"[근거]({value})"
    return link(root, "근거", value) if (root / value.split("#", 1)[0]).is_file() else cell(value)


def validate_freshness(root: Path, summaries: dict, analysis: dict, catalog: list) -> None:
    if not is_pass(analysis):
        return
    receipts = {r["stage"]: r for r in analysis.get("stage_sources", [])}
    for stage in ("steering", "validation", "expanded"):
        if is_pass(summaries[stage]):
            receipt = receipts.get(stage, {})
            if not is_pass(receipt) or receipt.get("run_id") != summaries[stage].get("run_id"):
                raise ValueError(f"Analysis is stale for {stage}; run scripts/analyze_results.py first")
    for relative, expected in analysis.get("source_sha256", {}).items():
        path = root / relative
        # Local raw artifacts can be archived; public evidence must remain available.
        if path.is_file() and digest(path) != expected:
            raise ValueError(f"Analysis source changed: {relative}")
        if relative.startswith(("configs/", "logs/summary/")) and not path.is_file():
            raise ValueError(f"Missing public analysis source: {relative}")
    by_id = {c["id"]: c for c in analysis.get("candidates", [])}
    for entry in catalog:
        source = by_id.get(entry.get("id"))
        if source and any(entry.get(k) != source.get(k) for k in ("correlation", "causal_signature", "counterexamples")):
            raise ValueError("Catalog differs from analysis; regenerate both together")


def render(root: Path = ROOT) -> str:
    names = (*STAGES, "model_load", "model_download", "analysis", "synthesis", "resources", "publication", "utility")
    summaries = {name: load(root / f"logs/summary/{name}.json", {}) for name in names}
    recorded_stages = STAGES + (("utility",) if summaries["utility"] else ())
    model, smoke, survey = (summaries[x] for x in ("model_load", "smoke", "survey"))
    analysis, synthesis = summaries["analysis"], summaries["synthesis"]
    frozen = load(root / "configs/hypotheses.json", {})
    catalog = load(root / "catalog/public/neurons.json", [])
    if isinstance(catalog, dict):
        catalog = catalog.get("neurons", catalog.get("entries", []))
    if not isinstance(catalog, list) or any(not isinstance(c, dict) for c in catalog):
        raise ValueError("Invalid public neuron catalog")
    validate_freshness(root, summaries, analysis, catalog)
    candidates = frozen.get("candidates", [])
    by_analysis = {c["id"]: c for c in analysis.get("candidates", [])}
    by_catalog = {c["id"]: c for c in catalog}
    out = ["# Gemma 4 Neuron Lab — 연구 종합 보고서", "",
           "실제 실행한 **첫 번째 유한한 연구 주기**의 기록이다. 구현 여부·실험 실행 PASS·과학적 가설 상태를 구별한다. "
           "동결한 text-only 실험과 다음 연구 후보를 정리한다.", ""]
    r2 = load(root / "logs/summary/round2_ef_analysis.json", {})
    if r2.get("status") == "PASS" and r2.get("analysis_phase") == "confirmation":
        ce = r2["phases"]["confirmation"]["E"]
        cf = r2["phases"]["confirmation"]["F"]["groups"]["translate"]
        out += [f"**후속 Round 2 E/F도 완료했다.** {count(r2['total_condition_rows'])}개 조건 행·{count(r2['total_forward_calls'])}회 실제 BF16 forward. "
            f"5번은 ΔR²={num(ce['primary']['delta_r2'])}의 control-probe-assisted 예측 개선, 6번은 손상 gate 실패로 {cf['hypothesis_status']}다. "
            "원 연구의 수치를 섞지 않고 [새 연구의 전체 결과](ROUND2_EF_RESULTS_KO.md)에 설계·통계·한계를 별도로 기록했다. "
            "[GPT·Claude의 실측 재검토와 다음 추천](ROUND3_RECOMMENDATIONS_KO.md)은 후속 설계이며 새 실험 결과가 아니다.", ""]
    r3a = load(root / "logs/summary/round3a_p0.json", {})
    if r3a.get("status") == "PASS":
        out += [f"**추천 1번 R3-A는 실제 P0 {count(r3a['forward_calls'])} forward 후 수치 gate에서 중단했다.** "
            "같은 저장 값을 쓰는 BF16/FP32 비교에서 네 좌표 모두 등록 기준에 미달해 discovery·확증은 NOT_EVALUATED다. "
            "[실측 수치와 해석 한계](ROUND3A_RESULTS_KO.md)를 별도로 기록했다.", ""]
    if is_pass(model):
        if model.get("parameter_dtypes") != ["torch.bfloat16"]:
            raise ValueError("Model PASS does not document exclusively BF16 parameters")
        if model.get("layers") is not None and model.get("intermediate_size") is not None and model.get("neuron_count") is not None:
            if model["layers"] * model["intermediate_size"] != model["neuron_count"]:
                raise ValueError("Runtime neuron dimensions do not reconcile")
        out += [f"실제 BF16 모델에서 **{count(model.get('layers'))}계층 × {count(model.get('intermediate_size'))} = "
                f"{count(model.get('neuron_count'))}개 MLP 중간 좌표**를 확인했다. "
                + (f"Discovery {count(survey.get('prompts'))} prompts·계층당 {count(survey.get('tokens_per_layer'))} user-content tokens를 관측했다. " if is_pass(survey) else "")
                + f"모델 revision은 `{cell(model.get('revision', NE))}`이다.", ""]
    out += ["종합 상태: **" + state(synthesis) + "**. " + cell(synthesis.get("scope", "종합 판단은 아직 기록되지 않았다.")), "",
            "초기 CP4 증거: " + link(root, "INITIAL_FINDINGS_KO.md", "reports/INITIAL_FINDINGS_KO.md") + ".", "",
            "## 1. 실제 구현한 기능과 검증 범위", "",
            "| 기능 | 구현 산출물 | 실제 검증 |", "|---|---|---|"]
    components = [("BF16 native adapter·capture·개입·hook 제거", "src/neuron_lab/hooks.py", state(smoke)),
                  ("GPU streaming 통계·Parquet/JSON/SQLite 기록", "src/neuron_lab/statistics.py", state(survey)),
                  ("강도 sweep·seeded norm controls·restoration", "src/neuron_lab/steering.py", state(summaries["steering"])),
                  ("동결 가설·반례·독립 분할 재분석", "scripts/analyze_results.py", state(analysis)),
                  ("공개 neuron catalog", "catalog/public/neurons.json", f"{len(catalog)}개 측정 기록" if catalog else NE),
                  ("로컬 Explorer·Live Steering", "explorer/app.py", link(root, "실행 QA 기록", "docs/EXPLORER_QA.md"))]
    for label, path, validation in components:
        out.append(f"| {label} | {link(root, '구현', path)} | {validation} |")
    out += ["", "Codex는 discovery 증거로 경쟁 가설과 반례를 작성했다. Hook API의 multiplicative/replacement 지원과 실제 모델에서 평가한 조건은 같은 범위가 아니다. "
            "이미지·음성·외부 계측 engine·대표 행동 정확도는 별도 미평가 범위다.", "",
            "## 2. 실행된 실험과 Run 기록", "",
            "| 단계 | 상태 | Run ID | 원본 corpus / 추가 사례 | 조건–prompt rows | 시간(s) | Peak reserved(GiB) |",
            "|---|---|---|---:|---:|---:|---:|"]
    for stage in recorded_stages:
        d = summaries[stage]
        corpus_n = d.get("corpus_prompt_count", d.get("prompts", d.get("prompt_count")))
        extra = count(d.get("counterexample_prompt_count")) if "counterexample_prompt_count" in d else "해당 없음"
        row_count = count(d["intervention_count"]) if "intervention_count" in d else (f"생성 {count(d['generation_count'])}" if "generation_count" in d else "해당 없음")
        out.append(f"| {stage} | {state(d)} | `{cell(d.get('run_id', NE))}` | {count(corpus_n)} / {extra} | "
                   f"{row_count} | {num(d.get('elapsed_seconds'))} | {gib(d.get('peak_reserved_bytes'))} |")
    out += ["", f"Smoke: {count(smoke.get('passed'))}/{count(smoke.get('total'))} checks; read-only 최대 logit 차이 "
            f"{num(smoke.get('max_read_only_logit_difference'))}. Activation의 표본 단위는 사용자 본문 token·prompt이고, 조건 row나 random seed를 독립 prompt로 세지 않는다.", "",
            "## 3. 실패한 실행과 실패 이유", ""]
    failures = [(name, d) for name, d in summaries.items() if state(d) == "FAIL"]
    if failures:
        out += [f"- `{name}` / `{cell(d.get('run_id', NE))}`: {cell(d.get('error', d.get('reason', '상세 로그 참조')))}."
                for name, d in failures]
    else:
        out.append("읽은 stage 요약에 `FAIL` 실행은 기록되어 있지 않다. 누락된 단계는 NOT_EVALUATED이며 성공으로 간주하지 않는다. 과학적 음성 결과·반례는 8–9절의 등록된 종합 판단과 구별한다.")
    out += ["", "## 4. 후보 뉴런과 경쟁 가설", "",
            "| ID | Layer / neuron (0-based) | 동결 가설 | 경쟁 설명 | 카탈로그 상태 |", "|---|---|---|---|---|"]
    for c in candidates:
        out.append(f"| {cell(c.get('id'))} | {count(c.get('layer'))} / {count(c.get('neuron'))} | {cell(c.get('hypothesis', 'Unknown'))} | "
                   f"{cell('; '.join(c.get('competing_hypotheses', [])))} | {cell(by_catalog.get(c.get('id'), {}).get('status', NE))} |")
    if not candidates:
        out.append(f"| {NE} | {NE} | Unknown | {NE} | {NE} |")
    out += ["", "가설 동결 KST: " + cell(frozen.get("frozen_kst", NE)) + ". 선택은 discovery에 한정했다. "
            "관측 분리도는 범주 어휘·문자·길이·구문과 혼동될 수 있으며, 의미 기능의 인과 라벨이 아니다.", "",
            "## 5. 정량 Activation 및 Steering 결과", ""]
    if is_pass(survey):
        out.append(f"Survey: {count(survey.get('prompts'))} prompts / {count(survey.get('categories'))} categories / "
                   f"계층당 {count(survey.get('tokens_per_layer'))} tokens / {count(survey.get('neuron_count'))}좌표. "
                   f"Quantile 방법: {cell(survey.get('quantile_method', NE))}. "
                   "범주 평균은 prompt 균형, token moments는 token 가중이다. Quantile의 작은 reservoir는 정밀한 꼬리 추정을 보장하지 않는다.")
    else:
        out.append("전 계층 활성 통계: " + state(survey) + ".")
    out += ["", "Baseline 대비 next-token KL(P_base || P_changed), nats. 평균 [95% percentile bootstrap CI]; "
            "random seed는 prompt 안에서 평균한 후 독립 corpus prompt를 resample했다. 반례 family는 이 CI에 섞지 않았다.", "",
            "| 단계 / 후보 | 조건 | n prompts / rows | KL 평균 [95% CI] | 첫 token top-1 변화 비율 |", "|---|---|---:|---|---:|"]
    display_stage = next((s for s in ("expanded", "validation", "steering") if is_pass(summaries[s])), None)
    out.append("| 표 범위 | " + (cell(display_stage) if display_stage else NE) + " · 가장 최근 완료한 분할 | — | — | — |")
    for stage in ([display_stage] if display_stage else ["steering"]):
        d = summaries[stage]
        rows = {(r["candidate_id"], r["condition"]): r for r in d.get("aggregates", [])} if is_pass(d) else {}
        for c in candidates:
            cid = c["id"]
            if not any(key[0] == cid for key in rows):
                out.append(f"| {stage} / {cell(cid)} | {NE} | {NE} | {NE} | {NE} |")
                continue
            for condition, label in CONDITIONS:
                row = rows.get((cid, condition), {})
                metric = row.get("metrics", {})
                out.append(f"| {stage} / {cell(cid)} | {label} | {count(row.get('n_prompts'))} / {count(row.get('n_interventions'))} | "
                           f"{interval(metric.get('kl_base_to_changed', {}))} | {num(metric.get('top1_changed', {}).get('mean'))} |")
    out += ["", "조건 표는 가장 최근 완료한 분할에 한정했다. 이전 단계의 모든 조건은 "
            + " · ".join(link(root, s, f"logs/summary/{s}.json") for s in ("steering", "validation", "expanded")) + "에 보존되어 있다.",
            "생성은 사전 선정한 소수 prompt의 greedy 출력에 한정했다.",
            "개입 관찰치는 다음과 같다. 같은 prompt의 반복 조건은 새로운 독립 표본이 아니며, collapse proxy는 의미 정확도·일반 성능 검증이 아니다."]
    for stage in ("steering", "validation", "expanded"):
        d = summaries[stage]
        if not is_pass(d):
            out.append(f"- {stage}: {state(d)}.")
            continue
        gens = [r for r in d.get("aggregates", []) if r.get("condition") != "baseline" and isinstance(r.get("generation_n"), int) and r["generation_n"] > 0]
        if gens and all(r.get("generation_changed_fraction") is not None for r in gens):
            n = sum(r["generation_n"] for r in gens)
            fraction = sum(r["generation_n"] * numeric(r["generation_changed_fraction"]) for r in gens) / n
            out.append(f"- {stage}: 고정 prompt {count(len(d['generation_prompt_ids'])) if 'generation_prompt_ids' in d else NE}개; "
                       f"비-baseline 생성 비교 {count(n)}개, token sequence 변화 비율 {num(fraction)}. "
                       f"정렬된 기술적 restoration logits exact: {cell(d.get('restoration_logit_exact', NE))}.")
        else:
            out.append(f"- {stage}: 생성 비교 {NE}; 기술적 restoration logits exact: {cell(d.get('restoration_logit_exact', NE))}.")
    out += ["", "## 6. 무작위 대조군과 실제 norm", "",
            "대조군은 |dose|=1 SD에서만 명목상 activation 또는 raw down-projection norm을 맞춘다. ±0.5/±2 SD와 직접 크기 비교해 선택성을 주장하지 않는다. "
            "BF16 반올림 뒤 실제 norm을 별도로 기록했다. 아래 비율은 candidate +1 SD / projected random-direction control의 실제 raw-down norm 비율이다.", "",
            "| 후보 / 분할 | n prompts / control seeds | 비율 평균 [관측 min, max] |", "|---|---:|---|"]
    for c in candidates:
        a = by_analysis.get(c["id"], {})
        for split in ("discovery", "validation", "test"):
            n = a.get("norm_matching", {}).get(split, {})
            ratios = n.get("pair_ratios", {})
            out.append(f"| {cell(c['id'])} / {split} | {count(n.get('n_prompts'))} / {count(n.get('n_control_seeds'))} | "
                       + (f"{num(ratios.get('mean'))} [{num(ratios.get('min'))}, {num(ratios.get('max'))}]" if is_pass(n) else state(n)) + " |")
    out += ["", "관측 통계는 user-content 평균이지만 next-token 개입은 마지막 chat-wrapper token에 적용한다. "
            "생성에서는 매 forward의 마지막 token에 적용한다. Post-MLP RMSNorm 뒤 최종 residual norm까지 일치했다고 해석하지 않는다. "
            "Restoration은 zero로 손상한 동일 입력 좌표를 clean 값으로 되돌리는 기술 검증이며 의미적 mediation 증거가 아니다.", "",
            "## 7. 독립 검증·사전 등록된 분포 효과·반례", "",
            "Held-out AUC는 사용자 본문 prompt-mean activation의 고정 target-vs-rest 판별이다. "
            "양·음 표본 수와 2-sided permutation p의 Holm 보정값을 함께 표시한다. 각 분할의 고정 4가설이 하나의 family다.", "",
            "| 후보 / 분할 | n positive / negative | AUC | Holm p |", "|---|---:|---:|---:|"]
    for c in candidates:
        a = by_analysis.get(c["id"], {})
        for split in ("validation", "test"):
            v = a.get("correlation", {}).get(split, {})
            out.append(f"| {cell(c['id'])} / {split} | {count(v.get('n_positive'))} / {count(v.get('n_negative'))} | {num(v.get('auc'))} | {num(v.get('permutation_p_holm'))} |")
    out += ["", "등록한 분포 signature는 target의 (candidate +1 SD KL − seed 평균 projected-direction KL)에서 other의 같은 값을 뺀 차이다.",
            "95% stratified prompt bootstrap CI이며 의미 행동의 인과 효과를 뜻하지 않는다. Validation과 test 모두 CI 하한 양수인지 사전 조건을 평가한다.", "",
            "| 후보 / 분할 | n target / other | Signature 평균 [95% CI], nats | CI 하한 > 0 |", "|---|---:|---|---|"]
    for c in candidates:
        a = by_analysis.get(c["id"], {})
        for split in ("validation", "test"):
            v = a.get("causal_signature", {}).get(split, {})
            out.append(f"| {cell(c['id'])} / {split} | {count(v.get('n_positive'))} / {count(v.get('n_negative'))} | "
                       f"{interval(v, 'interaction_mean')} | {cell(v.get('positive_lower_bound', NE))} |")
    out += ["", "반례는 각 candidate에 소유된 사례만 집계했다. Positive 원본과 paraphrase는 같은 source_id family로 묶고, "
            "family 평균 high-vs-low AUC 및 별도 4가설 Holm p를 계산한다. Role 평균의 n은 prompt 수, family AUC의 n은 독립 source family 수다.", "",
            "| 후보 | Role별 n / 평균 activation | Family n high / low | Family AUC / Holm p | 역전 / 동률 pairs |", "|---|---|---:|---|---:|"]
    for c in candidates:
        v = by_analysis.get(c["id"], {}).get("counterexamples", {})
        family = v.get("family_level", {})
        roles = "; ".join(f"{label} {count(v.get('roles', {}).get(key, {}).get('n'))}/{num(v.get('roles', {}).get(key, {}).get('mean'))}" for key, label in ROLES) if is_pass(v) else state(v)
        out.append(f"| {cell(c['id'])} | {roles} | {count(family.get('n_positive'))} / {count(family.get('n_negative'))} | "
                   f"{num(family.get('auc'))} / {num(family.get('permutation_p_holm'))} | {count(family.get('n_pairwise_inversions'))} / {count(family.get('n_pairwise_ties'))} |")
    utility = summaries["utility"]
    if utility:
        out += ["", "### 별도 post-validation utility 점검", "",
                "원래 동결 test의 확인적 평가와 구별하는 별도 사례집이다. "
                f"상태 {state(utility)}, Run `{cell(utility.get('run_id', NE))}`, prompt {count(utility.get('prompt_count'))}개, "
                f"생성 {count(utility.get('generation_count'))}개 / 최대 {count(utility.get('max_new_tokens'))} tokens. "
                f"Validation 이후 설계 flag: {cell(utility.get('design_after_validation', NE))}. "
                + link(root, "원자료 요약·채점·한계", "logs/summary/utility.json") + "."]
        if is_pass(utility):
            if model.get("revision") and utility.get("model_revision") != model["revision"]:
                raise ValueError("Utility model revision differs from the measured model")
            config_path = root / "configs/utility.yaml"
            if not config_path.is_file() or digest(config_path) != utility.get("config_sha256"):
                raise ValueError("Utility configuration differs from the frozen measured configuration")
            raw_path = root / "runs" / utility.get("run_id", "") / "utility_generations.jsonl"
            if raw_path.is_file() and digest(raw_path) != utility.get("raw_sha256"):
                raise ValueError("Utility raw generation hash differs from its summary")
            score_rows = {(r["candidate_id"], r["condition"], r["group"]): r for r in utility.get("aggregates", [])}
            out += ["", "표의 각 개입 cell은 정답 수/n 및 baseline 대비 paired accuracy Δ [95% CI]다. "
                    "Baseline 출력은 후보 사이에 공유하므로 반복 표시를 독립 표본으로 세지 않는다. Accepted-answer exact score의 적용 범위이며 대표 benchmark·의미 기능의 인과 증명이 아니다.", "",
                    "| 후보 / group | Baseline correct/n | +1 SD correct/n; Δ [CI] | −1 SD correct/n; Δ [CI] | Zero correct/n; Δ [CI] | Projected RD correct/n; Δ [CI] |",
                    "|---|---:|---|---|---|---|"]
            for c in candidates:
                for group in ("general", "translation"):
                    prefix = (c["id"], group)
                    baseline = score_rows.get((prefix[0], "baseline", prefix[1]), {})
                    cells = [f"{count(baseline.get('correct'))}/{count(baseline.get('n_prompts'))}"]
                    for condition in ("add_+1sd", "add_-1sd", "zero_ablation", "random_direction_projected"):
                        row = score_rows.get((prefix[0], condition, prefix[1]), {})
                        cells.append(f"{count(row.get('correct'))}/{count(row.get('n_prompts'))}; {interval(row.get('paired_accuracy_delta', {}))}")
                    out.append(f"| {cell(c['id'])} / {group} | " + " | ".join(cells) + " |")
            out.append("Utility scope: " + cell(utility.get("scope", NE)) + "; scoring: " + cell(utility.get("scoring", NE)) + ".")
            out.append("Utility control scope: " + cell(utility.get("control_matching", NE)) + ".")
        else:
            out.append("Utility 정확도·paired 차이의 측정 결과: NOT_EVALUATED.")
    else:
        out += ["", "별도 post-validation utility 측정: NOT_EVALUATED. 설정 파일의 존재를 실행 결과로 간주하지 않는다."]
    out += ["", "## 8. 예상하지 못한 관찰과 등록된 종합 판단", ""]
    findings = synthesis.get("findings", [])
    if not isinstance(findings, list):
        raise ValueError("Synthesis findings must be a list")
    if findings:
        for f in findings[:8]:
            if not isinstance(f, dict) or not f.get("text"):
                raise ValueError("Each synthesis finding requires recorded text")
            out.append("- " + cell(f["text"]) + " " + evidence(root, f.get("evidence")))
    else:
        out.append("실제 음성 결과·경쟁 설명·예상 밖 현상의 종합 판단은 아직 기록되지 않았다. 아래 수치에서 기능 반증이나 새로운 발견을 자동 추론하지 않는다.")
    relationships = analysis.get("category_profile_relationships", {})
    if is_pass(relationships):
        out += ["", "Discovery category-profile 관계는 다음과 같다. 같은 15개 범주 평균 간 Pearson r의 기술 통계이며 "
                "선택된 후보와 범주 구성·어휘 혼동에 의존한다. 연결·상호작용·같은 기능·독립 재현의 증거가 아니다."]
        out.append("측정 pair: " + "; ".join(
            f"{cell(' / '.join(r.get('candidate_ids', [])))} r={num(r.get('pearson_r'))} (범주 평균 n={count(r.get('n_category_means'))})"
            for r in relationships.get("relations", [])[:6]) + ".")
    out += ["", "## 9. 반증된 가설과 보류한 라벨", ""]
    if catalog:
        counts = Counter(c.get("status", "unknown") for c in catalog)
        out.append("카탈로그의 실제 상태: " + " · ".join(f"`{cell(k)}` {v}개" for k, v in sorted(counts.items())) + ".")
        out.append("`falsified` 기록이 없는 경우에도 가설이 참이라는 뜻은 아니다. 반증·제한 해석은 8절의 등록된 종합 진술에만 근거하며, KL·AUC나 기술적 restoration만으로 의미 기능 라벨을 승격하지 않는다.")
    else:
        out.append("측정 카탈로그 상태: NOT_EVALUATED. 반증 라벨을 생성하지 않았다.")
    out += ["", "## 10. 다음 연구 후보와 추가 비용", "",
            "후속 실험은 등록한 실제 관찰에서 도출한다. 비용은 제안자가 적은 **예상 GPU 초**이며 실제 소모량이 아니다.", "",
            "| 우선순위 | 후보 | 검증할 문제 | 예상 GPU 초 |", "|---|---|---|---:|"]
    proposals = synthesis.get("next_experiments", [])
    if not isinstance(proposals, list):
        raise ValueError("Synthesis next_experiments must be a list")
    for item in proposals[:6]:
        out.append(f"| {cell(item.get('priority', NE))} | {cell(item.get('title', NE))} | {cell(item.get('text', NE))} | {num(item.get('estimated_gpu_seconds'))} |")
    if not proposals:
        out.append(f"| {NE} | 미기록 | 미기록 | {NE} |")
    out += ["", "## 11. 관련 선행 구현과 출처", "",
            "이번 환경에서 upstream 프로젝트의 실험을 재현한 것은 아니다. 아래는 실제 소스·구조·라이선스를 확인한 설계 참고이며 upstream 수치는 본 연구 수치에 섞지 않았다. "
            + link(root, "고정 revision·라이선스·경로 감사", "docs/REFERENCE_AUDIT.md") + ".", "",
            "- [Lyra neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): steering-first 후보 탐색·대조군·연구 로그.",
            "- [Apple ml-refusal-neurons](https://github.com/apple-aiml-research/ml-refusal-neurons): 단일 좌표 훅·평가 분리.",
            "- [EleutherAI Delphi](https://github.com/EleutherAI/delphi): 활성 증거·설명 가설·독립 점수화 분리. SAE feature와 native neuron은 구별했다.",
            "- [interp-engine](https://github.com/decoderesearch/interp-engine), [NNsight](https://github.com/ndif-team/nnsight), "
            "[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): 계측 point·backend parity 참고. 본 실험의 기준선은 native HF eager다.", "",
            "## 12. 남은 한계", "",
            "직접 작성한 작은 편의 corpus, held-out 범주별 소수 양성, discovery의 수십만 좌표 선택 편향이 있다. "
            "Greedy 생성의 소수 prompt·짧은 길이와 반복 proxy는 대표 정확도·일반 품질·모든 문맥 안정성을 평가하지 않는다. "
            "멀티모달·장문 benchmark·외부 engine parity·독립 모델 반복·행동의 의미적 인과 검증은 NOT_EVALUATED다."]
    if (root / "docs/RELATED_WORK.md").is_file():
        out += ["", link(root, "원전 논문 5편과 신규성·필요 대조 검토", "docs/RELATED_WORK.md") + ". 후속 연구의 신규성은 미평가다."]
    limitations = synthesis.get("limitations", [])
    if not isinstance(limitations, list):
        raise ValueError("Synthesis limitations must be a list")
    if limitations:
        out.append("")
    for limitation in limitations[:8]:
        out.append("- " + cell(limitation))
    out += ["", "## 13. 실제 계산·저장 자원", ""]
    resources, download = summaries["resources"], summaries["model_download"]
    env = model.get("environment", {})
    out += [f"GPU: {cell(env.get('gpu', NE))}. CPU: {cell(resources.get('cpu', NE))}. OS: {cell(resources.get('os', NE))}. "
            f"RAM 원기록: {cell(resources.get('memory', {}).get('MemTotal', NE))}.",
            f"Resource snapshot KST: {cell(resources.get('captured_kst', NE))}; 프로젝트 disk free {gib(resources.get('project_disk', {}).get('free'))} GiB. "
            "단일 시점 snapshot이며 stage peak와 구별한다.",
            f"모델 다운로드: {state(download)}, {gib(download.get('download_bytes'))} GiB / {num(download.get('elapsed_seconds'))}초. "
            f"실제 모델 로딩 {num(model.get('load_seconds'))}초 / allocated {gib(model.get('allocated_bytes'))} GiB."]
    elapsed = [d["elapsed_seconds"] for stage in recorded_stages if is_pass(d := summaries[stage]) and d.get("elapsed_seconds") is not None]
    if elapsed:
        out.append(f"표에 있는 PASS stage elapsed 합계: {num(sum(numeric(v) for v in elapsed))}초. "
                   "설치·다운로드·모델 로딩·개발 시간을 제외하며, GPU 이용률을 적분한 계산 시간이 아니다.")
    out += [f"실행 환경: Python {cell(env.get('python', NE))}, CUDA build {cell(env.get('cuda_build', NE))}; "
            + ", ".join(f"{cell(k)} {cell(v)}" for k, v in sorted(env.get("packages", {}).items())) + ".", "",
            "## 14. 재현 방법과 원본·해시", "",
            "프로젝트 루트에서 실행한다. HF cache를 프로젝트 디스크에 두고 deterministic cuBLAS 설정을 worker 시작 전에 적용한다. "
            "다음 명령은 게시된 discovery 요약·고정 가설을 유지한 채 독립 평가를 재실행한다. Test를 후보 재선정에 사용하지 않는다.", "", "```bash",
            "export HF_HOME=\"$PWD/data/huggingface\"", "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "python3 -m venv .venv", ".venv/bin/python -m pip install -r requirements.lock.txt",
            ".venv/bin/python -m pip install --no-deps -e .", ".venv/bin/python scripts/prepare_model.py",
            "for stage in smoke steering validation expanded shutdown; do",
            "  PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit \"$stage\"", "done",
            "PYTHONPATH=src .venv/bin/python -m neuron_lab.cli worker"]
    if (root / "scripts/run_utility.py").is_file() and (root / "configs/utility.yaml").is_file():
        out += ["PYTHONPATH=src .venv/bin/python scripts/run_utility.py --check",
                "PYTHONPATH=src .venv/bin/python scripts/run_utility.py"]
    out += [
            ".venv/bin/python scripts/analyze_results.py", ".venv/bin/python scripts/analyze_results.py --check",
            ".venv/bin/python scripts/make_figures.py", ".venv/bin/python scripts/make_steering_figures.py",
            ".venv/bin/python scripts/build_report.py", ".venv/bin/python scripts/build_report.py --check",
            ".venv/bin/python scripts/build_readme.py", ".venv/bin/python scripts/build_readme.py --check", "```", "",
            "Pilot·survey 수집도 다시 수행하려면 새 연구 사본에서 가설 동결 전에 `submit pilot`, `submit survey`를 실행하고 "
            "새 run ID와 가설을 등록한다. 기존 등록 가설 파일을 덮어쓰거나 원래 discovery 요약을 교체한 뒤 동일 연구로 분석하지 않는다.", "",
            "가중치·전체 activation·원본 생성 로그는 Git에 올리지 않는다. 로컬 `runs/<run_id>/`의 metadata·Parquet·NPZ·JSONL과 "
            "SQLite를 공개 run ID·artifact SHA256으로 연결한다. 재실행 시 현재 게시된 summary 파일은 갱신되지만 이전 Run은 보존된다. "
            "보고서의 해석은 기록된 synthesis.json에서 가져오므로 재현자가 수치 재계산 뒤 종합 진술의 근거도 검토해야 한다.", "",
            "| 공개 재현 입력 | SHA256 |", "|---|---|"]
    hash_inputs = ["configs/hypotheses.json", "configs/validation_cases.json", "requirements.lock.txt", "logs/summary/analysis.json", "logs/summary/synthesis.json"]
    if utility:
        hash_inputs.append("logs/summary/utility.json")
        hash_inputs.append("configs/utility.yaml")
    for relative in hash_inputs:
        path = root / relative
        out.append(f"| {link(root, relative, relative)} | `{digest(path) if path.is_file() else NE}` |")
    out += ["", "| 단계 | 원본 interventions.jsonl SHA256 |", "|---|---|"]
    for stage in ("steering", "validation", "expanded"):
        out.append(f"| {stage} | `{cell(summaries[stage].get('raw_sha256', NE))}` |")
    if utility:
        out.append(f"| utility (utility_generations.jsonl) | `{cell(utility.get('raw_sha256', NE))}` |")
    out += ["", "## 15. GitHub 체크포인트와 결과 링크", "",
            "공개 저장소: [PigeonLabs/Gemma4_neuron_LAB](" + REPOSITORY + "). 원격 반영 여부는 publication receipt에 기록된 값만 표시한다.", "",
            "| 체크포인트 | KST 날짜 | Commit | 원격 확인 |", "|---|---|---|---|"]
    checkpoints = summaries["publication"].get("checkpoints", [])
    for cp in checkpoints:
        sha = cp.get("commit", "")
        ref = f"[{sha[:8]}]({REPOSITORY}/commit/{sha})" if re.fullmatch(r"[a-f0-9]{40}", str(sha)) else "미기록"
        out.append(f"| {cell(cp.get('checkpoint', NE))} | {cell(cp.get('date_kst', NE))} | {ref} | "
                   f"{'확인됨' if cp.get('remote_verified') is True else '미확인'} |")
    if not checkpoints:
        out.append(f"| {NE} | {NE} | 미기록 | 미확인 |")
    out += ["", " · ".join(link(root, label, path) for label, path in (
        ("실험 분석", "logs/summary/analysis.json"), ("종합 근거", "logs/summary/synthesis.json"),
        ("Neuron catalog", "catalog/public/neurons.json"), ("방법론", "docs/METHODOLOGY.md"),
        ("연구 로그", "docs/RESEARCH_LOG.md"), ("초기 보고서", "reports/INITIAL_FINDINGS_KO.md"))), ""]
    for filename, caption in (("heldout_auc.svg", "동결 가설의 held-out activation 판별"),
                              ("counterexample_activation.svg", "반례 role·source-family 관찰")):
        if (root / "figures" / filename).is_file():
            out += [f"![{caption}](../figures/{filename})", ""]
    result = "\n".join(line.rstrip() for line in out).rstrip() + "\n"
    if len(result.splitlines()) > 300:
        raise ValueError("Report exceeds the 300-line bound; reduce recorded narrative length")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only deterministic freshness check")
    parser.add_argument("--stdout", action="store_true", help="Render without writing the report")
    args = parser.parse_args()
    expected = render()
    path = ROOT / DESTINATION
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            print("Research report is missing or stale; run scripts/build_report.py.", file=sys.stderr)
            return 1
        print("Research report matches recorded evidence.")
        return 0
    if args.stdout:
        print(expected, end="")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(path)
    print("Updated " + DESTINATION + " from recorded evidence.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"Report generation rejected inconsistent evidence: {exc}", file=sys.stderr)
        sys.exit(1)
