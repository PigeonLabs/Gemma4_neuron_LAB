#!/usr/bin/env python3
"""Render measured R2 results and static scientific figures from summary JSON."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE = ROOT / "logs/summary/round2_ef_analysis.json"
REPORT = ROOT / "reports/ROUND2_EF_RESULTS_KO.md"


def f(value):
    return "NOT_EVALUATED" if value is None else f"{value:.6g}"


def ci(item):
    return f"{f(item['mean'])} [{f(item['ci_low'])}, {f(item['ci_high'])}]"


def report(s):
    diagnostic = json.loads((ROOT / "logs/summary/round2_ef_diagnostics.json").read_text())
    publication = json.loads((ROOT / "logs/summary/round2_ef_publication.json").read_text())
    result_receipt = next((r for r in reversed(publication["checkpoints"]) if r["checkpoint"] == "R2_EF_RESULTS" and r["remote_verified"]), None)
    conf = s["phases"]["confirmation"]
    e, ff, q = conf["E"], conf["F"], conf["quality"]
    ft = ff["groups"]["translate"]
    core = e["primary"]
    rows = ["# Round 2: 5번·6번 실제 BF16 실험 결과", "",
        f"두 실험의 calibration과 확증을 모두 완료했다. 실행은 PASS이며, 5번의 등록 가설은 **{e['hypothesis_status']}**, 6번 translate군은 **{ft['hypothesis_status']}**다. "
        "실행 완료는 뉴런 기능이나 논문 신규성의 입증과 다르다.", "",
        f"총 **{s['total_condition_rows']:,}개 조건 행**, 검증·norm 보정을 포함해 **{s['total_forward_calls']:,}회 forward**를 실제 Gemma 4 12B BF16에서 수행했다. "
        "E는 calibration 32·확증 64 원본 family, F는 16·64 family다. 각 family의 선택지 교환, 부호와 대조 seed는 독립 표본으로 세지 않았다.", "",
        "[등록 설계](../docs/ROUND2_EF_PREREGISTRATION.md) · [고정 파라미터](../configs/round2_ef_frozen.json) · [계산 가능한 요약](../logs/summary/round2_ef_analysis.json)", "",
        "## 5번: 입력 취약도 공통 요인의 확증 예측", "",
        "목표는 네 후보 좌표에 ±0.5 calibration SD를 넣었을 때의 평균 |정답−오답 logit margin 변화|다. family 내 평균에 log(·+1e−6)를 적용한다. "
        "기본 모형은 원래 margin·top-1 margin·entropy·길이와 실제 후보 개입 geometry 2변수다. 확장 모형에 무작위 대조 반응에서 계산한 S를 추가했다. 모든 계수·표준화는 calibration에서 고정했다.", "",
        "| 확증 모형 | R² | MAE (log magnitude) |", "|---|---:|---:|"]
    names = {"baseline_only": "Baseline 4변수", "baseline_geometry": "Baseline + 실제 geometry", "baseline_geometry_controls": "Baseline + geometry + control S (1차)",
        "baseline_controls": "Baseline + control S", "zero_probe_chained": "Baseline→S→후보 (zero-probe 보조)",
        "s_native": "Geometry + native 대조 S", "s_direction_0": "Geometry + 방향 실현값 1 S", "s_direction_1": "Geometry + 방향 실현값 2 S"}
    rows += [f"| {names[k]} | {f(v['r2'])} | {f(v['mae'])} |" for k, v in e["models"].items()]
    rows += ["", f"1차 ΔR² = **{f(core['delta_r2'])}**, 95% family bootstrap CI [{f(core['delta_r2_ci_low'])}, {f(core['delta_r2_ci_high'])}]. "
        f"MAE 감소 = {ci(core['mae_reduction'])}, 두 1차 검정 Holm p = {f(core['holm_two_primary_p'])}.", "",
        "이것은 새 입력에서 대조 개입을 측정하고, 후보 개입의 geometry에도 조건화하는 **control-probe-assisted 예측**이다. "
        "추가 forward 없는 예측으로 표현하지 않는다. zero-probe 보조 모형은 원래 baseline 특징만 사용한다. 후보 의미 기능이나 잠재 공통 원인의 인과적 식별로도 해석하지 않는다.", "",
        "이번에는 각 입력에서 후보를 먼저 forward해 post-norm norm을 얻고 대조의 크기를 맞췄다. 따라서 geometry 변수를 뺀 Baseline+S 모형도 "
        "후보 개입을 측정하지 않는 독립 probe 프로토콜의 검증은 아니다. 결과는 측정된 geometry에 조건부인 새 family 전이이며, 호출 절감이나 "
        "후보 관측 전 예측을 입증하지 않는다. 후보와 대조가 공유하는 dose 정보가 얼마나 기여했는지는 다음 연구에서 분리해야 한다.", "",
        "| 큰 반응 family 제거 | 남은 family | ΔR² | 95% CI |", "|---|---:|---:|---|"]
    rows += [f"| 상위 {100*r['deleted_top_fraction']:.0f}% | {r['n_families']} | {f(r['delta_r2'])} | [{f(r['delta_r2_ci_low'])}, {f(r['delta_r2_ci_high'])}] |" for r in e["robustness"]]
    shares = e["concentration_top_family_shares"]
    rows += ["", f"후보 절대 반응 총합의 상위 1·3·5 family 몫은 각각 {f(shares['1'])}, {f(shares['3'])}, {f(shares['5'])}다. "
        f"후보와 대조의 log 반응 상관은 {f(e['candidate_control_magnitude_correlation'])}, 두 방향 실현값의 S 상관은 {f(e['direction_realization_correlation'])}다. "
        f"모든 개입의 unrestricted top-1 변경 {e['total_top1_flip_condition_rows']}개 조건 행은 {e['top1_flip_source_families']}개 source family에서 관측했다. "
        "이는 개입 반복을 독립 prompt로 세는 지표가 아니다.", "",
        "등록 기준을 통과했지만, 최종 R²는 약 0.14로 낮다. 방향 실현값별 R²도 다르므로 보편적인 입력 취약도를 확인했다고 표현하지 않는다. "
        "아래 과제별 분해는 그림을 확인한 뒤 추가한 **사후 기술 진단**이며, 등록된 확증 검정이나 새 가설 검정이 아니다.", "",
        "| 사후 진단: 과제 (각 16 family) | MSE 감소 | 전체 감소 기여율 | 과제 내 후보–대조 log 반응 r |", "|---|---:|---:|---:|"]
    rows += [f"| {r['task']} | {f(r['mse_reduction'])} | {100*r['share_of_total_squared_error_reduction']:.2f}% | {f(r['candidate_control_log_response_correlation'])} |" for r in diagnostic["tasks"]]
    rows += ["", f"과제별 평균을 뺀 후보–대조 상관은 {f(diagnostic['task_centered_candidate_control_correlation'])}다. "
        "예측 오차 감소의 약 92%는 문자열 과제에서 나왔고, 모든 과제에서 같은 크기의 예측 개선을 확인한 것은 아니다. "
        "과제 내 상관은 각 과제의 단일 shell에 조건부인 기술 통계이며, 새로운 과제로의 전이를 증명하지 않는다. "
        "[사후 진단 JSON](../logs/summary/round2_ef_diagnostics.json)은 주 분석과 분리했다.", "",
        "![확증 예측과 실제 반응](../figures/round2_e_predictions.svg)", "",
        "그림은 같은 64개 확증 family와 같은 축 범위를 사용한다. R²는 확증 target 평균 기준이며, 구간은 calibration 재적합 없이 family만 재표집했다.", "",
        "## 6번: H001 손상과 H003 복구", "",
        "주 손상 D는 L15 N9637을 같은 family의 반대 지시에서 얻은 자연 활성로 교체한다. R은 L26 N719의 calibration 지시별 자연 median clamp다. "
        "층 사이 raw scalar는 복사하지 않았다. 14개 등록 조건을 모두 실행했다. 같은 좌표 clean 복원은 기술적 양성 대조이며 의미적 복구 증거가 아니다.", "",
        "| 확증 지표 (nats) | Translate: 64 family | Copy: 64 family |", "|---|---:|---:|"]
    metrics = {"damage": "자연 counterfactual 손상", "interaction": "손상×복구 상호작용 I", "raw_recovery": "손상 후 raw 복구", "direct_steering": "Clean에서 R 단독 효과",
        "specificity": "I − native random I", "direction_specificity": "I − residual direction I", "oracle_block_recovery": "Prompt별 clean H003 oracle block 회복",
        "zero_damage": "Zero H001 손상 (2차)", "zero_interaction": "Zero 손상×복구 I (2차)", "joint_zero_nonadditivity": "Joint zero 손상의 비가산성 (2차)"}
    rows += [f"| {name} | {ci(ff['groups']['translate']['stats'][k])} | {ci(ff['groups']['copy']['stats'][k])} |" for k, name in metrics.items()]
    rows += ["", "I = (Y_D+R−Y_D)−(Y_R−Y_clean)이다. R 자체가 출력을 움직이는 효과를 차감한다. 양의 I만으로 동일 기능이나 직접 간선을 증명하지 않는다.", "",
        f"Translate 손상 gate(손상 평균의 95% CI 하한 >0.10 nats): **{'PASS' if ft['damage_gate_passed'] else 'FAIL'}**. "
        f"I의 Holm p = {f(ft['stats']['interaction']['holm_two_primary_p'])}. "
        f"Raw recovery fraction 상태는 **{ft['raw_recovery_fraction']['status']}**다."]
    if not ft["damage_gate_passed"]:
        rows += ["", "손상 gate가 실패했으므로 복구 비율은 계산하지 않는다. 이 결과는 '두 뉴런이 동등하다', '대체가 불가능하다', '기능이 없다' 어느 것도 입증하지 않는다. "
            "이 위치·자연 counterfactual 손상·고정 median 복구 조합으로는 대체 가능성을 식별할 수 없다는 판정이다. 더 강한 개입으로 결과를 만들지 않았다."]
    else:
        rows += ["", f"Raw fraction = {f(ft['raw_recovery_fraction'].get('estimate'))}; conditional fraction = {f(ft['conditional_recovery_fraction'].get('estimate'))}. "
            "이는 family 평균의 비율이며 개별 비율의 평균이 아니다. 대조 geometry와 copy 부작용을 확인한 뒤에만 범위를 제한해 해석한다."]
    rows += ["", f"H001 변경 뒤 H003의 자연 활성 변화(FS)는 translate에서 {ci(ft['stats']['first_stage_H003'])} activation units다. "
        "이 값과 oracle block은 보조 측정이다. 여러 매개자·잔차 경로가 있으므로 순수한 단일 매개 경로로 명명하지 않는다. "
        "Copy에서는 copy용 median을 쓰므로 translate와 같은 raw clamp의 off-target 이전을 검증하지 않는다.", "",
        "![손상·복구의 요인 대비](../figures/round2_f_factorial.svg)", "",
        "점은 family 평균, 수평선은 95% percentile bootstrap CI, 회색 세로선은 0이다. 각 mode에서 두 A/B 순서를 묶었다. 부가 지표 구간에는 다중비교 보정이 없다.", "",
        "## 측정 품질과 해석 범위", "",
        "| Split / 기본 과제 | 문항 수 | 강제 A/B 정답 | Unrestricted top-1 정답 |", "|---|---:|---:|---:|"]
    for phase, item in s["phases"].items():
        rows += [f"| {phase} / {r['study']} {r['task']} {r['mode']} | {r['cases']} | {r['forced_choice_correct']} | {r['raw_top1_correct']} |" for r in item["quality"]["baseline_accuracy"]]
    rows += ["", "정답 A/B logit이 같으면 강제 선택 정답으로 세지 않았다. argmax의 동률 처리 때문에 unrestricted 수치와 차이가 날 수 있다. "
        "Copy군의 원래 오류도 포함해 모든 등록 family를 분석했다. 정확도나 효과를 보고 확증 입력을 걸러내지 않았다.", "",
        "| 확증 geometry 점검 | Native random | Residual random directions |", "|---|---:|---:|"]
    native, residual = (q["E_norm_matching"][k] for k in ("random_neuron", "random_direction"))
    rows += [f"| E post-norm norm 오차 중앙값 | {f(native['median_norm_relative_error'])} | {f(residual['median_norm_relative_error'])} |",
        f"| E norm 오차 ±10% 이내 비율 | {f(native['fraction_within_10_percent'])} | {f(residual['fraction_within_10_percent'])} |", "",
        f"E 후보 개입 후 활성값의 calibration min–max 이탈 비율은 {f(q['E_candidate_outside_calibration_support_fraction'])}다. "
        f"F native random/candidate R norm 비율 중앙값은 {f(q['F_native_random_to_candidate_rescue_norm']['median'])}, 0.5–2 범위 비율은 {f(q['F_native_random_to_candidate_rescue_norm']['fraction_within_half_to_twice'])}다. "
        "Norm 불일치 때문에 native 좌표 특이성을 강하게 주장하지 않는다. 층당 random 좌표 1개는 random-coordinate 모집단에 대한 rank 검정을 허용하지 않는다.", "",
        f"F 같은-forward 기술 복원의 최대 logit 오차는 **{f(q['F_technical_restore_max_abs_logit_delta'])}**다. 매 로딩의 기존 BF16 검사 28개와 정렬 훅 검사 34개를 통과했고, 보호된 R1 요약·가설·입력·카탈로그 8개 해시도 유지됐다.", "",
        "이번 과제는 배경지식이 거의 필요 없는 합성 문제지만, 정해진 네 task shell에 조건부인 생성 인스턴스 평가다. 새로운 사람 작성 과제, 여러 모델, 본문 위치, 자유 생성, 일반적인 한국어 번역, 큰 dose의 효과는 미평가다. "
        "두 1차 검정만 Holm으로 조정했다. 가족 bootstrap 구간은 고정된 calibration·모델·대조 좌표에 조건부이며 전체 연구 선택 과정의 불확실성을 포함하지 않는다.", "",
        "Identity·복원의 bit 일치는 기술적 불변성 검사다. BF16의 작은 활성·잔차·logit 변화가 반올림으로 사라지는 해상도를 FP32 모델과 비교한 검사는 **NOT_EVALUATED**다. "
        "따라서 관측된 작은 값이나 0을 정확한 기능 부재로 해석하지 않는다. 실제 post-norm와 layer-output dose를 함께 기록해 반올림 전 요청량과 전달량을 구분했다.", "",
        "## 계산 자원과 재현", "",
        "| Phase | Run | Forward | 실행 초 | 최대 reserved GB |", "|---|---|---:|---:|---:|"]
    rows += [f"| {phase} | `{r['run_id']}` | {r['forward_calls']:,} | {r['elapsed_seconds']:.2f} | {r['peak_reserved_bytes']/1e9:.3f} |" for phase, r in s["source_runs"].items()]
    if result_receipt:
        remote = result_receipt["remote_sha"]
        rows += ["", f"검증된 GitHub 결과 checkpoint: [{remote[:8]}](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/{remote}). "
            "[게시 receipt](../logs/summary/round2_ef_publication.json)는 main의 SHA를 다시 조회한 기록이다."]
    rows += ["", "모델은 `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`다. RTX PRO 6000 Blackwell Max-Q, native eager BF16, batch 1, seed 20261029로 실행했다. "
        "환경·전체 source hash·raw SHA-256·로딩 시간·입력 위치는 run metadata와 phase 요약에 보존했다. 위 실행 시간은 각 phase의 본 측정이며 로딩·smoke를 뺀 값이다. 합산 budget 시간은 frozen JSON에 calibration, 최종 run 로그에 확증을 기록했다.", "",
        "- [Calibration 실행 요약](../logs/summary/round2_ef_calibration.json)",
        "- [확증 실행 요약](../logs/summary/round2_ef_confirmation.json)",
        "- [실행 코드](../scripts/run_round2_ef.py), [분석 코드](../scripts/analyze_round2_ef.py), [공개 family 요약 재계산](../scripts/replay_round2_ef.py)",
        "- CPU raw 재검사: `.venv/bin/python scripts/analyze_round2_ef.py --check` (로컬 raw 필요).",
        "- 공개 요약 재계산: `.venv/bin/python scripts/replay_round2_ef.py` (모델·raw 불필요).",
        "- Manifest 재생성: `.venv/bin/python scripts/run_round2_ef.py --prepare`. SHA-256은 결과 JSON과 같아야 한다. 전체 prompt manifest와 tensor/logit은 로컬에 보존한다.",
        "- GPU 재실행은 별도 연구 사본에서 한다. 이 lab의 Git helper는 `runs/git-metadata`를 사용한다. 일반 clone에서는 `git clone --bare . runs/git-metadata`로 로컬 기록용 metadata를 준비하고 생성 manifest 및 등록 소스를 해당 Git에 commit한 뒤 calibration을 실행한다. frozen JSON을 별도 commit한 뒤 confirmation을 실행한다. 기존 증거를 덮어쓰지 않는다.", "",
        "## 논문 방향과 다음 실험의 우선순위", "",
        "1. **소수 random probe가 보지 않은 뉴런·답변 코드의 반응으로 전이되는가.** E 양성을 일반화 가능한 발견으로 키우는 우선 실험이다. 주어진 사전·문자열·단순 비트 규칙 등 정답을 계산할 수 있는 과제를 쓴다. 새 calibration 32·확증 64 family에서 template 묶음과 원본을 분리하고, 후보당 dose는 calibration에서만 고정한다. Control은 새 입력의 후보 dose를 보고 맞추지 않고 층별 고정 post-norm norm으로 수행한다. A/B와 C/D처럼 서로 다른 single-token 답변 코드를 교차하고, 후보 하나를 학습에서 뺀 상태의 예측까지 등록한다. 1차량은 task-ID·margin·entropy·길이 baseline 대비 held-out ΔR²다. 각 과제 안에서 S를 섞는 음성 대조, 여러 고정 방향 ensemble, 제한된 FP32 수치 검증을 둔다. 새 코드·후보 전이의 CI 하한이 0 이하이거나 과제 내 S를 섞어도 동등한 개선이면 보편적 입력 요인 가설을 중단한다. 기존 token-position bias와 구별되는 **교차 전이**가 없으면 novelty를 주장하지 않는다.",
        "2. **단서 중복도가 signed 개입 효과를 가리는가.** E의 크기 예측을 넘어 메커니즘을 직접 조작하는 질문이다. 같은 정답·같은 지시에서 단서 수(1/2/4)를 바꾸고 규칙 준수 방향의 margin을 측정한다. 새 discovery 24 family에서 측정 가능한 후보를 고른 뒤 독립 64 family에서 후보−대조의 단서 수 기울기를 확증한다. Random 대조에도 같은 기울기가 나타나거나 후보−대조 CI가 0을 포함하면 중단한다. 마지막 본문 위치와 답변 경계를 비교하는 보조 arm으로 R1의 위치 불일치를 분리한다. 이번 E/F 결과로 중복 마스킹이 확인됐다고 미리 주장하지 않는다.",
        "3. **H001/H003 대체 연구는 손상 가능성부터 다시 확인한다.** 새 discovery에서 재현 가능한 의미적 손상이 생기는 과제·위치를 먼저 찾아 별도 등록한다. 새 확증 set에서 gate를 통과하기 전에는 rescue fraction·기능 동등성을 논문 중심 주장으로 삼지 않는다. 현 조건에서 dose만 키우는 후속은 추천하지 않는다.", "",
        "[기존 아이디어 검토](NOVELTY_IDEAS_ROUND2_KO.md)의 E/F 추천은 실행 전 우선순위였고, 이 문서의 실측 결과가 이후 판단의 근거다. 다음 제안은 새 연구 설계이며 아직 NOT_EVALUATED다.", "",
        "설계 단계에는 실제 GPT Pro Deep research와 Claude Opus 5.5 Extra Research의 완료 보고서를 반영했다. 실측 결과를 외부 채팅에 전달하는 추가 검토는 자동 승인 검토에서 거절되어 **NOT_EVALUATED**다. "
        "따라서 위 결과 해석과 사후 후속 실험 순위를 두 모델이 검증했다고 주장하지 않는다.", "",
        "## 선행연구와 기여의 경계", "",
        "[Analyzing the Generalization and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404)는 입력별 steerability와 spurious bias를 이미 다룬다. "
        "[SPARK](https://arxiv.org/abs/2607.10296)와 [Stress-Testing Random-Vector Controls](https://aclanthology.org/2026.blackboxnlp-1.73/)도 susceptibility 보정·대조 실현값 문제를 선행한다. 이번 E는 새로운 family에서 고정 예측식을 평가한 제한된 검증 사례다.", "",
        "[Every Ablation Is a Dose](https://arxiv.org/abs/2610.02173)는 개입을 signed counterfactual dose와 하류 gain으로 분석한다. "
        "[The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510)는 patching 추정량의 상호작용을 지적한다. 이번 F의 등록 gate와 standalone 차감은 이 한계를 다루며, 최초 발견이라는 근거는 아니다. "
        "GPT·Claude의 검색에서 겹치는 정확한 설계를 찾지 못했다는 보고는 신규성 증명이 아니다. 현재 수치만으로 논문 기여를 확정하지 않는다.", ""]
    return "\n".join(rows)


def figures(s):
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "svg.hashsalt": "round2-ef-fixed", "axes.spines.top": False, "axes.spines.right": False})
    c = s["phases"]["confirmation"]
    data = c["E"]["family_predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    keys = ("baseline_geometry", "baseline_geometry_controls")
    all_values = [r["y"] for r in data] + [r["predictions"][k] for r in data for k in keys]
    lo, hi = min(all_values) - .2, max(all_values) + .2
    tasks = {"provided_translation": ("#245A81", "o", "Dictionary"), "string_rule": ("#B36124", "s", "String"), "rule_entailment": ("#72783F", "^", "Logic"), "provided_facts": ("#A45078", "D", "Known / unknown")}
    for ax, key, title in zip(axes, keys, ("Baseline + measured geometry", "+ Measured control susceptibility")):
        ax.plot([lo, hi], [lo, hi], color="#777777", lw=1, ls="--")
        for task, (color, marker, label) in tasks.items():
            selected = [r for r in data if r["task"] == task]
            ax.scatter([r["y"] for r in selected], [r["predictions"][key] for r in selected], s=33, color=color, marker=marker, alpha=.8, label=label)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Observed log mean |semantic log-odds change|", title=title)
        metric = c["E"]["models"][key]
        ax.text(.04, .96, f"R² = {metric['r2']:.3f}\nMAE = {metric['mae']:.3f}", transform=ax.transAxes, va="top")
        ax.grid(alpha=.12)
    axes[0].set_ylabel("Calibration-frozen prediction")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("R2-E confirmation: 64 generated families, two choice orders averaged", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    output = {"round2_e_predictions": fig}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True)
    metrics = [("damage", "Natural-CF damage"), ("interaction", "Damage × rescue I"), ("direct_steering", "Standalone rescue effect"),
        ("specificity", "I minus native-random I"), ("direction_specificity", "I minus direction I"), ("oracle_block_recovery", "Oracle-block recovery")]
    for ax, mode, color in zip(axes, ("translate", "copy"), ("#245A81", "#B36124")):
        st = c["F"]["groups"][mode]["stats"]
        for i, (key, label) in enumerate(metrics):
            item = st[key]
            ax.errorbar(item["mean"], i, xerr=[[max(0, item["mean"] - item["ci_low"])] , [max(0, item["ci_high"] - item["mean"])]], fmt="o" if mode == "translate" else "s", color=color, capsize=3, lw=1.4)
        ax.axvline(0, color="#777777", lw=1)
        ax.scatter([.1], [0], marker="|", s=200, color="#333333", label="Damage floor: 0.10")
        ax.set(yticks=range(len(metrics)), yticklabels=[label for key, label in metrics], xlabel="Semantic log-odds contrast (nats)", title=f"{mode.title()}: 64 families")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.12)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("R2-F confirmation: mean and 95% family-bootstrap interval", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    output["round2_f_factorial"] = fig
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    s = json.loads(SOURCE.read_text())
    text = report(s)
    if args.check:
        if REPORT.read_text() != text:
            raise SystemExit("R2 report is stale")
    else:
        REPORT.write_text(text)
    figures_ = figures(s)
    for name, fig in figures_.items():
        stream = io.BytesIO()
        fig.savefig(stream, format="svg", metadata={"Date": None})
        raw = b"\n".join(line.rstrip() for line in stream.getvalue().splitlines()) + b"\n"
        path = ROOT / "figures" / f"{name}.svg"
        if args.check:
            if path.read_bytes() != raw:
                raise SystemExit(f"R2 figure is stale: {name}")
        else:
            path.write_bytes(raw)
            proof = ROOT / "runs/round2_ef_collaboration" / f"{name}.png"
            fig.savefig(proof, dpi=140)
        plt.close(fig)
    metadata = {"source": "logs/summary/round2_ef_analysis.json", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "figures": {name: hashlib.sha256((ROOT / "figures" / f"{name}.svg").read_bytes()).hexdigest() for name in figures_},
        "data_scope": "actual BF16 confirmation; family-clustered; fixed calibration", "synthetic_data_used": False}
    path = ROOT / "figures/round2_ef_metadata.json"
    if args.check:
        if json.loads(path.read_text()) != metadata:
            raise SystemExit("R2 figure metadata is stale")
    else:
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "report": str(REPORT), "figures": list(figures_)}))


if __name__ == "__main__":
    main()
