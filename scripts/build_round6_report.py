#!/usr/bin/env python3
"""Generate the R5 peer-review/R6 recommendation report from auditable sources."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"reports/ROUND6_RECOMMENDATIONS_KO.md"


def load(name):
    return json.loads((ROOT/name).read_text())


def validate():
    review=load("logs/summary/round6_peer_review.json")
    diagnostic=load(review["posthoc_diagnostic"])
    cfg=yaml.safe_load((ROOT/review["plan"]).read_text())
    assert review["status"]=="REVIEW_COMPLETE" and review["new_GPU_forwards"]==diagnostic["new_GPU_forwards"]==0
    assert cfg["status"]=="DRAFT" and cfg["registration_status"]=="NOT_REGISTERED"
    assert cfg["execution_status"]=="NOT_EVALUATED" and cfg["actual_new_model_forwards"]==0
    for name,expected in review["protected_measured_sources_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for name,expected in review["private_evidence_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    assert diagnostic["source_result_sha256"]==hashlib.sha256((ROOT/"logs/summary/round5_discovery.json").read_bytes()).hexdigest()
    assert diagnostic["analysis_code_sha256"]==hashlib.sha256((ROOT/"scripts/round6_diagnostics.py").read_bytes()).hexdigest()
    b=cfg["pilot_budget"]
    total=sum(math.prod(c["factors"].values()) for c in b["components"])
    assert total==b["expected_max_planned_calls"] and total+b["reserve_calls"]==b["max_actual_forward_calls"]
    assert [r["id"] for r in review["recommendations"]]==[r["id"] for r in cfg["recommendations"]]
    assert all(r["novelty"].startswith("UNVERIFIED") for r in review["recommendations"])
    assert cfg["confirmation"]["arm_manifest"]=="NOT_PREPARED" and cfg["confirmation"]["full_forward_budget"]=="NOT_ENUMERATED"
    return review,diagnostic,cfg,total


def render():
    s,d,c,total=validate()
    actual=load("logs/summary/round5_discovery.json")
    cell={(r["arm"],r["witness"]):r for r in d["cells"]}
    dual=d["healthy_margin_nats_by_witness"]["dual_route"]
    out=["# R5 실측 재검토와 R6 후속 실험 추천", "",
        "**추천 1순위: 답의 내용·문자와 분리된 출처 선택을 전달하는 작은 인과적 병목.** "
        "GPT Pro와 Claude Opus 5.5 Extra의 완료 답변, 추가 margin 진단에 대한 수정, 최종 교차 비판을 반영했다. "
        "모든 후속은 **DRAFT / NOT_REGISTERED / NOT_EVALUATED**, 신규성은 **UNVERIFIED**다. 이번 작업의 새 GPU forward는 **0회**다.","",
        "## R5에서 측정한 사실과 새 사후 진단", "",
        f"실제 R5는 **{actual['actual_forward_calls']:,} forward** 후 `{actual['analysis']['gate_outcome']}`로 종료했다. "
        "16개 차단의 양쪽 표 정답률은 모두 24/24로 유지됐으며 적격 차단 쌍은 0개다. "
        "피드백 회복과 모집은 측정하지 않았다. 원 실측·등록·gate는 변경하지 않았다. "
        "[원 보고서](ROUND5_RESULTS_KO.md), [원 요약](../logs/summary/round5_discovery.json).", "",
        "재검토 중 로컬에 저장된 같은 discovery logits를 CPU로 재계산했다. 아래는 **사후 기술 통계**이며, "
        "새 독립 확증이나 R5 gate의 대체가 아니다. 관측 24개는 8 episode의 파생이고, 16후보 중 최대를 택한 예시는 선택 편향이 있다. "
        "p값이나 선택 후보의 확증 CI를 만들지 않았다.","",
        "| 원 R5 후보·조건 | 평균 signed margin 손상 | 평균 절대 margin 변화 | 정답률 손상 |", "|---|---:|---:|---:|"]
    for arm,witness,label in [("L15_K128_composition","dual_route","L15·128·compose 대비 / 양쪽 표"),
                             ("L26_K128_composition","direct_only","L26·128·compose 대비 / 직접 표"),
                             ("L26_K128_composition","code_map_only","L26·128·compose 대비 / 코드 변환만")]:
        v=cell[arm,witness]
        out.append(f"| {label} | {v['mean_margin_damage_nats']:.4f} nats | {v['mean_absolute_margin_change_nats']:.4f} nats | {100*v['accuracy_damage']:.1f}pp |")
    out += ["", "손상 부호는 healthy−차단이다. 음수는 margin 개선이다. "
        f"Healthy 양쪽 표의 정답 margin은 최소 **{dual['minimum']:.2f}**, 중앙 **{dual['median']:.2f}**, 최대 **{dual['maximum']:.2f} nats**였다. "
        "정확도는 보존돼도 margin이 바뀌었고, 코드 변환 대조에서도 연속 효과가 있었다. "
        "이것만으로 내부 효과를 0이라고 하거나 모든 변화를 전역 confidence로 단정할 수 없다. "
        "특정 의미 기능·병목의 증거도 아니다. [64조건 전체 진단과 lineage](../logs/summary/round6_margin_diagnostics.json).", "",
        "## 실제 두 채팅의 피드백", "",
        "| 검토자 | 이번 실제 수행 | 추천에 반영한 핵심 |", "|---|---|---|",
        "| GPT Pro | 웹 검색을 포함한 완료 검토 10분 12초 + 최종 답변 2분 55초 | 수신 문제의 다른 답과 donor 답을 구분하는 선택 전이, 독립 donor 복원, 제약된 균형 무작위화 |",
        "| Claude Opus 5.5 Extra | 완료 검토 + margin/row-patch 해석 수정 + GPT 제안에 대한 완료 교차 비판 | 표 행 상태의 출처 혼합, 정확도-only guard의 한계, cue 문자 전달이라는 대안과 cue 제외·별도 어휘 검증 |", "",
        "두 채팅에 동일한 원 실측 요약을 전송했고, 새 CPU 진단도 추가로 전달했다. 원 답변과 모든 수정본을 실제로 읽었다. "
        "이번에는 새 독립 Deep Research 문서를 생성했다고 주장하지 않는다. 두 AI는 원시 Gemma 추론을 독립 재현하지 않았다. "
        "GPT는 최종 검토에서 CoAx v3와 Rewired or Gated 본문 접근 실패를 명시했다. Codex가 해당 원전의 아래 기재 범위를 별도로 확인했다.","",
        "처음 순위는 달랐다. GPT는 선택 병목→중간값→백업, Claude는 지속 차단 뒤 출처 전환→출처 지도→선택 변수 교환이었다. "
        "Codex가 답 복사·출처 혼합·cue 전달과 비용 누락을 되물은 뒤, 두 검토자는 최종 ①선택 병목→②조건부 백업의 정답성/출처 충실성→③중간값 재사용 순서에 동의했다. "
        "이 동의는 신규성 보증이 아니다. [검토·수정 근거와 원문 해시](../logs/summary/round6_peer_review.json).", "",
        "## 추천순", "", "| 순위 | 알아내고 싶은 명제 | 우선순위 이유 | 상태 |", "|---|---|---|---|"]
    for r in s["recommendations"]:
        out.append(f"| {r['rank']} | {r['proposition_ko']} | {r['reason_ko']} | {r['novelty']} |")
    out += ["", "## 1위 — 답을 복사하지 않고 출처 선택을 옮기는 작은 병목", "",
        "> 작은 고정 native 집합의 질의 시점 상태를 바꾸면, donor의 답을 복사하지 않고 받는 문제의 다른 표 계산 결과를 선택하게 되는가?", "",
        "같은 답을 주던 R5와 달리, h와 g∘f가 서로 다른 답을 주는 새 과제를 primary로 둔다. "
        "올바른 출처는 지시 cue로 명시하고, 동의·단일 경로·코드 변환 과제는 따로 평가한다. 아래 a/b/c/d는 설명용 기호이며 실측 예시가 아니다.","",
        "| 입력 | 직접 표 답 | 합성 표 답 | 선택 cue |", "|---|---|---|---|",
        "| 받는 문제 | a | b | 직접 표 |", "| 다른 세계의 donor 1 | c | d | 직접 표 |",
        "| 같은 donor 세계의 donor 2 | c | d | 합성 표 |", "",
        "donor 1/2의 사실과 코드표는 같고 cue만 다르다. 작은 집합의 상태를 donor 2에서 옮겼을 때, "
        "받는 문제의 **b**로 이동해야 한다. donor가 선택한 d, 선택하지 않은 c, 또는 donor의 출력 문자를 복사하면 실패다. "
        "네 의미값과 두 입력의 관련 코드 문자를 구별하고 허용 셀 안에서 균형화한다. 충돌 회피 제약이 있으므로 완전 요인설계라고 부르지 않는다.","",
        "1차 대비는 받는 문제의 대체 답 b에 대한 post-softcap margin 증가의 "
        "**반대-selector donor−같은-selector donor** 차이에서, 고정 무작위 3집합의 같은 차이를 뺀 것이다. "
        "두 교환 방향과 여러 donor를 episode 안에서 평균한다. 받는 입력의 코드표로 의미값을 읽는다. "
        "정답 코드 선택과 전체 어휘 top1도 기록한다.","",
        "병목이라는 해석에는 세 종류의 증거를 모두 요구한다.","",
        "1. **손상:** 같은 G를 고정한 query/template 구간에서 차단하면 그 문맥의 출처 선택 행동이 선택적으로 손상된다.",
        "2. **전이:** 반대 선택의 다른 세계 donor는 donor 내용이 아닌 받는 문제의 대체 계산 결과로 옮긴다.",
        "3. **독립 복원:** 다른 입력의 같은 선택 donor가 손상된 선택을 복원하며, 반대 선택·무관 내용 donor보다 낫다.","",
        "복원 arm은 query G를 독립 donor 값으로 다시 여는 별도 개입이다. G가 계속 0인 상태라고 하지 않는다. "
        "같은 forward에서 원래 값을 그대로 되돌리는 항등 검사는 기술 동등성만 확인한다. "
        "이 R6-A는 R5의 지속 zero 조건을 유지한 채 재개하는 실험이 아니다.","",
        "통제에는 같은 선택·다른 내용, 다른 답 donor, 한 표의 사실만 바꾸는 반사실, 코드표 교환, "
        "donor의 선택/미선택 답 복사 검사, 3 seed 무작위군, code-only의 signed margin·분포·entropy를 포함한다. "
        "원래 cue 위치를 제외한 query patch에서도 효과가 필요하고, cue-only patch·전체 residual patch를 비교하며 "
        "새 cue 표현과 표 이름 재배정에서 검증한다. 후반 표 행의 hidden에는 다른 출처 정보가 섞일 수 있다. "
        "전체 residual 교환은 수학적인 상한이나 복원률 분모로 가정하지 않는다.","",
        "가장 가까운 선행은 [Causal Abstraction](https://www.jmlr.org/papers/v26/23-0058.html), "
        "[Function Vectors](https://arxiv.org/abs/2310.15213), "
        "[Competition of Mechanisms](https://arxiv.org/abs/2402.11655), "
        "[Rewired or Gated](https://aclanthology.org/2026.blackboxnlp-1.28/)다. 정보/함수 전이나 source gating 자체는 새롭지 않다. "
        "남을 수 있는 기여는 답·코드·cue 어휘와 구분되는 **같은 작은 집합의 문맥 내 필요성, 다른 세계로의 선택 전이, 독립 복원**을 함께 보이는 범위다. 최초성은 미확인이다.","",
        "**폐기/중단:** 큰 residual 교환에서만 성공하거나, cue 위치·특정 cue 어휘에만 의존하거나, donor의 어느 답/문자를 복사하거나, "
        "같은/반대 selector donor가 똑같이 복원하거나, 무작위·코드 변환의 비선택적 변화와 구별되지 않으면 작은 선택 병목 해석을 버린다. "
        "제한된 탐색 실패는 모든 선택 회로의 부재를 뜻하지 않는다.","",
        "## 2위 — 백업은 정답을 교정하는가, 대체 출처를 그대로 따르는가", "",
        "> 원래 경로가 손상된 뒤의 백업은 정답을 계산해서 회복하는가, 남은 표가 틀리면 그 특정 오답까지 전달하는가?", "",
        "적격 primary L과 backup R을 별도 discovery에서 확보한 뒤에만 여는 조건부 실험이다. "
        "정답 oracle은 권위 있는 primary 표에 고정하고, 대체 표가 정답과 일치하는 조건과 충돌하는 조건을 만든다. "
        "L 유무×R 차단 유무×대체 표의 일치/충돌을 교차한다. 여기서는 L을 모든 토큰에서 계속 차단하며 R은 구분된 질의 구간에서 다룬다.","",
        "D_R를 기준 정답 margin의 R 차단 손상이라 두고, "
        "**[D_R(L,일치)−D_R(healthy,일치)]−[D_R(L,충돌)−D_R(healthy,충돌)]**에서 "
        "고정 무작위군의 같은 대비를 뺀다. 추가로 대체 표의 값을 자연 donor 값으로 바꿨을 때 "
        "그 **특정 오답의 정체성**까지 따라가야 한다. 일반적인 정확도 하락만으로 이 명제를 지지하지 않는다.","",
        "[CoAx](https://arxiv.org/html/2607.01940v3)와 [Hydra](https://arxiv.org/abs/2307.15771)는 이미 차단 뒤의 보상과 "
        "조건부 중요성을 다룬다. 새 질문 후보는 **보상의 정답성과 출처 충실성의 분리**다. "
        "적격 L/R이 없거나 특정 대체값 전달 없이 confidence 변화만 남으면 중단한다. "
        "준비 조건이 많아 2위이며 R5 차단 집합을 적격으로 간주하지 않는다.","",
        "## 3위 — 작은 병목의 중간값이 새 후속 함수에서도 재사용되는가", "",
        "> G에 옮긴 값 y′가 donor의 최종 답을 밀어주는 대신, 받는 문제의 새로운 g에 입력되어 g(y′)로 계산되는가?", "",
        "직접 표를 제거한 f→g 과제에서 donor 중간값×받는 문제의 새 g×코드표를 교차한다. "
        "받는 g(y′)와 donor 최종 답은 다르게 구성한다. 1차 대비는 이 두 답의 logit 차이 변화에서 sham·무작위군 효과를 뺀 것이다. "
        "같은 y′·다른 g의 donor, 양방향 교환, 새 함수, 코드 변환 보존과 독립 donor 복원이 필요하다.","",
        "이는 [When Does Activation Steering Change What a Model Computes From?](https://arxiv.org/abs/2606.29522)의 "
        "edited register·후속 연산 검사와 직접 겹친다. 단지 다른 모델에서 중간값을 이식한 결과는 기여가 약하다. "
        "고정한 작은 집합이 미관측 함수들에서 재사용되는 범위를 독립 검증해야 남을 여지가 있다. "
        "donor의 최종 답을 따르거나 새 g에서 효과가 없어지면 중간변수 해석을 폐기한다.","",
        "## 공통 첫 단계: 병목 후보를 찾는 제한된 pilot", "",
        "6개 층 구간×attention/MLP×사실/cue/query 위치를 제한적으로 탐색하고 실제 개입 대비가 남는 위치를 찾는다. "
        "다음에 한 층의 native 집합 또는 명확히 이름 붙인 attention 성분으로 압축한다. "
        "MLP 뉴런은 down_proj 입력 좌표로 한정한다. attention만 중요하면 attention 결과로 보고하며 MLP 뉴런이라고 하지 않는다. "
        "[ACDC](https://arxiv.org/abs/2304.14997), [EAP-IG](https://arxiv.org/abs/2403.17806) 같은 도구 사용은 신규성 순위에 넣지 않는다.","",
        f"초안은 **{total:,}개 호출 슬롯 + reserve {c['pilot_budget']['reserve_calls']}회 = cap {c['pilot_budget']['max_actual_forward_calls']:,}회**다. "
        "다음 표는 구현 완료된 arm manifest가 아닌 **탐색 자원 배분안**이다. 모든 실제 donor/recipient forward를 센다. "
        "새 donor나 재계산은 reserve를 쓰거나 중단하며, 무료 prefix 호출을 숨기지 않는다.","",
        "| Pilot 부분 | 슬롯 |", "|---|---:|"]
    names={"technical":"기술 검증", "task_readiness":"과제 준비", "code_only_readiness":"코드 변환 준비",
        "localization_sources":"Coarse 탐색의 donor/기준 입력", "coarse_localization":"제한된 site·위치 탐색",
        "compression_sources":"압축 단계의 source 재계산", "finer_compression":"최대 32개 finer mask",
        "screen_sources":"고정 후보 screen의 source", "code_guard_sources":"Code guard의 source",
        "candidate_and_panel_screen":"후보 1개+무작위 3개 screen", "candidate_code_guard":"연속 code guard",
        "removal_and_technical_restoration":"제거·기술 원상복귀"}
    for part in c["pilot_budget"]["components"]:
        out.append(f"| {names[part['id']]} | {math.prod(part['factors'].values()):,} |")
    out += ["", "주 과학 산술은 같은 저장 BF16 값의 전체 FP32 upcast, 실제 BF16/FP32 훅 검증, 한 GPU, use_cache=False, "
        f"최대 {c['max_input_tokens']}토큰·run disk {c['max_run_disk_gb']}GB·{c['max_seconds']//3600}시간을 제안한다. "
        "첫 50회 완료 뒤 입력 길이와 실제 시간을 측정한다. R5의 평균 forward 시간을 새 설계의 보장된 처리율로 쓰지 않는다. "
        "CPU activation은 한 세계씩 streaming하고 실제 저장 예산을 넘으면 멈춘다.","",
        f"새 discovery **{c['discovery']['episodes']}**→후보·scope·대비 동결→독립 calibration **{c['calibration']['episodes']}**→"
        f"sealed confirmation **{c['confirmation']['episodes_proposed']} episode 제안**이다. "
        "기존 R5의 열지 않은 calibration/confirmation도 새 확증에 재사용하지 않는다. Confirmation 64는 검증된 power 계산 결과가 아니다. "
        "전체 확증의 arm manifest·forward 산식은 **NOT_PREPARED / NOT_ENUMERATED**다. 이를 완성하고 실행 코드·새 입력·예산·분석을 등록하기 전에는 모델 실행 준비 완료라고 하지 않는다.","",
        f"실용 목표 초안은 margin **{c['estimands']['practical_margin_target_proposed_nats']:.2f} nat**, 행동 대비 "
        f"**{100*c['estimands']['practical_behavior_target_proposed_points']:.0f}pp**다. 기술 replay 바닥과 실용 중요도를 분리하고 "
        "결정 규칙은 calibration 전에, 최종 계획은 confirmation 전에 고정한다. 후보를 calibration에서 바꾸지 않는다. "
        "Episode 안에서 donor·질의·codebook을 평균한 paired CI를 쓰고 여러 주가설이면 Holm을 적용한다. "
        "실용적 양의 효과보다 상한이 작으면 그 방향 가설 미지지, 넓으면 미결정, 적격 조건 부재는 식별불가다. "
        "동등성은 별도로 등록한 동등성 판정이 필요하다.","",
        "## 해석과 준비 상태", "",
        "출력 제어만으로 자연 메커니즘을 찾았다고 할 수 없다는 "
        "[subspace patching의 해석 문제](https://arxiv.org/abs/2311.17030)와 "
        "[patching 지표·corruption 선택의 영향](https://arxiv.org/abs/2309.16042)을 반영했다. "
        "관련 RMSNorm replay는 baseline parity를 통과한 별도 modified-graph 민감도 분석이며 모든 정규화·비선형 대안을 제거하지 않는다. "
        "유효한 텍스트 과제를 만들었다고 훈련 분포의 on-support임을 입증한 것도 아니다.","",
        "**검토와 CPU 진단은 완료, 다음 모델 실험은 미실행**이다. "
        f"기존 실측·등록 파일 **{s['protected_prior_files_verified']}개**를 해시로 보존했고, 원 대화·스크린샷·전송문은 로컬 runs에만 보관했다. "
        "14개 원전 항목은 공식 초록 또는 명시된 절 범위까지 확인한 목록이며 포괄적 문헌조사나 신규성 증명이 아니다. "
        "접근하지 못한 DOI는 근거로 사용하지 않았다. 원 R5 판정은 계속 중단 상태다.","",
        "[공동 검토 기록·접근 범위](../logs/summary/round6_peer_review.json) · "
        "[계획·예산 초안](../configs/round6_plan.yaml) · [새 CPU 진단](../logs/summary/round6_margin_diagnostics.json)",""]
    return "\n".join(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--check",action="store_true");a=p.parse_args()
    expected=render()
    if a.check:
        assert DEST.read_text()==expected,"R6 report stale"
    else:
        DEST.write_text(expected)
    print("PASS: R6 report agrees with completed peer review, CPU diagnostics and draft plan; new GPU forwards=0.")


if __name__=="__main__":
    main()
