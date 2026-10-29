#!/usr/bin/env python3
"""Render the reviewed R5 protocol; never turn planning checks into model data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/ROUND5_DESIGN_KO.md"


def render() -> str:
    s = json.loads((ROOT / "logs/summary/round5_design_peer_review.json").read_text())
    v = json.loads((ROOT / s["cpu_verification"]).read_text())
    c = yaml.safe_load((ROOT / s["plan"]).read_text())
    for p, expected in {**s["protected_measured_sources_sha256"], **v["source_sha256"]}.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed: {p}")
    budget = v["budget"]
    out = ["# R5 공동 설계 — 지속 차단 아래 오류 피드백에 따른 질의 시점의 인과적 의존성", "",
        "**협업 설계와 CPU 검증 완료 / 실제 12B R5 실험 NOT_EVALUATED / 전체 확증 사전등록 전.** "
        "사용자가 선택한 2번 주제다. R3-A의 dose·후보·gate를 바꾸어 재개하는 연구가 아니다. "
        "아래 임계값·예산은 미래 프로토콜의 제안이며 관측 결과가 아니다.", "",
        "## 우리가 알아내려는 명제", "",
        "> 고정된 native MLP 좌표 집합을 모든 토큰에서 계속 차단해도, 동일한 문제·참조답에 붙은 차단 유형별 오류 status 이력이 "
        "새 질의의 정답률을 회복시키고, 그 회복과 함께 다른 사전 고정 좌표 집합에 대한 질의 시점의 인과적 의존성이 달라지는가?", "",
        "여기서 모집(recruitment)은 **같은 1차 차단에서 history를 바꿀 때 2차 차단의 효과가 달라지는 현상**으로 정의한다. "
        "새 가중치·새 물리적 연결·특권적 자기성찰·형식적 매개·새 알고리즘 생성을 뜻하지 않는다. "
        "활성 증가나 정답률 회복만으로 이 명제를 지지했다고 하지 않는다.", "",
        "## 실제 GPT·Claude 협업", "",
        "| 검토자 | 실제 수행 | 주요 기여 |", "|---|---|---|",
        "| GPT Pro Deep Research | 새 Research 보고서 완료 + 짧은 최종 기술 검토 완료 | nonce 표 과제, 동일 참조답·status-only 이력, history crossover와 episode 분석 |",
        "| Claude Opus 5.5 Extra Research | 새 Research 문서 완료 + 교차 비판 후 최종 수정 완료 | CoAx 중복, 코드표 조회 대조, status 순열의 층화 함정, query-only 2차 차단과 비용 |", "",
        "두 원문 보고서와 후속 답변을 실제로 읽었다. Claude의 공식 저장소 실측 보고서 본문 접근 실패를 기록했으며 그 부분은 전달 수치에 의존했다. "
        "Research UI의 검색·source 개수는 독립적인 원전 검증 개수가 아니다. 두 AI는 자문을 수행했으며 Gemma 실험을 독립 재현하지 않았다. "
        "[구조화 검토 기록](../logs/summary/round5_design_peer_review.json)에 접근 범위와 비공개 원문 해시를 남겼다.", "",
        "## 선행과 기여의 경계", "",
        "[CoAx](https://arxiv.org/html/2607.01940v3)는 이미 primary 제거 뒤 ablation 효과가 커지는 backup을 찾고, 정규화와 손상 크기 대조를 한다. "
        "따라서 이중 차단이나 backup 발견 자체는 새 기여가 아니다. R5의 추가 질문은 **물리적 차단을 고정해도 오류 이력만으로 "
        "질의 계산의 의존성이 선택적으로 달라지는가**다. 해당 조합의 신규성은 UNVERIFIED이며 최초성을 주장하지 않는다.", "",
        "[Hydra](https://arxiv.org/html/2307.15771v1)와 [Self-Repair](https://arxiv.org/html/2402.15390v1)는 즉각적 보상과 정규화·Anti-Erasure를 다룬다. "
        "[Lesioned Transformer 회복](https://aclanthology.org/2026.blackboxnlp-1.25/)은 readout·mask·encoder 재학습을 사용한다. "
        "R5는 파라미터와 1차 mask를 고정하고, 외부 피드백 문맥에 따른 변화만 검정한다. CoAx 원전의 abstract/HTML에 나타난 수치 차이 때문에 "
        "그 수치 상수를 설계에 전용하지 않았으며, BERT 논문은 공식 초록까지 확인했다.", "",
        "## 과제와 이력", "",
        f"에피소드마다 새 X {c['task']['X_keys']}개→Y {c['task']['Y_keys']}개→Z {c['task']['Z_values']}값의 표 f,g와, "
        "같은 답을 주는 직접 표 h를 제공한다. 예를 들어 f(x)=y, g(y)=z이고 h(x)=z다. 모든 사실은 문맥에 주며 전문 지식은 필요 없다. "
        "직접·두 단계·양쪽 표 witness와 **Z→응답 코드만 찾는 대조**를 별도로 둔다. witness와 충돌 probe는 특정 내부 알고리즘의 증명이 아니다.", "",
        f"이력용 {c['task']['training_queries']}개 X와 평가용 {c['task']['held_out_queries']}개 X는 겹치지 않는다. "
        "질의마다 A/B/C/D 코드표를 바꾸고, 각 집합의 의미 정답과 정답 코드 빈도를 모두 균형화한다. 이진 라벨 반전이나 고정 출력 기호의 편향을 줄이는 설계다.", "",
        "진단 시행은 실제 차단 모델의 답을 로컬에 기록한다. 보이는 record에는 **문제·평가자가 제공한 올바른 참조답·status**만 넣는다. "
        "모델이 참조답을 생성한 것처럼 기록하지 않는다. 다음 진단 시행은 이전까지의 이 records를 보며, 평가 질의에서는 피드백을 주지 않는다.", "",
        "| 이력 | 참조답 정보 | status |", "|---|---|---|",
        "| H_A / H_B | 동일 문제·순서·올바른 참조답 | 각 source 차단의 실제 진단 정오 0/1 |",
        "| H_demo | 동일 | 2 = 미보고 |",
        "| H_Aperm / H_Bperm | 동일 | 각 source의 status만 순열 |", "",
        "순열은 앞 4회·뒤 4회 안에서만 한다. 오류를 예측하는 route/문항 특징으로 층화하면 핵심 프로파일이 보존될 수 있어 사용하지 않는다. "
        "순열이 원래 벡터와 같아도 제외하지 않고 비율을 기록한다. status 0/1/2는 실제 토크나이저에서 길이가 같다. "
        "이 설계는 제공된 오류 프로파일에 조건화한 계산을 평가하며, 텍스트 밖의 비공개 기억이나 자기인식을 검정하지 않는다.", "",
        "## 차단 선정과 실행 계약", "",
        "Discovery에서만 두 primary 집합 L_A/L_B와 남은 R_A/R_B, 무작위 패널의 identity 및 오류 프로파일 특징·방향을 정한다. "
        "Calibration에서 새 좌표를 찾지 않는다. 층 후보는 15/22/26/35, 크기는 32/128의 제한된 grid다. R은 두 L과 겹치지 않는다. "
        "이전 네 ambiguous 좌표를 R5의 기능적 차단으로 가정하지 않는다.", "",
        "Calibration의 제안 자격은 healthy 4지선다 정확도≥.90, dual 과제 손상 점추정≥.20 및 paired95%CI 하한>.10, "
        "반대 route witness 손상 차≥.15, 코드표-only 손상≤.05, 두 L의 총 손상 차≤.10이다. 코드표 guard는 여기서 통계적 동등성 주장이 아니다. "
        "발견 시 고정한 가시적 표 위치 특징이 새 calibration 진단 오류를 예측하는 AUC≥.70도 요구한다. "
        "이 조건을 만족하는 두 집합이 존재할지는 아직 모른다. 실패하면 같은 연구의 차단을 키워 통과시키지 않고 NOT_IDENTIFIABLE로 중단한다.", "",
        "모든 과학적 단계는 같은 저장값의 FP32-upcast 산술을 사용한다. BF16는 기술 동등성 검사용이며 FP32를 실수 참값으로 부르지 않는다. "
        "L은 모든 토큰에서 zero, R과 대조는 고정한 새 query 구간과 template 경계에서만 적용한다. "
        "**주 실행은 use_cache=False로 전체 문맥을 매번 다시 계산**한다. 한 constrained 답은 실제 forward 1회이며 별도의 prefix 호출이나 decoder 생성은 없다. "
        "전체 vocab top1은 별도로 기록하고, 이를 4지선다 정확도와 섞지 않는다. 향후 cache 최적화는 별도 parity 및 조건 hash 검증이 필요하다.", "",
        "## 미리 고정할 1차 대비", "",
        "에피소드별로 held-out 질의들을 먼저 평균한다. 단위는 토큰이나 조건 행이 아닌 독립 에피소드다.", "",
        "- 회복 R: mean_L[Acc(L,H_L)−Acc(L,H_demo)]",
        "- 진실 이력 F: mean_L[Acc(L,H_L)−Acc(L,H_Lperm)]",
        "- 교차 T: ½[(Acc(A,H_A)−Acc(A,H_B))+(Acc(B,H_B)−Acc(B,H_A))]", "",
        "반대 이력이 해치기만 해서 생긴 T 양성을 회복으로 세지 않는다. R→F→T의 고정 순서에서 정확도 효과 점추정≥.08과 양의 CI를 요구한다. "
        "이 행동 gate를 통과한 뒤에만 기제를 확증적으로 연다. 질의/에피소드를 개입 후 정오답이나 margin으로 골라내지 않는다.", "",
        "Y는 tanh softcap 뒤 정답 logit−logsumexp(다른 세 응답 logit)이다. D(L,H,G)=Y(L,H,none)−Y(L,H,G)로 정의한다. "
        "주 기제 M은 own−demo에서 R_L의 D 증가를 구하고, 고정 무작위 3개 군의 같은 증가 평균을 뺀 뒤 L_A/L_B에서 평균한다. "
        "own−cross 이력 대비와 R_A/R_B 교차 해리도 사전 지지 조건으로 요구한다. 정답률이 그대로면 margin 변화만으로 행동 회복이라 하지 않는다.", "",
        "무작위 군은 층·크기·활성 및 calibration에서의 실제 postnorm 잔차 dose를 맞추고 전역 attenuation을 동결한다. "
        "새 확인 입력의 출력 Fisher 변위에 맞춰 재선택하지 않는다. 확인 중 matching 실패 비율>.20이면 기하를 통제한 기제 해석은 식별불가로 두고 "
        "전체 행동 표본은 보존한다. 같은 좌표 수가 같은 에너지라는 뜻은 아니다.", "",
        f"확인은 {c['estimands']['confirmation_episodes']}개 에피소드이며 paired episode bootstrap {c['estimands']['CI'].split(',')[1].strip()}를 사용한다. "
        "기제의 실용 기준은 calibration에서 max(.05 nats, 5×기술 replay floor)로 동결한다. "
        "48은 8%p 효과에 대한 80% 검정력을 보장하지 않는다. 예를 들어 episode SD=.25라는 가정이면 정규 근사의 80% MDE는 약 .10이다. "
        "이는 모델 실측 SD나 power가 아닌 계획 민감도다. 추가 개별 대조를 독립 발견으로 보고할 때에는 다중비교를 따로 보정한다.", "",
        "## 정규화 민감도 분석과 주장 범위", "",
        "사전 지정 첫 8개 확인 episode에서 각 (L,H)의 L-only reference로 RMSNorm 분모를 기록한다. "
        "가장 이른 R 층 이후의 실제 inventoried RMSNorm 및 terminal norm의 query 위치에 그 분모를 재생하고 baseline replay 자체를 먼저 검사한다. "
        "Q/K 등 tensor 모양과 순서까지 로컬 구현에서 명시해야 하며 해당 GPU 구현은 아직 없다. 일부 norm만 구현되면 그 범위로 해석을 줄인다. "
        "이 분석은 변형된 연산 그래프의 부분집합 진단이다. 모든 normalization·attention 재가중·비선형 대안을 제거하거나 "
        "전체 48개 episode에 대해 강건성을 확인했다는 뜻이 아니다. formal mediation이나 새 알고리즘 주장은 하지 않는다.", "",
        "## 셀·자원 예산", "",
        "모든 physical 상태 healthy/A/B × 5 history =15개에, **세 physical 상태의 no-history를 각각 더해 18개**다. "
        "GPT 후속의 16개 집계는 이 세 baseline을 모두 세지 않아 수정했다. 기제는 A/B×own/demo/cross의 6문맥, 각 RA/RB/무작위3군이다.", "",
        "| 단계 | 실제 full-prefix forward 계획 |", "|---|---:|",
    ]
    for row in budget["stages"]:
        out.append(f"| {row['stage']} | {row['planned_actual_forward_calls']:,} |")
    out += [f"| 합계 | **{budget['planned_actual_forward_calls']:,}** |", "",
        f"**상한 {c['budget']['max_actual_forward_calls']:,}회·{c['budget']['max_seconds']//3600}시간 중 먼저 도달하면 중단**, "
        f"국소 모듈 계산 {c['budget']['max_local_module_evaluations']:,}회·run 디스크 {c['budget']['max_run_disk_gb']}GB도 제한한다. "
        "첫 50-call의 실제 길이별 시간을 보고 예산 적합성을 확인한다. 시간 수치는 아직 측정하지 않았으며, "
        "norm replay·code-map guard·source 진단을 모두 포함한 call 산식은 설정과 CPU checker에 있다.", "",
        "## 진행·중단 판정", "",
        "| 결과 | 허용하는 결론 |", "|---|---|",
        "| 기술/과제/두 차단의 자격 실패 | 해당 설계는 NOT_IDENTIFIABLE; 적응 능력 전체의 부재 아님 |",
        "| Calibration matched−demo 또는 crossover 점추정≤0 | 확증을 열지 않고 예비 단계에서 중단 |",
        "| 행동만 통과 | 이 조건의 행동 적응; 모집·알고리즘 재배치 미입증 |",
        "| 행동·기제·사전 지지 조건 통과 | 이 표본과 차단에서 history-dependent query-time causal recruitment 지지 |",
        "| 목표 양의 실용 효과보다 upper CI가 작음 | 해당 방향·범위의 가설 미지지/기각 |",
        "| 90%CI가 사전 동등 범위 안 | 그 대비의 실용적 동등성만 지지 |",
        "| CI가 넓음 | INCONCLUSIVE |", "",
        "등록되지 않은 추가 후보·과제·강도로 결과를 바꾸려면 새 연구로 분리한다. 작은 bounded search의 실패는 모든 회로에 대한 부재 증명이 아니다.", "",
        "## 이번에 검증한 것과 아직 남은 것", "",
        f"CPU 합성 오라클 {v['synthetic_oracle_episodes_checked']} episode, 실제 로컬 토크나이저 {v['tokenized_example_episodes']} example episode를 검사했다. "
        f"최대 입력은 **{v['max_example_input_tokens']}토큰**이며 {c['execution_contract']['max_input_tokens']} cap 안이다. "
        "0/1/?의 ?는 문장부호와 합쳐져 길이가 달랐기 때문에 0/1/2로 바꿔 재검증했다. "
        "서로 다른 오라클, 질문 분리, 답·코드 균형, 순열 count 보존 및 위양성 시나리오를 검증한 7개 합성 단위 테스트가 통과했다.", "",
        "**실제 R5 GPU forward는 0회**다. 적격 L/R 선정, 오류 회복, query-time 모집 및 norm replay는 전부 NOT_EVALUATED다. "
        "CPU 예시는 확증 manifest가 아니며, GPU runner와 norm 재생도 아직 구현되지 않았다. "
        "다음 실제 실행은 이 예시를 확증 자료로 재사용하지 않고 별도 technical/discovery 실행을 등록하는 단계부터다.", "",
        "[기계 판독 계획](../configs/round5_plan.yaml) · [CPU 검증](../logs/summary/round5_design_verification.json) · "
        "[협업 근거](../logs/summary/round5_design_peer_review.json)", ""]
    return "\n".join(out)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--check",action="store_true"); a=p.parse_args()
    expected=render()
    if a.check:
        if DEST.read_text()!=expected:
            raise ValueError("R5 report is stale")
    else:
        DEST.write_text(expected)
    print("PASS: R5 report agrees with reviewed design, source hashes and CPU verification; no new model results.")


if __name__=="__main__":
    main()
