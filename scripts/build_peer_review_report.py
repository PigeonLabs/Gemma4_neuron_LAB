#!/usr/bin/env python3
"""Render approved external-review synthesis; no new model results are created."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "logs/summary/round2_ef_peer_review.json"
DEST = ROOT / "reports/ROUND3_RECOMMENDATIONS_KO.md"


def render():
    s = json.loads(SUMMARY.read_text())
    for name, expected in s["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Review source changed: {name}")
    cfg = yaml.safe_load((ROOT / s["draft_plan"]).read_text())
    a, b = cfg["experiments"]
    v = s["canonical_results"]
    rows = ["# 실측 결과의 GPT·Claude 재검토와 다음 실험 추천", "",
        "**Novelty를 우선하면 1위는 내용·답변 토큰·규칙 효과의 부호 분해, 2위는 후보 관측 전 고정 probe 전이 예측이다.** "
        "두 모델에 승인된 동일 실측 요약과 핵심 설계 한계를 전달했고, 완료된 새 답변을 읽고 통합했다. 두 모델의 1순위는 달랐으며 아래 순위는 사용자의 Novelty 우선 조건을 반영한 Codex의 최종 판단이다.", "",
        "신규 GPU 추론은 하지 않았다. 아래 연구는 전부 **DRAFT / NOT_EVALUATED / 미등록**이다. 현재 R2의 수치·가설·임계값·확증 자료를 변경하지 않았다. "
        "[실측 보고서](ROUND2_EF_RESULTS_KO.md) · [재검토 요약](../logs/summary/round2_ef_peer_review.json) · [예산·설계 초안](../configs/round3_plan.yaml)", "",
        "## 실제로 받은 재검토", "",
        "| 검토자 | 새 응답 | 자체 1순위 | 자체 2순위 |", "|---|---|---|---|"]
    for reviewer in s["participants"]:
        rows.append(f"| {reviewer['name']} | 완료 · {reviewer['response_language']} | {reviewer['ranking'][0]} | {reviewer['ranking'][1]} |")
    rows += ["", "GPT에서는 새 Deep Research 보고서 「Critical review of R2-E/F and the two best next experiments」를 확인했다. Claude에서는 기존 Research 문맥의 Opus 5.5 Extra 후속 답변을 확인했다. "
        "Claude가 새 장시간 웹 조사를 수행했다고 주장하지 않는다. 원문·전송 payload·화면은 로컬 `runs/round2_ef_collaboration/`에 보존하며 공개 요약에는 해시만 싣는다. "
        "이 검토는 제공된 결과에 대한 해석·설계 자문이며 독립적인 모델 재실행이나 결과 재현은 아니다.", "",
        "두 모델 모두 E의 후보 의존 norm 매칭, 과제 구조, random 방향 실현값, BF16 해상도를 미해결 설명으로 봤다. F는 손상 gate 실패로 대체성 해석이 불가능하다는 데 동의했다.", "",
        "## 현재 결과에서 출발할 수 있는 범위", "",
        f"R2는 {v['total_condition_rows']:,}개 조건 행·{v['total_forward_calls']:,}회 실제 BF16 forward다. E의 ΔR²는 {v['E_delta_r2']:.6f}, "
        f"95% CI [{v['E_delta_r2_ci'][0]:.6f}, {v['E_delta_r2_ci'][1]:.6f}], Holm p={v['E_holm_p']:.6f}, 최종 R²={v['E_final_r2']:.6f}다. "
        "후보 개입을 먼저 실행해 실측 post-norm 크기를 대조에 전달했으므로, 후보를 보기 전 예측이나 호출 절감은 검증하지 않았다.", "",
        f"사후 진단에서 MSE 감소의 {100*v['E_string_MSE_reduction_share']:.2f}%가 문자열 과제에서 왔고 task 평균 제거 후 상관은 {v['E_task_centered_correlation_posthoc']:.6f}였다. "
        "이것은 사후 기술 통계다. ‘오차 감소의 92%가 task 간 차이로 설명됐다’는 뜻이 아니며, 과제 내 상관을 독립 확증으로 승격하지도 않는다. "
        f"E 확증 top-1 변경은 {v['E_top1_flip_rows']}회였으므로 행동 개선을 주장하지 않는다.", "",
        f"F 손상은 {v['F_damage']['mean']:.6f} nats, 95% CI [{v['F_damage']['ci_low']:.6f}, {v['F_damage']['ci_high']:.6f}]다. "
        "등록한 하한>0.10 gate를 통과하지 못해 NOT_IDENTIFIABLE이며, rescue fraction은 정의하지 않았다. 이 특정 조건의 구간 상한이 약0.031이라는 제한된 추정은 말할 수 있지만, "
        "정확한 무손상·기능 부재·두 뉴런의 동등성/비동등성·중복성·연결 부재·매개 없음·공통원인을 결론 낼 수 없다.", "",
        "## 1위: 내용·답변 토큰·규칙 효과의 부호 분해", "",
        "**질문:** 같은 native 좌표에 같은 방향의 신호를 넣었을 때 특정 결과 내용을 선호하게 되는가, 특정 답변 기호를 선호하게 되는가, 현재 지시된 규칙을 더 따르게 되는가? "
        "E는 절대 변화 크기를 예측했다. 이 실험은 효과가 어느 방향을 향하는지와 경쟁 기제를 묻는다. Claude의 1순위와 GPT의 응답 코드/계산 축 분리 제안을 합쳤다.", "",
        "**배경지식이 적은 과제:** 문자열의 reverse 대 sort, 문맥에 주어진 사전 적용 대 원문 복사처럼 답을 프로그램으로 정할 수 있는 두 규칙을 쓴다. "
        "각 원본에서 가능한 결과 u와 v를 고정한다. 규칙만 바꾸어 어느 것이 정답인지 바꾸고, 독립적으로 u/v를 A/B에 배정하는 순서도 뒤집는다. C/D로 한 번 더 반복한다. "
        "외부 사실 지식, SAE 학습, 전 뉴런 역전파는 필요하지 않다.", "",
        "### 라벨을 뒤집어서 부호를 만든다는 함정을 피하는 정의", "",
        "r=+1은 u 정답, r=−1은 v 정답이다. c=+1은 u가 첫 라벨(A 또는 C), c=−1은 u가 두 번째 라벨(B 또는 D)에 대응하는 경우다. "
        "항상 같은 **내용축 u−v**의 margin m=c·(logit_first−logit_second)를 사용한다. 정답이 바뀌었다고 metric 자체의 의미축을 바꾸지 않는다.", "",
        "±δ의 부호 있는 효과 d=(m(+δ)−m(−δ))/2를 구하고 각 코드 쌍에서 다음 성분을 분해한다.", "",
        "| 성분 | 식 | 예측하는 효과 패턴 |", "|---|---|---|",
        "| 내용 편향 C | mean(d) | 어느 규칙·라벨 배정에서도 u 쪽 |",
        "| 규칙 정합 G | mean(r·d) | u 정답일 때 u, v 정답일 때 v 쪽 |",
        "| 답변 기호 편향 L | mean(c·d) | 내용 배정과 무관하게 첫 라벨 쪽 |",
        "| 잔여 상호작용 J | mean(r·c·d) | 단순 세 모델로 설명되지 않는 교차 패턴 |", "",
        "이 표는 **미래 가설의 수학적 예측**이며 실제 새 측정값이 아니다. +δ가 규칙을 약화하면 G는 음수가 될 수 있으며, 확증을 본 뒤 좌표 부호를 뒤집지 않는다.", "",
        "**확신 증폭과의 추가 구분이 필수다.** 단순히 기존 margin을 키우는 d≈β·m_clean도 원래 답이 맞는 쉬운 문제에서는 G처럼 보인다. "
        "Calibration에서 각 코드 쌍·좌표별 C/G/L/J와 clean margin을 함께 적합해 β를 고정한다. 확증에서는 d−β_hat·m_clean의 성분을 평가한다. "
        "쉬운/어려운 입력 recipe를 discovery에서 정하고, calibration/confirmation에 각각 오답이 있는 원본 family가 최소8개 있어야 잘못된 확신을 증폭하는 것과 규칙 방향 효과를 구별하는 주장을 허용한다. "
        "표준화한 설계행렬의 full rank와 condition number≤50도 필요하다. 이 gate가 실패하면 규칙 대 확신 구분은 NOT_IDENTIFIABLE로 둔다. 확증 입력을 개별 정답률이나 효과로 거르지 않는다.", "",
        "### 분할·대조·판정", "",
        f"Discovery {a['discovery']['families']} family에서 기존 네 후보 중 최대 하나의 primary 후보와 비교할 규칙·난이도 recipe를 결정한다. "
        f"Calibration {a['calibration']['families']} family에서 SD·모든 대조 identity·전역 dose 계수·β·분석식을 고정하고, 새로운 {a['confirmation']['families']} family에서 확증한다. "
        "Template family는 분할 사이에 격리한다. AB/CD는 두 개의 고정 코드로 각 분할에서 모두 측정하므로, 두 코드의 재현을 처음 보는 토큰으로의 전이라고 부르지 않는다. "
        "각 family의 2규칙×2라벨배정×2코드, 총8변형은 한 통계 단위다.", "",
        "대조는 같은 층 native 좌표6개와 residual 방향2개다. 크기 조정은 calibration에서 정한 전역 계수만 쓰며 새 입력의 후보 효과에 맞추지 않는다. "
        "실제 전달 norm·방향·자연 활성 지원범위를 기록한다. 1차량은 primary 후보의 confidence-adjusted C/G/L에서 **고정 대조 패널 평균을 뺀 family 대비**이며 세 대비에 Holm을 적용한다. "
        "대조6개의 95백분위를 넘는 것만으로 뉴런 모집단 수준의 p<.05를 주장하지 않는다.", "",
        "효과 기준은 제안값 max(0.05 nats, 3×calibration 기술오차 99백분위)다. AB/CD 양쪽에서 방향이 재현되고 고정 대조와 확신 설명을 넘어서는 성분이 있어야 해당 제한된 기제의 증거로 본다. "
        "한 성분만 있다고 분류하려면 나머지 성분이 작다는 것도 확인해야 한다. 90% family CI가 ±SESOI 안에 드는 TOST는 이 등록 대비의 실용적 동등성만 말한다. ‘뉴런 기능이 없다’는 결론으로 넓히지 않는다. "
        "정답 선택 변화가 없으면 결과는 logit 수준으로 한정한다.", "",
        "**중단:** discovery에서 수치 오차를 넘는 signed 효과가 없으면 본 확증으로 확장하지 않는다. 확증에서 대조 대비 실용 효과·두 코드 재현이 지지되지 않으면 이 기제 가설의 추천을 내린다. "
        "규칙/확신 분리가 되지 않으면 더 강한 dose로 결과를 만들지 않고 식별 불가로 기록한다.", "",
        f"**제안 비용:** 기본 {a['planned_base_forward_calls']:,} forward, 기술 검사 포함 cap {a['max_forward_calls']:,}회, wall cap {a['max_seconds']/3600:.0f}시간. "
        "필요 개념은 logit 차, 2×2 대비, paired 대조다. 실제 소요 시간과 모델 결과는 NOT_EVALUATED다.", "",
        "**새 기여가 되려면:** 단순한 ‘부호가 바뀐다’나 ‘작업을 더 잘한다’를 넘어서 내용·기호·확신 경쟁 설명을 배제한 규칙 정합 효과가 독립 family와 두 응답 코드에서 예측대로 나타나야 한다. "
        "기능벡터·확신 조절·문맥별 steering 연구와의 정확한 중복은 아직 미확정이며, 이 실험이 양성이어도 단독 논문 기여가 자동 확정되지는 않는다.", "",
        "## 2위: 후보 관측 전 고정 probe의 새 과제·뉴런 전이", "",
        "**질문:** 후보 개입을 전혀 보기 전에, 소수의 고정 probe 반응으로 나중에 실행할 후보의 반응 크기를 예측할 수 있는가? "
        "현재 E의 가장 직접적인 후속이며 GPT의 1순위, Claude의 2순위다. Novelty 우선 순위에서는 2위로 두지만, 기존 양성을 더 강한 예측 주장으로 바꾸는 가장 직접적인 경로다.", "",
        "8개 단순 생성기에서 training용4개와 확증용4개를 미리 나눈다. Bit parity, table lookup, 문자열 치환, 리스트 필터, 순서 연산, 유한상태 이동, 명시적 Boolean 규칙, 제공 사전 등을 후보로 쓴다. "
        "최종 생성기와 난이도·길이 범위는 discovery에서 정하고 바꾸지 않는다. 새 생성기는 기존 prompt shell의 paraphrase만으로 구성하지 않는다.", "",
        f"Discovery {b['discovery']['families']} family, calibration {b['calibration']['families']} family, 처음 보는 생성기의 confirmation {b['confirmation']['families']} family를 사용한다. "
        "원본당 A/B 순서2개는 같은 family에 둔다. 기존 네 좌표의 calibration 효과로 예측식을 적합하고, 같은 네 층에서 seed로 고정한 audit 좌표4개의 개입 결과는 학습에서 완전히 제외한다. "
        "Audit 좌표의 자연 calibration SD·가중치 열 norm은 dose 정의에 사용할 수 있지만, 그 좌표의 개입 출력으로 모형을 적합하지 않는다.", "",
        "Probe는 층마다 residual 방향2개에 ±개입을 준다. 층별 크기는 독립적인 clean calibration 잔차 scale에서 한 번 정하며 후보의 post-norm 변화량을 사용하지 않는다. "
        "BF16 전달량을 맞추는 계산은 clean 상태와 고정 목표 norm만 볼 수 있다. 새 입력에서 target 후보를 실행해 대조 크기를 정하는 경로는 금지한다.", "",
        "확증은 반드시 (1) clean+probe만 실행 → (2) 모든 예측을 파일·SHA·commit으로 봉인 → (3) target 후보 개입을 실행해 채점 순서다. "
        "Primary는 새 생성기에서 보지 않은 audit 좌표4개의 반응을 family 내 평균한 값이다. 64×4를 독립 n=256으로 세지 않는다.", "",
        "기본 모형은 clean margin·top-1 margin·entropy·길이·층·알려진 nominal dose다. 새 task ID는 미리 정한 UNKNOWN으로 처리한다. "
        "확증 task의 더미 계수를 정답으로 적합하는 ‘task 고정효과’는 누수이므로 하지 않는다. 생성기 내 S를 사전 규칙으로 섞는 대조와 생성기별 paired loss 대비로 task 간 차이 설명을 검정한다.", "",
        "**성공 제안:** ΔR² 점추정≥0.05, family CI 하한>0, 확장 모형 R²>0, 생성기 내 S shuffle보다 나은 예측. "
        "재표집은 generator 안에서 family 단위로 하며 생성기별 결과도 전부 공개한다. 주장은 선택된4개 새 생성기와4개 audit 좌표에 조건부다. "
        "새 검정이 실패하면 전이 주장을 지지하지 못한 것이며, 기존 E의 원인이 오직 task-class였다고 소급 결론 내리지 않는다.", "",
        "전이가 성공해도 같은 출력의 일반적인 국소 민감도를 공유한다는 설명이 남는다. 입력별 반응 크기를 예측했다는 것만으로 새 의미 기제나 최초성을 주장하지 않는다. "
        "현재 설계의 probe16회/문항은 audit 좌표4개의 ±개입8회보다 비싸다. 따라서 이 작은 검증 자체는 호출 절감 연구가 아니다. "
        "많은 좌표로 probe 비용을 분담하는 후속이나 단순 국소 민감도 설명을 넘는 방향·규칙 구조가 있어야 추가 논문 기여를 검토할 수 있다.", "",
        f"**제안 비용:** 기본 {b['planned_base_forward_calls']:,} forward, cap {b['max_forward_calls']:,}회·{b['max_seconds']/3600:.1f}시간. "
        "지식 부담은 고정 특징 회귀와 held-out 평가다. 실제 절감된 호출 수는 후보 관측 전 예측이 성공한 뒤 별도 비용 대비 효용으로 평가해야 한다. 현재는 모두 NOT_EVALUATED다.", "",
        "## 추천 순위에서 분리한 선행 조건과 보류", "",
        "P0는 새로운8 family에서 identity·같은-forward 복원·actual nonzero dose와 BF16/FP32 signed 효과를 비교하는 수치 검증이다. "
        "같은 저장 BF16 가중치를 FP32로 올려 arithmetic precision과 가중치 값 변경을 구분한다. 기본576회, cap1,000회·30분을 제안한다. "
        "순차 로딩·단일 GPU 프로세스를 유지한다. Precision에 따라 효과 방향/순위가 불안정하거나 전달량이 반올림으로 무너지면 일반적인 기제 해석을 멈추고 precision 한계를 명시한다. "
        "이 위생 검사를 논문 Novelty 1위로 포장하지 않는다. 실행 순서는 P0→1위 discovery gate→1위 본 연구다. Gate가 실패하면 2위의 별도 연구로 전환할 수 있다.", "",
        "현재 H001/H003 rescue의 단순 반복·dose 증가는 보류한다. 독립적인 새 연구에서 의미적 손상이 먼저 확인된 경우에만 다시 설계한다. "
        "기존 R2를 실패가 사라질 때까지 재최적화하지 않는다.", "",
        "## 피드백을 반영하면서 수정한 부분", "",
        "| 피드백/제안 | 반영한 방식 |", "|---|---|",
        "| 두 모델: 후보 의존 dose가 독립 예측 주장을 막음 | 2위에서 control-only 예측 봉인을 후보보다 앞에 둠 |",
        "| Claude: 토큰축과 규칙축 비교 | 고정 내용축+독립 라벨 배정+AB/CD로 강화; 정답 라벨 교환의 항등적 부호 반전 배제 |",
        "| 단순 G를 규칙 이득으로 해석할 위험 | β·원래 margin과 오답 family를 넣어 확신 증폭 경쟁 설명 분리 |",
        "| random6의 95백분위를 넘으면 특이적이라는 제안 | 채택하지 않음. 고정 패널에 조건부인 family 대비·다중보정으로 제한 |",
        "| 새 within-task 검정 실패면 E가 task-class 효과라는 제안 | 채택하지 않음. 해당 새 전이 가설의 미지지로만 판정 |",
        "| 19.2% dose 외삽·direction2 S=.00033 표현 | 각각 개입 후 활성값의 cal min/max 이탈, 한 predictor의 R²로 바로잡음 |",
        "| TOST로 부호 기능 없음, F 손상 없음이라는 표현 | 등록 대비의 실용적 동등성과 미식별로 범위를 제한 |", "",
        "## 확인한 원전과 Novelty의 경계", "",
        "[Function Vectors](https://arxiv.org/abs/2310.15213)는 문맥을 넘어 작업 실행을 유도하는 내부 벡터를 이미 다룬다. "
        "[Confidence Regulation Neurons](https://arxiv.org/abs/2406.16254)는 entropy·token-frequency 조절을 다루며, "
        "[SVF](https://arxiv.org/abs/2602.01654)는 문맥에 따라 고정 steering 효과가 약해지거나 역전되는 문제를 다룬다. 그러므로 작업 효과·확신 조절·부호 반전 자체는 새 기여가 아니다.", "",
        "[Analyzing the Generalization and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404), "
        "[SPARK](https://arxiv.org/abs/2607.10296), [Stress-Testing Random-Vector Controls](https://aclanthology.org/2026.blackboxnlp-1.73/)는 입력별 변동, length 교란, random 실현값·구현 민감도를 선행한다. "
        "2위가 기여하려면 이런 일반 진단을 넘어 후보와 독립적인 새 생성기·좌표 전이 예측이 필요하다.", "",
        "[Every Ablation Is a Dose](https://arxiv.org/abs/2610.02173)와 [The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510)는 dose와 상호작용에 따른 patch 해석의 한계를 다룬다. "
        "이번 재검토는 해당 원전의 초록·공식 출처를 재확인한 범위이며 체계적 문헌조사 전체가 아니다. 정확한 신규성·최초성은 **UNVERIFIED**다. "
        "두 AI의 동의나 검색에서 미발견은 신규성의 증명이 아니다.", ""]
    return "\n".join(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    text = render()
    if args.check:
        if DEST.read_text() != text:
            raise SystemExit("Peer-review report is stale")
    else:
        DEST.write_text(text)
    print(json.dumps({"status": "PASS", "report": str(DEST), "new_model_inference": False}))
