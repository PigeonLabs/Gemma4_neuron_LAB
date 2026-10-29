# 누적 논문 후보 순위

기록 기준 2026-10-30 KST. **논문 Go 0개.** Hold는 신규성·식별력이 미해결인 옵션이며 실행 선정이 아니다. No-Go는 현재 주장·설계의 투자 판단이며 현상 부재의 증명이 아니다. 과거 실측과 중단 판정을 보존한다.

| 순위 | 질문 | 판단 | 확보한 증거 |
|---:|---|---|---|
| 1 | [S1 · 같은 값의 미사용 출처와 명시 trace 과승인](../logs/summary/round7_next_candidate_screening.json) | Hold | 새 모델 측정0; 실제 두 교차심사 뒤 Go 철회. |
| 2 | [S2 · 오류의 상보성에 따른 검증자 선택](../logs/summary/round7_next_candidate_screening.json) | Hold | 새 모델 측정0. |
| 3 | [R2-E · 대조 반응으로 후보 개입 크기 예측](../reports/ROUND2_EF_RESULTS_KO.md) | Hold | 등록 ΔR² .204328; 원시 재감사상 MSE 이득 약91.92%가 string_rule에 집중. |
| 4 | [R7-C · 취소 이력의 logit 흔적](../reports/ROUND7_EXECUTION_RESULTS_KO.md) | No-Go | 등록 합성 특이 증폭 Negative. 512쌍 중 top1변경7·정답변경6, 현재 obsolete 답 선택0행. |
| 5 | [R7-E · 교란 계열 간 고정 예측기의 전이](../reports/ROUND7_EXECUTION_RESULTS_KO.md) | No-Go | 등록 ΔR² −.099841 CI[−.210115,.0149589], 80예측 target 이전 저장 확인. |
| 6 | [R1 · 단일 native 뉴런의 의미 기능](../reports/RESEARCH_SUMMARY_KO.md) | No-Go | 카탈로그 네 후보 모두 ambiguous; 무작위 대조와 독립 검증 통과 없음. |
| 7 | [R2-F · 두 뉴런 사이 손상과 복구](../reports/ROUND2_EF_RESULTS_KO.md) | No-Go | NOT_IDENTIFIABLE; 기존 판정 보존. |
| 8 | [R3-A · 초기 후보 네 개의 수치 자격](../reports/ROUND3A_RESULTS_KO.md) | No-Go | 네 후보 모두 numerical qualification 실패. |
| 9 | [R4 · 정밀도/연합/미관측 개입 예측 설계](../reports/ROUND4_RECOMMENDATIONS_KO.md) | No-Go | DRAFT/NOT_EVALUATED; 일부 문제의식은 별도 R5/R7에 반영. |
| 10 | [R5 · 피드백에 따른 경로 모집](../reports/ROUND5_RESULTS_KO.md) | No-Go | 16고정 후보·적격쌍0. |
| 11 | [R6-A/B/C · 선택 병목·백업 교정·중간값 재사용](../reports/ROUND6_EXECUTION_RESULTS_KO.md) | No-Go | 원래 A와 별도 A2, B, C 모두 등록 prerequisites에서 중단. |
| 12 | [R7-A · 진단 예시의 논리 영향 범위](../reports/ROUND7_INTERPRETATION_AUDIT_KO.md) | No-Go | Inconclusive; 48모순 R1 anchors 보존, main 미측정. |
| 13 | [R7-B · 질의 전 carrier 경로의 색인 전달](../reports/ROUND7_INTERPRETATION_AUDIT_KO.md) | No-Go | 등록 protocol Negative; full-vocab 코드 붕괴; 12draw/11고유view. |
| 14 | [R7-D · 선행 무의미 문맥의 선택적 순서 효과](../reports/ROUND7_INTERPRETATION_AUDIT_KO.md) | No-Go | Inconclusive; forward57/64,inverse32/64. |
| 15 | [S3/S4/S5 · namespace·결말 편향·목표 정보 요청](../logs/summary/round7_next_candidate_screening.json) | No-Go | 모두 NOT_EVALUATED, 신규성 gate에서 중단. |

## 1. S1 · 같은 값의 미사용 출처와 명시 trace 과승인

Hold — 값의 진실성과 trace의 허용을 분리하는 인과 대비가 남는다. 정확한 선행 차이는 미확정.

부족한 결정적 근거: LLMCAPLENS/CoRe/NeuroTaint를 넘어서는 메커니즘 예측과 신규성 확인.


## 2. S2 · 오류의 상보성에 따른 검증자 선택

Hold — CAVEAT의 외부 선택기와 모델 자신의 선택 능력은 다르나 독립 기여 미확정.

부족한 결정적 근거: 가치 함수·산술 준비도·오류 중첩의 인과 교환을 정의하고 기존 연구와 구별.


## 3. R2-E · 대조 반응으로 후보 개입 크기 예측

Hold — 고정 조건에서의 예측 관계는 남지만 posthoc dose 매칭·과제 집중과 선행 중복을 분리해야 함.

부족한 결정적 근거: 기존 확증을 반복 최적화하지 않고 독립적 새 예측 명제/이론이 필요. R7-E 음성은 R2 소급 무효화가 아님.


## 4. R7-C · 취소 이력의 logit 흔적

No-Go — 현재 좁은 관찰 이상의 독립적 학술 기여 부족.

부족한 결정적 근거: 새 예측 이론 없이 marker 교체·prompt 재탐색 금지.


## 5. R7-E · 교란 계열 간 고정 예측기의 전이

No-Go — Prospective fixed-algorithm Negative로 보존; 보편적 공통 인자 부재 명제는 불가.

부족한 결정적 근거: 과제별 사후 양성을 고르지 않고 독립적인 이론이 있어야 새 연구 가능.


## 6. R1 · 단일 native 뉴런의 의미 기능

No-Go — 검출 대비만으로 행동 의미를 부여할 수 없음.

부족한 결정적 근거: 새 분할에서 의미 선택적 인과 효과와 대체 설명 배제. 현재 탐색을 그대로 재개하지 않음.


## 7. R2-F · 두 뉴런 사이 손상과 복구

No-Go — 등록 손상 자격이 부족해 복구 의미 식별불가.

부족한 결정적 근거: 새 원인 명제 없이 dose를 높여 손상 만들기 금지.


## 8. R3-A · 초기 후보 네 개의 수치 자격

No-Go — 정밀도·readout 위생 자체를 novelty로 세지 않음.

부족한 결정적 근거: Native 개입의 동등 전달 및 단순 정밀도 재발견을 넘는 새 명제.


## 9. R4 · 정밀도/연합/미관측 개입 예측 설계

No-Go — 아이디어 검토 라운드였고 실험 완료로 세지 않음.

부족한 결정적 근거: 다른 라운드 결과를 R4 확증으로 소급 처리하지 않음.


## 10. R5 · 피드백에 따른 경로 모집

No-Go — 적격 차단이 없어 모집 명제 미측정.

부족한 결정적 근거: 새 식별 가능한 질문이 있어야 별도 연구.


## 11. R6-A/B/C · 선택 병목·백업 교정·중간값 재사용

No-Go — 준비도/선행 자격 실패이며 실제 병목이나 재사용 부재가 아님.

부족한 결정적 근거: 층/크기를 늘리기보다 구별되는 명제와 준비도가 필요.


## 12. R7-A · 진단 예시의 논리 영향 범위

No-Go — 우리의 모순 anchor 설계 오류를 모델 결함으로 제시할 수 없음.

부족한 결정적 근거: 단순 수정 재실행을 새 논문 질문으로 세지 않음.


## 13. R7-B · 질의 전 carrier 경로의 색인 전달

No-Go — 47층 궤적 교환·원문 재접근과 format 붕괴로 자연 저장 기제 주장 불가.

부족한 결정적 근거: 허용 경로 양성 format 대조는 위생 과제이며 독립 novelty 아님.


## 14. R7-D · 선행 무의미 문맥의 선택적 순서 효과

No-Go — 역조회 준비도 실패로 priming 미측정.

부족한 결정적 근거: Serialization 또는 알려진 reversal의 재발견을 독창성으로 세지 않음.


## 15. S3/S4/S5 · namespace·결말 편향·목표 정보 요청

No-Go — 현재 설계가 기존 binding/process verification/information-seeking 연구를 넘지 못함.

부족한 결정적 근거: 구별되는 메커니즘을 제시하기 전 실행 예산 배정 없음.

[실측 이후 실제 공동 심사](ROUND7_REVIEW_AND_NEXT_KO.md) · [계산 가능한 순위·근거 해시](../logs/summary/paper_candidate_ranking.json)
