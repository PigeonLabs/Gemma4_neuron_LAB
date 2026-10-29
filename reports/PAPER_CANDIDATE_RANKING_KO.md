# 누적 논문 후보 순위

기록 기준 2026-10-30 KST. **논문 Go 0개.** Hold는 신규성·식별력이 미해결인 옵션이며 실행 선정이 아니다. No-Go는 현재 주장·설계의 투자 판단이며 현상 부재의 증명이 아니다. 과거 실측과 중단 판정을 보존한다.

| 순위 | 질문 | 판단 | 확보한 증거 |
|---:|---|---|---|
| 1 | [G1 · 독립 중복 계산의 자발적 재사용과 오류 공유](../logs/summary/round8_selection_results.json) | No-Go | 결과 저장·재사용과 원입력/표 저장·재계산이 양성 witness와 주 대비를 함께 통과하는 CPU 반례가 있다. 실제 Gemma 모델 측정0. |
| 2 | [S1 · 같은 값의 미사용 출처가 금지 trace의 과승인을 유발하는가](../logs/summary/round8_selection_results.json) | No-Go | 값 일치가 승인에 미치는 효과는 식별 가능하지만, 현재 설계는 휴리스틱·정책 오해·태그 결합 실패를 구분하지 못한다. 정확한 선행 중복을 입증했다고 하지는 않는다. 실제 Gemma 모델 측정0. |
| 3 | [S2 · 개별 정확도보다 오류의 상보성으로 검증자를 선택하는가](../logs/summary/round8_selection_results.json) | No-Go | 쌍별 요약만으로 추가 검증의 정답 기준이 결정되지 않는다. 전체 결합표를 주는 판본도 현재는 새 기제 명제 없이 산술·의사결정 과제에 머문다. 실제 Gemma 모델 측정0. |
| 4 | [H1 · 취소된 값의 직접 읽기와 후속 상태 오염](../logs/summary/round8_selection_results.json) | No-Go | 배타적 기제 비교가 아니고 비율 해석도 성립하지 않는다. 알려진 복사·후속 상태 영향 이상의 명제가 없다. 실제 Gemma 모델 측정0. |
| 5 | [H2 · 답 형식 cue의 직접 재독과 운반](../logs/summary/round8_selection_results.json) | No-Go | 새 기제 예측보다 형식 붕괴 진단에 해당한다. 실제 Gemma 모델 측정0. |
| 6 | [R2-E · 대조 반응으로 후보 개입 크기 예측](../reports/ROUND2_EF_RESULTS_KO.md) | No-Go | 등록 ΔR² .204328; 원시 재감사상 MSE 이득 약91.92%가 string_rule에 집중. |
| 7 | [R7-C · 취소 이력의 logit 흔적](../reports/ROUND7_EXECUTION_RESULTS_KO.md) | No-Go | 등록 합성 특이 증폭 Negative. 512쌍 중 top1변경7·정답변경6, 현재 obsolete 답 선택0행. |
| 8 | [R7-E · 교란 계열 간 고정 예측기의 전이](../reports/ROUND7_EXECUTION_RESULTS_KO.md) | No-Go | 등록 ΔR² −.099841 CI[−.210115,.0149589], 80예측 target 이전 저장 확인. |
| 9 | [R1 · 단일 native 뉴런의 의미 기능](../reports/RESEARCH_SUMMARY_KO.md) | No-Go | 카탈로그 네 후보 모두 ambiguous; 무작위 대조와 독립 검증 통과 없음. |
| 10 | [R2-F · 두 뉴런 사이 손상과 복구](../reports/ROUND2_EF_RESULTS_KO.md) | No-Go | NOT_IDENTIFIABLE; 기존 판정 보존. |
| 11 | [R3-A · 초기 후보 네 개의 수치 자격](../reports/ROUND3A_RESULTS_KO.md) | No-Go | 네 후보 모두 numerical qualification 실패. |
| 12 | [R4 · 정밀도/연합/미관측 개입 예측 설계](../reports/ROUND4_RECOMMENDATIONS_KO.md) | No-Go | DRAFT/NOT_EVALUATED; 일부 문제의식은 별도 R5/R7에 반영. |
| 13 | [R5 · 피드백에 따른 경로 모집](../reports/ROUND5_RESULTS_KO.md) | No-Go | 16고정 후보·적격쌍0. |
| 14 | [R6-A/B/C · 선택 병목·백업 교정·중간값 재사용](../reports/ROUND6_EXECUTION_RESULTS_KO.md) | No-Go | 원래 A와 별도 A2, B, C 모두 등록 prerequisites에서 중단. |
| 15 | [R7-A · 진단 예시의 논리 영향 범위](../reports/ROUND7_INTERPRETATION_AUDIT_KO.md) | No-Go | Inconclusive; 48모순 R1 anchors 보존, main 미측정. |
| 16 | [R7-B · 질의 전 carrier 경로의 색인 전달](../reports/ROUND7_INTERPRETATION_AUDIT_KO.md) | No-Go | 등록 protocol Negative; full-vocab 코드 붕괴; 12draw/11고유view. |
| 17 | [R7-D · 선행 무의미 문맥의 선택적 순서 효과](../reports/ROUND7_INTERPRETATION_AUDIT_KO.md) | No-Go | Inconclusive; forward57/64,inverse32/64. |
| 18 | [S3/S4/S5 · namespace·결말 편향·목표 정보 요청](../logs/summary/round7_next_candidate_screening.json) | No-Go | 모두 NOT_EVALUATED, 신규성 gate에서 중단. |

## 1. G1 · 독립 중복 계산의 자발적 재사용과 오류 공유

No-Go — 계산 정체성에 따른 자연 경로 공유가 차별점 후보다. 일반 prefill note·재사용·transplant·necessity는 Models Take Notes/Hidden APIs가 이미 검사했다.

부족한 결정적 근거: 현재 안을 실행하지 않는다. 경쟁 기제가 다른 결과를 예측하는 독립 명제가 필요하다.


## 2. S1 · 같은 값의 미사용 출처가 금지 trace의 과승인을 유발하는가

No-Go — 후보 차이는 값 일치의 무작위 인과 대비다. LLM 자체의 계보 판정은 LLMCAPLENS·CoRe·NeuroTaint가 이미 다룬다. 정확한 같음 대비의 독립 기여는 UNVERIFIED이며 현재 실행 Go가 아니다.

부족한 결정적 근거: 현재 안을 실행하지 않는다. 경쟁 기제가 다른 결과를 예측하는 독립 명제가 필요하다.


## 3. S2 · 개별 정확도보다 오류의 상보성으로 검증자를 선택하는가

No-Go — 오류 상관 측정과 CAVEAT의 자동 correlation-aware routing은 알려져 있다. 표를 읽는 모델의 의사결정 자체는 다르지만, 이를 독립적인 새 학술 명제로 볼 근거가 부족하다.

부족한 결정적 근거: 현재 안을 실행하지 않는다. 경쟁 기제가 다른 결과를 예측하는 독립 명제가 필요하다.


## 4. H1 · 취소된 값의 직접 읽기와 후속 상태 오염

No-Go — 기존 misleading-context copying과 prefill downstream influence를 넘어서는 새 예측이 없다. 취소 이력이라는 대상 차이만으로 독립 기여가 되지 않는다.

부족한 결정적 근거: 현재 안을 실행하지 않는다. 경쟁 기제가 다른 결과를 예측하는 독립 명제가 필요하다.


## 5. H2 · 답 형식 cue의 직접 재독과 운반

No-Go — General Filter Heads의 eager/lazy 경로 검사나 prefill note를 새 형식 대상에 적용하는 수준이다.

부족한 결정적 근거: 현재 안을 실행하지 않는다. 경쟁 기제가 다른 결과를 예측하는 독립 명제가 필요하다.


## 6. R2-E · 대조 반응으로 후보 개입 크기 예측

No-Go — 등록된 조건부 예측 관계는 보존한다. dose 사후 매칭·과제 집중·가까운 susceptibility 선행을 넘는 새 인과 명제가 없어 현재 독립 논문/추가 실행은 No-Go.

부족한 결정적 근거: 기존 확증을 반복 최적화하지 않고 독립적 새 예측 명제/이론이 필요. R7-E 음성은 R2 소급 무효화가 아님.


## 7. R7-C · 취소 이력의 logit 흔적

No-Go — 현재 좁은 관찰 이상의 독립적 학술 기여 부족.

부족한 결정적 근거: 새 예측 이론 없이 marker 교체·prompt 재탐색 금지.


## 8. R7-E · 교란 계열 간 고정 예측기의 전이

No-Go — Prospective fixed-algorithm Negative로 보존; 보편적 공통 인자 부재 명제는 불가.

부족한 결정적 근거: 과제별 사후 양성을 고르지 않고 독립적인 이론이 있어야 새 연구 가능.


## 9. R1 · 단일 native 뉴런의 의미 기능

No-Go — 검출 대비만으로 행동 의미를 부여할 수 없음.

부족한 결정적 근거: 새 분할에서 의미 선택적 인과 효과와 대체 설명 배제. 현재 탐색을 그대로 재개하지 않음.


## 10. R2-F · 두 뉴런 사이 손상과 복구

No-Go — 등록 손상 자격이 부족해 복구 의미 식별불가.

부족한 결정적 근거: 새 원인 명제 없이 dose를 높여 손상 만들기 금지.


## 11. R3-A · 초기 후보 네 개의 수치 자격

No-Go — 정밀도·readout 위생 자체를 novelty로 세지 않음.

부족한 결정적 근거: Native 개입의 동등 전달 및 단순 정밀도 재발견을 넘는 새 명제.


## 12. R4 · 정밀도/연합/미관측 개입 예측 설계

No-Go — 아이디어 검토 라운드였고 실험 완료로 세지 않음.

부족한 결정적 근거: 다른 라운드 결과를 R4 확증으로 소급 처리하지 않음.


## 13. R5 · 피드백에 따른 경로 모집

No-Go — 적격 차단이 없어 모집 명제 미측정.

부족한 결정적 근거: 새 식별 가능한 질문이 있어야 별도 연구.


## 14. R6-A/B/C · 선택 병목·백업 교정·중간값 재사용

No-Go — 준비도/선행 자격 실패이며 실제 병목이나 재사용 부재가 아님.

부족한 결정적 근거: 층/크기를 늘리기보다 구별되는 명제와 준비도가 필요.


## 15. R7-A · 진단 예시의 논리 영향 범위

No-Go — 우리의 모순 anchor 설계 오류를 모델 결함으로 제시할 수 없음.

부족한 결정적 근거: 단순 수정 재실행을 새 논문 질문으로 세지 않음.


## 16. R7-B · 질의 전 carrier 경로의 색인 전달

No-Go — 47층 궤적 교환·원문 재접근과 format 붕괴로 자연 저장 기제 주장 불가.

부족한 결정적 근거: 허용 경로 양성 format 대조는 위생 과제이며 독립 novelty 아님.


## 17. R7-D · 선행 무의미 문맥의 선택적 순서 효과

No-Go — 역조회 준비도 실패로 priming 미측정.

부족한 결정적 근거: Serialization 또는 알려진 reversal의 재발견을 독창성으로 세지 않음.


## 18. S3/S4/S5 · namespace·결말 편향·목표 정보 요청

No-Go — 현재 설계가 기존 binding/process verification/information-seeking 연구를 넘지 못함.

부족한 결정적 근거: 구별되는 메커니즘을 제시하기 전 실행 예산 배정 없음.

현재 표의 순서는 검토·보존 순서이며 추가 실행 추천 순위가 아니다. [최종 탐색 결과와 종료 근거](AUTONOMOUS_DISCOVERY_OUTCOME_KO.md).

[R7 실측 이후 실제 공동 심사](ROUND7_REVIEW_AND_NEXT_KO.md) · [계산 가능한 순위·근거 해시](../logs/summary/paper_candidate_ranking.json)
