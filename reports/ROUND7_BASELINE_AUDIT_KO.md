# R7 연구 재설정: 기존 증거 재감사

현재 목표는 서로 다른 논문 질문 5개를 선정해 실제 P0부터 검증하고, 결과와 GPT·Claude의 독립 비판·교차검토에 따라 질문을 갱신하는 것이다. 이 문서는 원시 기록 재감사 단계의 중간 기록이다. 새 R7 모델 실험은 아직 0회이며 전체 목표는 진행 중이다.

## 재확인한 관찰

- R2-E: 기록된 64 family의 고정 예측값에서 직접 계산한 ΔR²=0.204327833, 최종 R²=0.141268365. MSE 감소 중 string_rule 몫은 91.9224%다. 원시 전체 재분석 --check도 통과했다. 후보 관측 이후 norm dose를 대조에 전달하는 실행 코드를 확인했으므로 candidate-blind 예측으로 해석하지 않는다.
- R2-F: translate damage 평균 0.011390686 nats, 등록 damage gate 실패. 복구 비율은 NOT_IDENTIFIABLE이다.
- R3-A: 원시 검산 31개를 통과했고 네 후보의 FP32 |signed effect|≥.02 사례는 모두 0/32다. 표본은 8 family의 반복변형이며 32개의 독립 표본이 아니다. 수치 자격 실패와 기능 부재를 구분한다.
- 초기 연구: test의 top-1 변경 39개 조건 행은 모두 test-reasoning-09 한 프롬프트에 집중되었다. validation의 29개 행은 세 프롬프트에 분포했다. 원래 로그에는 이 변경의 정답 oracle이 없어 의미 개선이나 정확도 개선으로 바꾸어 주장할 수 없다.
- R5: 원시 1632행 재검사를 통과했다. 총 1680 forward와 행동 lesion 자격 실패는 그대로다. margin 변화 관측과 행동 손상 자격을 구분한다.
- R6: A/A2/B/C의 기술 검증은 통과했으나 모두 과제 준비도에서 중단했다. 총 744 forward이며 native 후보·calibration·confirmation은 미평가다.

## 새 사후 진단: 오답이 어느 값을 따랐나

| 기존 발견 자료 | 지시 | 직접 표 답 | 합성 답 | 기타 답 | 코드북에 따라 의미 선택이 달라진 질의 쌍 |
|---|---|---:|---:|---:|---:|
| round6_A_discovery | conflict_direct | 26/32 | 4/32 | 2/32 | 5/16 |
| round6_A_discovery | conflict_compose | 23/32 | 7/32 | 2/32 | 6/16 |
| round6x_A_discovery | conflict_direct | 32/32 | 0/32 | 0/32 | 0/16 |
| round6x_A_discovery | conflict_compose | 20/32 | 10/32 | 2/32 | 5/16 |

이는 같은 과거 discovery를 다시 요약한 사후 기술 통계다. 표의 존재, 직렬화, 길이, 실제 계산 경로 중 어느 것이 원인인지는 식별하지 않는다. 새 독립 실험 결과로 세지 않는다.

## 다음 후보를 고르는 원칙

필수 다섯 논문의 방법과 핵심 결과를 1차 자료로 확인했다. [Where Steering Signals Come From](https://arxiv.org/html/2607.25270v1)의 source/position 대비, [Representation Without Control](https://arxiv.org/html/2605.25151v1)의 readout/control 분리, [Stress-Testing Random-Vector Controls](https://aclanthology.org/2026.blackboxnlp-1.73/)의 실현값·구현 감사, [SPARK](https://arxiv.org/html/2607.10296v1)의 길이 보정 susceptibility, [Angle–Norm Decomposition](https://arxiv.org/html/2606.06735v1)의 각도·반지름 분해는 새로운 이름으로 반복할 주제가 아니다. 답 코드 추종과 선계산/지연계산도 직접 가까운 선행이 있다. 각 후보는 이 범위를 넘는 구체적 질문과 기각 가능한 P0를 필요로 한다.

두 기존 채팅에 같은 공통 증거를 전송했다. 독립 검토를 모두 받은 뒤 적대적 교차검토도 완료했다. 실제 응답과 판단은 [R7 후보 보고서](ROUND7_CANDIDATES_KO.md)에 연결했다. 신규성은 UNVERIFIED이며, 두 모델의 동의 자체를 증거로 삼지 않는다. 실용 최소효과와 기술 noise floor, 층별 판독과 인과 기제 판정도 구분한다.

[재감사 JSON](../logs/summary/round7_baseline_audit.json) · [선택값 사후 진단](../logs/summary/round7_route_diagnostics.json) · [원전 방법 중복 감사](../logs/summary/round7_literature_audit.json) · [라운드 자원 한도](../configs/round7_loop.yaml)
