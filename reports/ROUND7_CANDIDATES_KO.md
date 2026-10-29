# R7: 독립적인 질문 다섯 개와 P0 등록

새 모델 출력을 보기 전 선택한 탐색 우선순위다. 논문화 Go와 실험 실행 Go를 구분한다. A의 원시 입력·코드·분석은 별도 해시 등록했고, 나머지 네 후보도 각자 최초 출력 전에 정확한 실행 프로토콜을 등록한다. P0 결과에 따라 P1/P2를 조건부로 진행하며 확증 자료는 열지 않는다.

## 1. A — 진단 예시의 논리적 영향 범위 (Go_P0)

- **1. 질문·가설**: 같은 비트열 과제에서 새 예시의 효과가 논리적으로 두 규칙이 다른 질의에 선택적으로 나타나며 해밍 거리 3에서도 유지되는가?
- **2. 신규성·선행 대비**: Corrective ICL의 오답 피드백이나 일반적인 ICL 일반화와 달리 규칙 동의/불일치×해밍 거리×관련/무관 시스템의 반사실 대비를 고정한다. 최초성 UNVERIFIED. 해밍 거리를 tokenizer/embedding 거리로 대체 주장하지 않는다.
- **3. 기존 실험 연결**: R6 합성 과제 준비도 실패는 일반 합성/내부 메커니즘 주장을 피할 이유다. R2-E의 사후 dose와 무관한, 입력 증거 자체의 인과 효과를 새 자료에서 묻는다.
- **4. 설계**: 24개 독립 규칙 세계. 명시적 규칙 anchor 96, 코드 anchor 48; 준비도 통과 시 1152개의 paired 예시/시스템/질의/코드 조건. 별도 P1/calibration/P2 split은 봉인한다.
- **5. 대조·지표**: TARGET/OTHER, 두 예시 라벨, 두 코드북, near/far 및 두 규칙 동의/불일치. semantic 1−0 logit 차이의 차이, 세계 bootstrap CI, 동의 질의 보존 guard. OTHER의 비식별 질의 정답은 null.
- **6. 결과별 해석**: 앵커 실패=Inconclusive/NOT_IDENTIFIABLE. primary와 far CI 하한>.1 nat 및 보존 guard 통과=탐색 Positive. CI 상한<.1=Negative. 나머지 Inconclusive.
- **7. 교란·실패**: 최근 라벨/특정 비트 복사, 규칙 이름 선호, tokenizer n-gram 중첩, ambiguous OTHER. 규칙/라벨/코드의 균형과 원시 token-ID 보존. 내부 규칙 표상의 증거로 과장하지 않는다.
- **8. 자원·난이도**: 최대 1296 science +64 technical forward; CPU 사전 검증 통과. 12B 단일 GPU, 구현 낮음.
- **9. 논문 기여**: 성공하면 논리 의존성 footprint의 제한된 행동 증거. 실패하면 명시적 규칙 실행과 예시 추론의 구분 및 준비도 한계를 기록. 논문화 Go는 별도 신규성·독립 검증 후 판단.

## 2. E — 교란 계열 사이의 입력별 반응 예측 (Conditional_Go_P0)

- **1. 질문·가설**: 고정된 source 교란 반응은 보지 않은 target native/attention/residual 교란 반응을 과제 내에서도 예측하고 readout-only baseline을 넘는가?
- **2. 신규성·선행 대비**: SPARK의 입력 임베딩 susceptibility/길이 보정과 달리 미관측 개입 계열의 target-blind 예측을 묻는다. geometry나 공통 head gain으로 설명되면 공통 계산 취약도 주장은 폐기한다. 최초성 UNVERIFIED.
- **3. 기존 실험 연결**: R2-E deltaR2의 91.92% 문자열 집중과 사후 dose 전달을 제거해야 한다. R3 수치 자격 실패로 실용 최소효과와 technical floor를 분리한다.
- **4. 설계**: 새 독립 calibration에서 dose envelope를 확인한 뒤 고정한다. 세계 단위 P0, 세 고정 무작위 개입 계열·복수 seed, leave-world-out 예측. source 반응과 target 반응을 다른 계열에서 측정.
- **5. 대조·지표**: clean margin/길이/과제/site norm/readout-only 민감도 baseline, 과제 내 source shuffle, 원시 delta/정답/실제 전달량, 세계 bootstrap heldout MSE 개선. target 전달 geometry는 사후 진단이며 예측 입력 금지.
- **6. 결과별 해석**: baseline을 넘는 target-blind 예측만 탐색 Positive. 수치·행동 자격 실패 또는 CI 불명확=Inconclusive. readout 대조로 모두 설명되면 해당 공통 취약도 명제 No-Go.
- **7. 교란·실패**: 동일 final norm/head, floor/ceiling, residual norm에 따른 실제 dose 차이, 특정 과제 집중, seed 반복을 독립 표본으로 세는 오류.
- **8. 자원·난이도**: ≤1500 science +64 technical forward, 별도 source 등록 후 실행. 단일 GPU, 구현 중간.
- **9. 논문 기여**: 성공하면 개입 계열 간 전이되는 입력 수준 예측 성질. 인과적 공통 기제는 추가 조작 전까지 미확정. 실패하면 susceptibility의 readout 대안을 정량화.

## 3. B — 질의를 모르는 중간 토큰의 색인 전달 (Hold_until_graph_qualification)

- **1. 질문·가설**: 원문부터 답까지 carrier 이외의 모든 경로를 봉인한 그래프에서 질의 전 중립 토큰이 여러 key의 값을 선택적으로 전달하는가?
- **2. 신규성·선행 대비**: Hidden APIs가 pre-future 상태와 재사용 transplantation을 이미 검사한다. 남는 좁은 질문은 구조적으로 전 경로를 막은 자연 토큰 경로의 색인별 전달이다. 최초성 UNVERIFIED.
- **3. 기존 실험 연결**: R6에서 token/표/직렬화가 경로인지 식별하지 못했다. query가 원문을 직접 보는 대안을 실제 계산 그래프에서 제거한다.
- **4. 설계**: 미공개 두 key 질의, 짧은/긴 간격, 질의 전/후 neutral carrier. CPU reachability와 actual additive-mask 기술 자격 후, 같은 key donor/다른 key donor/identity/위치 대조를 고정한다.
- **5. 대조·지표**: 모든 층에서 prefix→carrier 이외 우회 제거, carrier도 봉인한 null graph, same-key 대 other-key donor, 단어별 oracle와 코드북, world-cluster CI와 비표적 보존.
- **6. 결과별 해석**: seal 자격 또는 clean task 실패는 NOT_IDENTIFIABLE. key별 선택 전달 성공은 modified graph의 조건부 충분성. 봉인 실패를 자연 모델의 저장 부재로 해석하지 않는다.
- **7. 교란·실패**: chat marker/query/다른 prefix의 우회, donor의 단순 마지막 행 복사, 전체 hidden overwrite, query/code 누출, 자연 necessity와 modified-graph sufficiency 혼동.
- **8. 자원·난이도**: ≤1500 science +64 technical forward. 긴 입력은 memory/time gate 포함. 구현 높음; 자격 실패 즉시 중단.
- **9. 논문 기여**: 선행과 차별 가능한 path-complete 조건부 전달 검증. 자연 필요성이나 compact interface 발견은 P0 주장이 아니다.

## 4. C — 명시적으로 취소된 이력의 최종 상태 잔류 (Hold_P0_discrimination_only)

- **1. 질문·가설**: 최종 사실표가 동일해도 이미 취소된 수정 이력의 순서/옛 값이 파생 답에 선택적으로 남는가?
- **2. 신규성·선행 대비**: 단순 직접 갱신 대 합성 갱신은 RippleEdits와 겹쳐 폐기했다. 최종 상태를 정확히 같게 고정하고 옛 이력만 바꾸는 좁은 history-dependence로 재정의한다. DeReLab 등 정확한 중복 UNVERIFIED.
- **3. 기존 실험 연결**: R6의 직접표답 선호를 갱신 결함으로 단정할 수 없다. 현재 사실 상태 대 obsolete history를 분리한 새 세계가 필요하다.
- **4. 설계**: 반대 옛 값/동일값 재진술/무관 이력과 동일한 명시적 FINAL 표. direct와 compose를 paired 비교하고 최종표만 조건을 둔다. final-only readiness를 먼저 검사한다.
- **5. 대조·지표**: 동일 final state·query·코드, 이력에 있는 stale 값 지향 logit, direct 대 compose 상호작용, 행 순서·이력 길이 대조, 독립 세계 bootstrap.
- **6. 결과별 해석**: final-only 합성 실패면 NOT_IDENTIFIABLE. old history의 선택적 잔류가 있으면 탐색 현상. 단순 정확도 격차는 신규성 증거가 아니다.
- **7. 교란·실패**: 지시 불이행·이력 길이·자연어 추정·토큰 최근성, 실제 graph update와 혼동. final state를 맞춰도 mechanism은 후속 patch 전 미확정.
- **8. 자원·난이도**: ≤1500 science +64 technical forward. 구현 낮음~중간.
- **9. 논문 기여**: 선행의 편집 성공률을 넘어 상태 동치 입력의 이력 민감성을 검증할 가능성. 불충분한 독창성은 양성 결과와 별개로 No-Go.

## 5. D — 답과 독립인 선행 문맥의 과제 선택적 순서 효과 (Hold_P0_falsification_only)

- **1. 질문·가설**: target 사실을 보기 전 같은 token-ID 다중집합의 순서가 서로 다른 계산에 반대 방향의 효과를 주는가?
- **2. 신규성·선행 대비**: Invisible Reasoning의 filler type/order/patch와 일반 token 이득을 새 발견으로 세지 않는다. target 이전·정확한 토큰 다중집합·과제 선택적 반대 부호라는 좁은 조합만 검정하며 최초성 UNVERIFIED.
- **3. 기존 실험 연결**: R5/R6 scaffolding이나 길이 이득을 내부 계산 공간으로 해석하기 어려웠다. target과 독립인 prefix의 효과는 그 target을 미리 계산한 결과일 수 없다.
- **4. 설계**: 독립 prefix orientation 두 종류×두 target 계산×prefix before/after×코드. 시작/끝 marker는 동일하고 CPU tokenizer로 다중집합을 확인한다. anchor 통과 후 interaction을 측정한다.
- **5. 대조·지표**: 길이/정확한 token-ID multiset/출력형식 고정, 계산 종류별 양방향 selective contrast, 세계 bootstrap와 code별 효과. 단순 confidence 상승은 지지 아님.
- **6. 결과별 해석**: task-selective double dissociation만 후속 검토. 일반 길이/전체 margin 변화 또는 CI 불명확이면 No-Go/Hold. 구문 구조를 n-gram과 독립된 원인이라고 주장하지 않는다.
- **7. 교란·실패**: 순서를 바꾸면 n-gram·거리도 바뀐다. 이 P0는 ordering treatment만 식별한다. 위치·문법 추상화 mechanism이나 추가 계산 주장은 제외한다.
- **8. 자원·난이도**: ≤1500 science +64 technical forward. 구현 낮음.
- **9. 논문 기여**: 현재 입력과 독립인 전처리 문맥의 선택성이라는 제한된 경계 검정. 음성/중복이면 신속 폐기하고 토큰 순서 위생을 신규성으로 포장하지 않는다.

## 실제 협업에서 반영한 변경

두 채팅의 독립 응답을 받은 뒤 같은 교차 패킷을 전송하고 실제 완료 응답을 수집했다. GPT는 Pro 웹 원전 조사이며 별도 Deep Research 완료 보고서는 아니라고 명시했다. Claude는 교차검토의 선행 대부분이 초록 수준임을 명시했다. Codex는 추론 범위를 그 증거 수준에 맞췄다.

GPT와 Claude는 A 우선, B의 우회 봉인, E의 readout 대조에 동의했다. B의 실행 우선순위, C의 P0 가치, D의 즉시 폐기 여부는 달랐다. C는 단순 ripple 명제를 폐기하고 최종 상태 동치의 이력 영향으로 새로 정의했다. D는 추상적 구조나 숨은 계산을 주장하지 않는 저비용 반증 P0로 한정한다. 양성이어도 중복성을 별도로 통과해야 한다.

선행 원전: [Corrective ICL](https://arxiv.org/html/2503.16022v1), [Controlled generalization](https://arxiv.org/html/2505.00661v1), [Hidden APIs §3–5](https://arxiv.org/html/2607.27617v1), [RippleEdits §3–5](https://arxiv.org/html/2307.12976v2), [Invisible Reasoning §4.2](https://arxiv.org/html/2607.22925v1), [SPARK §4](https://arxiv.org/html/2607.10296v1).

[선택 JSON](../logs/summary/round7_candidates.json) · [실제 검토·불일치·판단](../logs/summary/round7_design_peer_review.json) · [A 사전등록](../logs/summary/round7_A_P0_registration.json) · [이전 증거 감사](ROUND7_BASELINE_AUDIT_KO.md)
