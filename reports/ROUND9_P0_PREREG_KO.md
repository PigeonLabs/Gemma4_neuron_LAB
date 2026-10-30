# R9 새로운 다섯 질문 — 출력 관측 전 사전등록

현재 논문 Go는0개다. 우선 실행 A→B→C, 더 엄격한 소규모 선별 D→E 순이다. GPT와 Claude의 독립 평가·실제 교차검토 및 이견을 [심사 JSON](../logs/summary/round9_peer_review.json)에 보존했다. 외부 동의는 실험 증거가 아니며 최초성은 미확정이다.

모델은 google/gemma-4-12B-it, revision707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7이다. 동일 BF16 저장값의 FP32-upcast 조건을 사용하며 참값으로 부르지 않는다. BF16/FP32 계측 검증을 먼저 수행한다. 뉴런을 해석할 경우 down_proj 입력 gated-product 좌표를 사용한다. 이 P0는 행동의 입력 개입이며 새 뉴런/내부 알고리즘을 확정하지 않는다.

| ID | 세계/family | main | 기본 anchor | 과학 상한 |
|---|---:|---:|---:|---:|
| A |16|768|128|896|
| B |24|576|96|672|
| C |24|768|96|864|
| D |8|476|64|540|
| E |16|512|128|640|

모두 통과하면 과학3612＋공유 기술34=3646회다. 기술64회를 포함한 hard cap3700회, wall time14400초, run별4GB다. 한 GPU 프로세스·동결 모델을 사용한다. anchor 중단 시 main을 측정한 것으로 세지 않는다. main 오답 자체는 readiness 기준이 아니다.

코드·질문·rendering·prefix는 family 안의 반복이다. 세계 bootstrap은 탐색적이며 작은 P0의0건 오류나 비유의성으로 모집단 동등성을 주장하지 않는다. top1·정답·margin·전체 어휘 유효 질량을 구별하고 제한 점수로 형식 실패를 구제하지 않는다. CPU 검산은 실제12B 결과와 별개다.

## A. 같은 정보량에서 행동에 필요한 구분을 놓치는가?

1. **명제·경쟁 설명:** 상태별로 성공 행동이 있어도 같은 관측을 받는 상태들이 서로 다른 행동을 요구하면 실행 불가능하다. 관측 호환성 검사와 상태별 해법의 부당한 합성이 경쟁한다.

2. **최근접 선행·남는 신규성:** GENSTRAT §3.1·4·Appendix C는 private information에 따른 최적 행동 차이의 진단축이다. 여기서는 같은 승패표·같은 그룹 수와 크기에서 grouping만 개입한다. 전략 융합 자체는 알려졌고 좁은 대비의 최초성은 UNVERIFIED다. https://arxiv.org/html/2605.23238v1

3. **기존 결과와 연결:** R6에서 main 준비도 실패로 병목을 검정하지 못했다. 이번에는 main의 잘못된 전략 승인 자체가 연구 대상이며 이를 readiness gate로 제거하지 않는다.

4. **구체 설계:** 6상태·4행동 비동형16 family × mixed/공통승리/불가능3구조 × 무관측/전체/유효부분/무효부분4관측 × 전략존재/제시계획2질문 × 코드2 =768 main.

5. **통제·지표·독립 단위:** 단일 승패·공개 순서·코드128 anchor. 유효/무효 부분관측은2그룹×3상태로 고정한다. mixed 무효관측 허위승인에서 불가능 구조의 일반 YES 편향을 뺀 세계별 행동 차이와 YES margin을 기록한다.

6. **결과별 해석·중단:** 행동10pp·margin0.1nat·질문/코드 부호 일치와 controls 조건을 통과하면 P0 Positive다. 정확한 판별은 오류 가설 Negative다. 기본 anchor 실패는 main NOT_EVALUATED이며 새 회로 부재의 증거가 아니다.

7. **핵심 교란·누출 위험:** 작은 게임 공간의 복제, 일반 YES 편향, 공개 시점 오독, 계획 문장의 난도가 대안이다. 모든 payoff 변형을 양쪽 node 순열 canonical form으로 전체 분할 중복 제거한다. 행동만으로 내부 전략 융합 회로를 유일 식별하지 않는다.

8. **비용·구현·후속 분할:** 최대896 science calls, 구현은 유한 정책 완전열거로 검증한다. P1-discovery16/calibration8/P2-32 새 family를 출력 관측 없이 분리했다. P1/P2는 NOT_EVALUATED다.

9. **논문 기여·현재 판정:** 새 관측 구조에서 오류를 사전 예측하는 법칙이 대안보다 잘 맞아야 논문 가치가 있다. 단순 오답률 표는 부족하다. 측정 P0 Go, 논문 Hold.


## B. 경로합을 보존해도 보상의 논리적 위치가 선택을 바꾸는가?

1. **명제·경쟁 설명:** 총액과 최적 경로를 고정한 채 이른 step의 credit를 옮기면 선택이 따라가는가? 합산 불변성, 논리적 앞 보상 선호, 문장에 먼저 보이는 숫자 선호가 경쟁한다.

2. **최근접 선행·남는 신규성:** Myopic Planning §4.1–4.2·5의 가치모형/trace 문단 개입과 Ng1999 §3의 알려진 potential 정리를 구별한다. 논리적 시간×텍스트 위치×명시합을 분리한 행동 위반만 남긴다. 알려진 불변성 정리는 발견이 아니다. https://arxiv.org/html/2605.06840v1 ; https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf

3. **기존 결과와 연결:** R3의 수치 자격 실패를 무시해 dose를 키우는 대신 정확 정수합을 고정하고 실제 답변 선택을 측정한다. 내부 손상/복구 후보를 재명명하지 않는다.

4. **구체 설계:** 24개 두 경로·3step 수치 세계 × baseline/U앞/V앞3배분 × 명시합2 × step나열 정/역순2 × 코드2 =576 main. r′=r+Φ(next)−Φ(current), 시작/종료Φ=0으로 경로합을 보존한다.

5. **통제·지표·독립 단위:** 96개 path-sum anchor. credit는 끝에 함께 지급하고 위험·할인·중도 사용을 제거한다. U앞/V앞의 U선택 차이와 margin을 두 텍스트 순서별로 계산한다. 명시합은 집계 실패 대조다.

6. **결과별 해석·중단:** 두 문장 순서 모두 행동10pp·margin 기준을 통과하고 명시합이 오류를 줄일 때 제한 P1 후보로 남긴다. 텍스트 순서 추종·무변화·기본 합산 실패는 중단한다. 로그잇만 이동하면 행동 명제 양성이 아니다.

7. **핵심 교란·누출 위험:** 숫자 크기, 특정 수 조합, 단일 topology, 지급 시점의 암묵적 해석이 대안이다. 논리적 시간 효과도 내부 할인율이나 얕은 탐색을 유일 입증하지 않는다.

8. **비용·구현·후속 분할:** 최대672 science calls. 새 P1/cal/P2 세계16/8/32개. P1은 미관측 재분배에 대한 경쟁 모형 예측을 고정하고 P2는 새 topology를 별도 등록해야 한다.

9. **논문 기여·현재 판정:** 일반 myopia 재현이면 No-Go다. 합계 불변성 위반의 새로운 예측 가능한 경계가 필요하다. 수정 P0 Go, 논문 Hold.


## C. 무관한 수집 절차와 우도를 바꾸는 선별을 구별하는가?

1. **명제·경쟁 설명:** 같은 비트 기록에서 optional stopping·독립 승인 선별에는 판단을 유지하고 값 기반 선별에만 반응하는가? 기록만 보는 규칙, 절차 어휘에 모두 반응하는 규칙, 실제 우도 추적이 경쟁한다.

2. **최근접 선행·남는 신규성:** From Prompts to Posteriors §4.2·5.1·6은 명시 prior/observation model의 posterior 및 발견 성능을 평가한다. Xiong §3.2·Appendix A.2는 strong/weak 문구 조작을 이미 검사했다. 따라서 C2 단독은 제외하고 우도 보존/변화의 동시 선택성만 남긴다. https://arxiv.org/html/2610.29331 ; https://arxiv.org/html/2605.05851

3. **기존 결과와 연결:** R8-S2의 관측/우도 정의 문제를 반복하지 않도록 oracle를 완전 명시한다. 검증자 선택의 재명명이 아니다.

4. **구체 설계:** 24 장치·기록 세계 × 고정길이/사건기반중단/값기반선별/독립무작위선별4절차 × bit대칭2 × 코드2 × narrative/정확 likelihood2 =768 main. 장치는 한 번만 고르고 discarded block에서도 같다.

5. **통제·지표·독립 단위:** 96개 정확 likelihood-weight anchor. 숨긴 block의 횟수·대기시간을 제공하지 않는다. fixed/stop/random은 같은 posterior, 정확k사건 block 선택은 prior1/2다. 정확한 분수 선택이며 코드 확률을 주관적 posterior로 부르지 않는다.

6. **결과별 해석·중단:** narrative의 선택 대비와 무관한 arm 불변성 screen을 함께 통과할 때만 P0 Positive다. 모든 절차에 같은 반응·어느 절차에도 반응 없음·기본 계산 실패는 중단한다. Positive는 알려진 Bayes 정리의 발견이 아니다.

7. **핵심 교란·누출 위험:** 장치 재선택, hidden rejected counts, 복원추출 누락, 불가능한 stopping record가 치명적이다. 독립 sample-space enumeration으로 확인한다. 작은 P0 screen이나 비유의성으로 모집단 동등성을 주장하지 않는다.

8. **비용·구현·후속 분할:** 최대864 science calls, 새16/8/32개 분할. P1에는 다른 수집 장치로의 고정 예측이 필요하다. 정확 likelihood에서만 풀리면 절차 해석과 일반 산술 실패를 분리해 평가한다.

9. **논문 기여·현재 판정:** 일반 posterior benchmark면 No-Go다. 인과적으로 무관/관련 절차의 새로운 일반화 경계가 있어야 한다. Claude Go/GPT 통합안 Hold 이견을 보존하며 Codex bounded P0 Go, 논문 Hold다.


## D. 저차 통계가 맞아도 고차 결합만 실패하는가?

1. **명제·경쟁 설명:** 각 bit와 모든 bit쌍의 균등성을 유지하면서 지정된 parity 질량 반전만 놓치는가? 일반 분포 생성 실패·규칙 해석 실패·선택적 고차 실패가 경쟁한다.

2. **최근접 선행·남는 신규성:** Bad Dice §3–5는1D target의 stateless/batch 샘플링을 검사한다. 세 bit를 여덟 범주로 바꿀 수 있으므로 다변량화 자체는 신규성이 아니다. 저차 정상/고차 실패가 규칙과 전체 확률표 모두에서 남아야 한다. https://arxiv.org/html/2601.05414v1

3. **기존 결과와 연결:** R8의 고차 결합 CPU 반례를 실제 LLM 결과로 세지 않았다. 이번에는 pretrained12B가 유도하는 명시 decoder 분포를 실제 forward로 계산한다.

4. **구체 설계:** 8개 서로 다른 q와1−q를 짝짓고 규칙/전체확률표2 × 의미bit뒤집기2 × 모든7prefix를 측정한다. 공유 균등 reference28행은 한번만 측정하며 독립 family로 세지 않는다. main476회.

5. **통제·지표·독립 단위:** 64 probability-table anchor. 각 prefix의 두 bit 조건부 확률로 정확한 joint를 재구성한다. 코드·설명별 저차 오차를 먼저 계산한다. 전체 어휘 top1과 유효 질량을 보존해 코드 평균이나 renormalization으로 형식 실패를 숨기지 않는다.

6. **결과별 해석·중단:** 8family 중6개 이상에서 모든8조건의 저차 최대오차≤.05, parity오차≥.15, 가중 유효질량≥.8일 때만 제한 Positive다. 저차부터 틀리거나 전체 표에서 오류가 사라지면 일반 생성/규칙 오류 Negative다.

7. **핵심 교란·누출 위험:** 강제 문법 자체, code bias, 경로 확률 평균 순서가 대안이다. 실제 tokenizer의 canonical 1token bit와 prefix 경계를 CPU 검증한다. 이 분포는 temperature1 binary-grammar decoder이며 자유생성이나 내부 난수원이 아니다.

8. **비용·구현·후속 분할:** 최대540 science calls. 새 q family8/4/16개를 분리했다. P1/P2는 미관측 고차 의존구조 예측의 별도 등록이 필요하다.

9. **논문 기여·현재 판정:** 일반 sampling 실패면 No-Go다. 두 외부 Hold를 유지하되 CPU oracle와 엄격한 대조를 고정한 작은 선별의 정보 가치 때문에 조건부 측정 Go로 결정했다. 논문 Hold다.


## E. 개별 자원 가능성을 전역 실행 가능성으로 합치는가?

1. **명제·경쟁 설명:** 각 task에 선택지가 있고 node별 degree도 같은데, 독점 자원 경쟁의 불가능성을 놓치는가? degree 기반 국소 근사·전역 matching·일반 그래프 난도가 경쟁한다.

2. **최근접 선행·남는 신규성:** NLGraph §2.2 Task6 및 §4는 bipartite matching을 이미 평가했다. 모든 node degree·토큰 빈도를 보존하는 한 번의2edge switch로 정답을 뒤집고 독점/재사용을 교차한다. 일반 matching 실패는 신규성이 아니다. https://arxiv.org/html/2305.10037

3. **기존 결과와 연결:** R5의 같은 답을 내는 여러 내부 경로는 사용 여부를 식별하지 못했다. 이번에는 국소 통계가 같아도 실제 정답이 다른 입력 쌍을 먼저 만든다.

4. **구체 설계:** 16개6×6 비동형 pair × 가능/불가능2 × 독점/재사용2 × 직렬화2 × 코드2 × 전체/표시subset-union2질문 =512 main. 불가능 그래프의3task가2resource에 몰리고 switch 후 완전 matching이 생긴다.

5. **통제·지표·독립 단위:** 128 edge/union-count anchor. Hall subset과 permutation matching oracle를 독립 대조한다. 전체 허위 승인과 같은 세계의 union 판독을 비교하고 가능한 그래프·재사용 양성 대조를 요구한다.

6. **결과별 해석·중단:** 국소 판독 및 양성 대조≥.85인데 전체 허위 승인의 선택적 초과가 등록 기준을 넘으면 Positive다. 정확한 전역 판별·모든 그래프 측정의 동시 실패·재사용에서도 같은 변화면 중단한다.

7. **핵심 교란·누출 위험:** 생성 degree 공간의 좁음, node 이름만 바꾼 복제, 표시subset 단서가 대안이다. 양쪽 node 순열 canonical key를 전체 분할에서 중복 제거한다. 4×4 degree2 정규군은 반례 쌍이 없어 사용하지 않는다.

8. **비용·구현·후속 분할:** 최대640 science calls. 새8/4/16 family 분할. P1은 병목 크기와 국소 선택지를 분리하고 P2는 다른 자원 경쟁 구조를 별도 등록해야 한다.

9. **논문 기여·현재 판정:** 새 matching 문제집으로 끝나면 No-Go다. 국소 가능성 합성이 실패하는 일반화 경계를 사전 예측해야 한다. 외부 Hold를 보존하며 CPU 구조 공간 확보 뒤 조건부 측정 Go, 논문 Hold다.


## 재현과 변경 규칙

[동결 설정](../configs/round9_P0.yaml)과 [등록 JSON](../logs/summary/round9_P0_registration.json)에 정확한 판단식·seed·소스/manifest/token ID 해시·분할 계약이 있다. 실제 원시 입력과 행은 로컬 runs/에 보존한다. CPU 준비에서11900개의 authoritative 입력을 검사했으며 최대313토큰이었다. P1/P2는 입력을 분리했어도 별도 등록·실측 전까지 NOT_EVALUATED다.

실행은 `.venv/bin/python scripts/run_round9.py --execute`, CPU 검증은 `.venv/bin/python -m pytest -q tests/test_round9.py`다. 사후 지표·입력·소스 변경이 필요하면 기존 등록과 실패를 보존하고 새 연구로 명시한다. 낮은 효과를 구제하려고 층·dose·prompt 범위를 확대하지 않는다.
