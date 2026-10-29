# R7 사후 공동 심사와 다음 연구 후보

다섯 P0를 실제 실행하고 같은 핵심 증거를 GPT Pro와 Claude Opus 5.5 Extra에 전달해 독립 심사 및 적대적 교차검토를 완료했다. **즉시 P1/P2 승격과 논문 Go는 모두 0개다.** 실행 성공, 등록된 과학 판정, 논문 투자 판단을 구분한다.

실제 모델 forward는 **4,278회**다(science 4,108 + technical 170). A/D는 준비도 중단으로 핵심 명제가 미측정이고, B/C/E는 각 고정 프로토콜의 등록 Negative다. P1/P2와 R8은 NOT_EVALUATED다. [원시 결과·그림·환경](ROUND7_EXECUTION_RESULTS_KO.md) · [해석·분할 감사](ROUND7_INTERPRETATION_AUDIT_KO.md)

## 실제 협업과 접근 범위

| 검토자 | 독립 검토 | 교차검토·한계 |
|---|---|---|
| [GPT · 연구 아이디어 제안](https://chatgpt.com/c/6ae2d6e2-af1c-83ee-b796-7145db91d8b2) | 실제 완료, UI 18m21s | 실제 완료, 7m17s. Pro 웹 검토이며 별도 Deep Research 보고서 아님. 원시 로그 독립 재계산·GPU 재현 아님. |
| [Claude · 신규성 중심 논문 아이디어 독립 검토](https://claude.ai/chat/aa908934-d215-41b7-834b-53f83f60a5e7) | 실제 완료, Opus 5.5 Extra | 실제 완료. 이번 응답에서 원전을 다시 읽지 않았다고 명시. 의견 속 인용은 Codex가 별도 원전 확인. |

다른 심사자의 의견을 처음부터 공유하지 않았다. 두 독립 답변 완료 후 양측 평가와 Codex의 사실 검산을 같은 교차검토 패킷으로 보냈다. GPT의 첫 긴 첨부 요청은 서버 오류가 났고, 전체 핵심 증거를 입력란에 펼쳐 재전송한 실제 완료 답변만 심사로 집계했다. 원문·전송문·완료 화면은 로컬 runs에 보존하며 공개 JSON에는 SHA256과 접근 범위를 기록한다.

## 수용한 비판과 바로잡은 사실

- **A:** 모순된 R1 anchor 48개는 우리의 설계 오류다. 규칙 일반화나 모델 능력 부재의 증거로 쓰지 않는다. 원본·gate를 바꾸거나 수정 반복으로 구제하지 않았다.
- **B:** full-vocabulary 형식 붕괴와 sealed explicit-answer 양성 대조의 부재는 실제 한계다. 반면 Claude의 “코드표도 차단됐다”는 설명은 249개 고유 encoding의 토큰 위치와 맞지 않아 철회됐다. 47층 궤적 교환과 반복 원문 접근은 단일 compact state의 기억을 검정하지 않는다.
- **C:** “정확도 미측정”은 틀려 철회됐다. 1,024행의 정답과 전체 어휘 top1을 기록했다. 512개 old-value 교환 쌍 중 top1은 7쌍, 정답 여부는 6쌍에서 바뀌었고 현재 언급된 obsolete 답 선택은 0행이다. 따라서 “행동 변화가 전혀 없다”도 지나친 표현이다. 무관 이력 값과 FINAL 값의 우연 일치가 편향을 반드시 0 쪽으로 만든다는 주장 역시 근거가 없어 철회됐다.
- **D:** forward와 inverse 답이 같은 14/32 질의는 연산자 사용을 구별하지 못했다. 원 oracle 모순은 아니지만 준비도 판독을 약하게 만든다. 본 순서 효과는 미측정이다.
- **E:** 6개 보정 세계로 고정한 특정 알고리즘의 prospective Negative는 유효하다. 보정 실패가 원인이라고 확정하거나 모든 예측 가능성이 없다고 말하지 않는다. target만 잔차화한 상관은 partial Spearman이 아니며 상관 부호로 두 설명을 이분하지 않는다.

B는 등록 12회 latent draw에 11개 고유 main view가 있었다. 복원추출의 중복 자체가 곧 통계 무효는 아니다. 원래 draw 가중 추정과 11개 view 동일가중 사후 민감도는 다른 추정 대상이므로 둘 다 보존한다. 미사용 P2의 B/D/E 입력 중복도 발견했지만 P2 출력은 보지 않았고 독립 확증으로 주장하지 않았다. 실제 E calibration–P0 prompt 중복은 0이다.

## 의견 차이와 연구 책임자의 판단

GPT는 C의 좁은 margin 관찰과 E의 고정 알고리즘 음성을 논문 가능성 Hold로 남겼다. Claude는 새 이론이나 독립 행동 명제가 없으므로 No-Go를 주장했다. Codex는 **측정 증거를 보존하되 현재 주장의 독립 논문·추가 실행은 No-Go**로 판단한다. 이는 효과가 0이라는 결론이 아니다. 다른 이론에서 새 예측이 생기면 별도 연구로 재검토할 수 있다.

GPT는 최초 후속 1순위의 조건부 Go를 LLMCAPLENS·CoRe를 확인한 후 Hold로 낮췄다. Claude는 내부 유사도 예측·철회 문구·직렬화 역조회 세 제안을 모두 철회했다. Codex는 명시 trace 승인 과제가 자동 채점 가능하다는 점은 인정하나, 같은 답만으로 실제 내부 사용 출처를 알아냈다고 주장하지 않는다. 기존 binding 평가 이상의 예측이 없는 namespace는 현재 No-Go다. 모델 간 합의는 신규성의 증거로 쓰지 않았다.

## 다음 다섯 제안의 신규성 심사 순위

**이 표는 다섯 실행 가능 연구의 선정표가 아니다.** Go 0, Hold 2, No-Go 3이다. 모두 미등록·미실행이며 아래 호출 수는 심사 중 제안된 예산 초안이다. 이미 기각한 연구의 층·dose·prompt 탐색을 새 질문으로 세지 않는다.

| 순위 | 제안 | 판단 |
|---:|---|---|
| 1 | 같은 값의 미사용 출처가 금지 trace의 과승인을 유발하는가 | Hold |
| 2 | 개별 정확도보다 오류의 상보성으로 검증자를 선택하는가 | Hold |
| 3 | 동명 대상을 가진 독립 namespace의 분리 | No-Go |
| 4 | 정답 결말이 잘못된 중간 단계를 승인하게 하는가 | No-Go |
| 5 | 전체 세계 정보량보다 목표 답 관련 질문을 선택하는가 | No-Go |

### 1. 같은 값의 미사용 출처가 금지 trace의 과승인을 유발하는가 — Hold

- **연구질문·가설:** 선택된 출처·실행 trace·실제 출력은 고정하고, 사용하지 않은 허용 출처의 값만 같게 만들면 금지 trace의 승인률이 증가하는가?
- **신규성:** 후보 차이는 값 일치의 무작위 인과 대비다. LLM 자체의 계보 판정은 LLMCAPLENS·CoRe·NeuroTaint가 이미 다룬다. 정확한 같음 대비의 독립 기여는 UNVERIFIED이며 현재 실행 Go가 아니다.
- **이전 실측과 연결:** R7-C에서 최종 사실이 같아도 이력에 따른 logit 흔적이 남았지만 obsolete 답 선택0이었다. 값의 진실성과 사용 허용을 분리하는 다른 명제를 검토할 동기일 뿐이다.
- **구체 설계:** 새 세계에서 사용 출처 권한×미사용 출처와 값 일치×단순 trace 깊이×응답 코드. 명시한 trace와 피연산자 태그 전파 정책을 생성기로 채점한다. 실제 내부 출처 사용을 추정하지 않는다.
- **대조·통제·지표:** nonce 출처·명시 whitelist·권한/코드/행 위치 교환. 금지−허용의 값 일치 승인 증가 차이와 logit margin을 세계 단위로 분석. 준비도는 값이 다른 별도 사례에서만 ≥.90, code ≥.95로 확인하고 주효과와 gate를 혼합하지 않는다.
- **결과별 해석:** 양성은 주어진 trace의 정책 판정에서 값 일치의 영향이다. 실제 내부 정보 흐름의 증거가 아니다. practical SESOI10pp 아래로 CI 상한이면 해당 크기의 과승인 가설 중단, 넓으면 미결정, anchor 실패는 식별불가다.
- **실패·교란:** 정책 존재 기준 대 taint 합집합의 혼동, 이름/거리/코드 편향, trace 읽기 실패, 승인 일반편향, 같은 값에서 실제 경로를 관측했다고 하는 오류. 주효과 셀의 계보 교환 성공을 gate로 쓰면 효과를 사전에 제거할 위험이 있다.
- **예산·난이도:** GPT 검토의 576-forward 상한 초안(384main+128anchor+64technical). 미등록 예산이며 구현·실행0회. 실험 전 configs/의 새 등록과 입력 계약이 필요하다.
- **논문 기여:** 명확한 새 원리/예측을 선행의 taint 판정 이상으로 제시해야 한다. 단일 모델의 새 분류 실패나 좁은 동등값 변형만으로는 논문 Go가 아니다.

재검토에 필요한 근거: 기존 benchmark가 이 인과 대비를 다루는지 방법·데이터 수준에서 확인하고, 새로운 메커니즘을 구분할 예측을 마련할 것.

원전: [Exploring Static Taint Analysis in LLMs: A Dynamic Benchmarking Framework for Measurement and Enhancement](https://yuanxzhang.github.io/paper/LLMTaint-ase25.pdf) · [CoRe: Benchmarking LLMs’ Code Reasoning Capabilities through Static Analysis Tasks](https://arxiv.org/html/2507.05269) · [Ghost in the Agent: Redefining Information Flow Tracking for LLM Agents](https://arxiv.org/html/2604.23374v1) · [Defeating Prompt Injections by Design](https://arxiv.org/html/2503.18813v1) · [AGATE: Provenance-Based Runtime Defense Against Compositional Attacks on LLM Agents](https://arxiv.org/html/2609.30830v1)

### 2. 개별 정확도보다 오류의 상보성으로 검증자를 선택하는가 — Hold

- **연구질문·가설:** 개별 검증자의 정확도·비용을 같게 두고 기존 판단자와 겹치는 오류 구조만 맞바꾸면 모델은 추가 정답 정보를 주는 검증자로 선택을 바꾸는가?
- **신규성:** 오류 상관 측정과 CAVEAT의 자동 correlation-aware routing은 알려져 있다. 표를 읽는 모델의 의사결정 자체는 다르지만, 이를 독립적인 새 학술 명제로 볼 근거가 부족하다.
- **이전 실측과 연결:** R7-E의 과제별 반응 차이는 보편적 한 변수 예측을 지지하지 않았다. 오류 중첩의 알려진 구조를 모델이 쓸 수 있는가로 옮기는 동기이지 현재 그 능력에 대한 증거는 없다.
- **구체 설계:** 생성 가능한 이진 정답·판정 이력표에서 검증자 정확도는 같게 고정하고 상보적인 오답 집합을 교환한다. 질의 시점에 알 수 있는 정보와 oracle-optimal 추가 정보량을 먼저 정의한다.
- **대조·통제·지표:** 오류 개수 세기 준비도, 동일 비용, 행/검증자/응답 코드 순열, 독립 오류 대조. 교환 시 oracle-optimal 선택 전환률 및 추가 정보량 대비 regret. 세계 내 반복 조건을 먼저 집계한다.
- **결과별 해석:** 양성도 우선 합성 표에서의 의사결정 능력이다. 실제 LLM 검증기 신뢰도 추정이나 일반적인 다중 에이전트 신뢰성 개선으로 확대하지 않는다. 산술 준비도 실패는 선택 능력 부재와 구분한다.
- **실패·교란:** 조건부로 어떤 판정이 주어졌는지에 따라 정보량 정의가 바뀜, 다수결과 가치 함수 혼동, 과거 오류표에서 미래 오류가 알려진다는 과도한 가정, 단순 카운팅의 재명명.
- **예산·난이도:** GPT의 704-forward 초안(512main+128anchor+64technical), 미등록·미실행. CAVEAT 원문 mirror 확인 뒤에도 실제 구현보다 질문 차별화가 먼저다.
- **논문 기여:** 알려진 Bayesian 선택기를 재현하는 표 문제를 넘어 모델의 오류 독립성 가정을 구분하는 예측이 필요하다. 현재 신규성 Hold.

재검토에 필요한 근거: 실제 선택 알고리즘 연구와 모델 평가의 정확한 차이 및 비산술적 메커니즘을 특정할 것.

원전: [CAVEAT: Reliability-aware active verification under correlated errors for multi-agent LLM reasoning](https://www.researchgate.net/publication/414994711_CAVEAT_Reliability-aware_active_verification_under_correlated_errors_for_multi-agent_LLM_reasoning)

### 3. 동명 대상을 가진 독립 namespace의 분리 — No-Go

- **연구질문·가설:** 같은 이름을 쓰는 독립된 두 세계의 답을 세계 경계에 따라 분리해 유지하는가?
- **신규성:** 계층 namespace는 가능성이지만 현재 제안은 기존 binding/scope 문제 이상으로 구별되는 메커니즘이 없다. Mixing Mechanisms의 실제 counterfactual patching보다 약한 정확도 비교로 되돌아간다.
- **이전 실측과 연결:** R7-A의 모순 anchor와 C의 이력 잔류는 서로 다른 세계를 구분해야 한다는 진단 동기다. 그러나 A 오류를 모델의 namespace 실패 근거로 쓰면 안 된다.
- **구체 설계:** 이름 공유×명시 세계 경계×질의 세계를 교차하고 서로 다른 세계의 정답을 경쟁 답으로 둔 독립 세계 생성기 초안.
- **대조·통제·지표:** 이름 빈도·순서·거리·코드 배정·정답 token 균형, 잘못된 세계의 답 선택률, 세계 단위 상호작용.
- **결과별 해석:** 양성도 현재는 알려진 문맥 binding 오류의 재현이다. 신규 메커니즘을 구분하지 못하면 본 실험에 진입하지 않는다.
- **실패·교란:** lexical/position/reflexive 경로 혼합, 세계 표지 빈도, 길이 증가, 겉보기 scope 오류를 단순 회수 실패와 혼동.
- **예산·난이도:** GPT 제안576-forward 초안. 미등록·미실행, 현재 비용을 배정하지 않는다.
- **논문 기여:** 현재 독립 논문 기여 불충분. 정확히 상충하는 메커니즘 예측이 생겨야 재검토한다.

재검토에 필요한 근거: 기존 세 binding 기제와 별개의 인과 예측.

원전: [Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context](https://arxiv.org/html/2510.06182v1)

### 4. 정답 결말이 잘못된 중간 단계를 승인하게 하는가 — No-Go

- **연구질문·가설:** 동일한 잘못된 중간 단계 뒤의 최종 답만 무작위화하면 그 단계의 유효성 판정이 달라지는가?
- **신규성:** 제안된 결말 무작위화는 자연 생성 풀이와 다르지만 process/outcome 불일치와 lucky guess의 분리는 PRIME가 이미 평가한다. 현재 일반 outcome bias 이상의 명제가 없다.
- **이전 실측과 연결:** R7-B의 제한 코드 판독과 full-vocabulary 출력의 분리는 측정 인터페이스를 분리해야 한다는 동기다. 검증 편향의 실제 증거로 전용하지 않는다.
- **구체 설계:** 자동 채점 가능한 비트 연산에서 고정된 중간 단계의 정오와 뒤에 붙인 결말 정오를 교차하는 초안.
- **대조·통제·지표:** 단계 자체의 정오 anchor, 결말 없는 조건, 코딩/순서 교환, 잘못된 단계 승인률과 결말 treatment 차이.
- **결과별 해석:** 현재 양성은 outcome bias 재현이다. 특정 인과 메커니즘을 구분하지 않으면 논문 주제로 승격하지 않는다.
- **실패·교란:** 전체 풀이 승인 대 한 단계 승인 지시 혼동, 과제 이해, 코드 편향, 참조 답 노출.
- **예산·난이도:** GPT 제안384-forward 초안. 미등록·미실행, 예산 배정0.
- **논문 기여:** 현재 기존 benchmark의 작은 변형. 신규성 게이트에서 중단한다.

재검토에 필요한 근거: 기존 process verification의 결과와 모순되거나 이를 예측하는 새 원리.

원전: [PRIME: A Process-Outcome Alignment Benchmark for Verifiable Reasoning in Mathematics and Engineering](https://aclanthology.org/2026.acl-long.683.pdf)

### 5. 전체 세계 정보량보다 목표 답 관련 질문을 선택하는가 — No-Go

- **연구질문·가설:** 가능세계 전체의 불확실성 감소가 같아도 목표 답을 식별하는 질문을 선택하는가?
- **신규성:** MT-InfoSeek는 이미 target identifiability·minimal sufficient set·질문 선택과 답 생성 분리를 형식화하고 평가한다. 현재 정확한 작은 대비의 최초성도 미확정이다.
- **이전 실측과 연결:** R7-A/D 준비도 실패는 답 생성과 평가 대상 능력을 분리할 필요를 보여준다. 새로운 정보 요청 능력의 실측은 아니다.
- **구체 설계:** 작은 생성 가능세계 표와 동일 전체 정보량의 질문 후보에서 목표 답을 구분하는 질문을 선택시키는 초안.
- **대조·통제·지표:** 최소 충분 집합 oracle, 목표/질문 이름·코드 교환, 답 사전분포 균형, 최종 정답과 정보 충분성 분리.
- **결과별 해석:** 현재 양성·음성 모두 알려진 정보 수집 benchmark의 변형 가능성이 크다. 주효과가 나와도 독창성이 자동으로 생기지 않는다.
- **실패·교란:** 집합 크기 세기, 가능한 세계 prior 불명확, 모든 목표와 조건의 반복을 독립 표본으로 세는 오류.
- **예산·난이도:** GPT 제안576-forward 초안. 미등록·미실행, 예산 배정0.
- **논문 기여:** 현재 독립 기여 부족으로 신규성 단계에서 중단한다.

재검토에 필요한 근거: MSS 및 target information gain 평가와 구별되는 새 검증 명제.

원전: [Do LLMs Know What to Ask and When? Evaluating Multi-Turn Information Seeking](https://arxiv.org/html/2608.14808v1)

## 현재 가장 강한 명제와 다음 실행 조건

이 고정 모델·동일 BF16 값의 FP32 upcast·표본·프로토콜에서 B/C/E의 등록 양성 기준은 충족되지 않았다. C는 동일한 FINAL 표에서도 취소 이력에 따라 old-value logit 대비가 달라졌지만 등록한 합성 특이 증폭은 지지되지 않았다. E는 독립 보정으로 고정한 예측기의 최소 이득 .02를 새 P0에서 달성하지 못했다. A/D의 핵심 가설은 검정되지 않았다. 자연 기억 부재, 모든 공통 취약도 부재, 의미 뉴런의 발견이나 다른 모델 일반화는 주장하지 않는다.

기존 R7은 닫고 새 실행 예산을 배정하지 않는다. S1/S2는 정확한 신규성·메커니즘 예측을 확인하는 단계다. 다섯 실행 가능한 새 연구는 아직 선정하지 못했다. 이는 가능한 모든 연구질문이 없다는 결론도, 자율 연구 목표 달성도 아니다. 다음 독립 후보 탐색은 열어 두며, 같은 실패 계열의 반복 실행은 하지 않는다.

새 연구는 첫 모델 출력 전에 deterministic anchor의 일관성, 실제 보이는 세계 signature와 authoritative token 입력의 분할 중복을 검사해야 한다. 추가된 입력 계약은 향후 연구용 CPU 인프라다. 실제 R7을 소급 수정하거나 새 실험으로 계산하지 않는다. 과학 실행 당시 CPU 시험 153개와 이후 추가한 4개를 분리했으며 현재 합계 157개가 통과했다. 독립 GPU 반복은 0회다.

[실제 심사 원문 해시·정정·반론](../logs/summary/round7_post_results_peer_review.json) · [27편 원전 접근·본문 확인 범위](../logs/summary/round7_literature_audit.json) · [후보 9항목 JSON](../logs/summary/round7_next_candidate_screening.json) · [누적 논문 순위](PAPER_CANDIDATE_RANKING_KO.md)
