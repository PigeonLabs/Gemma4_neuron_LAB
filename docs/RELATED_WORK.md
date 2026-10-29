# 선행연구와 신규성 검토

확인일: 2026-10-29 (Asia/Seoul). **신규성 판정: NOT_EVALUATED.** 아래는 직접 관련된 5개 원전의 제목·저자·연도·방법을 확인한 제한적 검토다. 체계적 문헌조사나 동등한 조건의 선행연구 재현을 완료한 것은 아니다. 특정 저장소에서 Gemma 4 사례를 찾지 못했다는 이유만으로 최초 발견을 주장하지 않는다.

## 1. 직접 관련된 원전

| 원전·확인한 저자·연도 | 이미 다룬 문제 | 이번 연구와의 관계·차이 |
|---|---|---|
| **Knowledge Neurons in Pretrained Transformers** — Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei. ACL **2022**; arXiv 초고 2021. [ACL 원전](https://aclanthology.org/2022.acl-long.581/), [본문 §4.5–5](https://arxiv.org/html/2104.08696v2) | BERT의 사실 cloze 과제에서 뉴런 attribution, 활성 억제·증폭과 사실 표현, 다른 사실에 대한 영향을 평가했다. | 활성과 정답의 상관을 찾은 뒤 개입으로 평가한다는 설계는 선행한다. 해당 과제의 지식 표현·뉴런 집합 결과를 Gemma의 단일 native 좌표나 번역 기능에 그대로 일반화할 수 없다. |
| **Finding Neurons in a Haystack: Case Studies with Sparse Probing** — Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, Dimitris Bertsimas. **2023**, arXiv v2. [메타데이터](https://arxiv.org/abs/2305.01610), [본문 §5.4·6.1](https://arxiv.org/html/2305.01610v2) | Sparse probe로 native neuron을 찾고 해석을 정제했다. 저자들은 라벨 구성·희소 음성 사례·허위 상관 때문에 분류 점수만으로 강한 인과 결론을 내릴 수 없다고 설명한다. | 높은 activation AUC와 기능 인과성이 구별된다는 문제는 이미 명시돼 있다. 이번 작은 작성 corpus의 높은 AUC는 어휘·문자·문장 형태를 통제한 의미 표상 증거가 아니다. |
| **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning** — Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn 외. **2023** (원문 10월 4일). [저자 원문](https://transformer-circuits.pub/2023/monosemantic-features/), [Anthropic 공식 소개](https://www.anthropic.com/news/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) | SAE dictionary feature와 개별 neuron을 구별하고, 입력에 대한 specificity·sensitivity와 downstream 행동을 나누어 조사했다. | 우리 좌표는 gated MLP의 `down_proj` 입력 한 성분이며 학습한 SAE latent가 아니다. 서로 다른 층의 유사 범주 반응만으로 같은 feature·회로·뉴런 간 연결을 판정할 수 없다. |
| **Steering Language Models With Activation Engineering** — Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid. **2023** 초고; 확인한 v5는 **2024**. [메타데이터](https://arxiv.org/abs/2308.10248), [본문 Appendix F–G](https://arxiv.org/html/2308.10248v5) | Prompt-pair activation 차이로 ActAdd 방향을 만들었다. 자연 activation 대비 개입 norm 및 비슷한 크기의 random-vector 대조도 조사했다. | Inference-time steering·dose·norm·random control 자체는 신규 기법이 아니다. Residual 방향과 native MLP 좌표의 개입 위치·공간이 다르므로 동일 실험이라고 취급하지 않는다. |
| **A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models** — Hamid Kazemi, Atoosa Chegini, Maria Safi. **2026**, arXiv 초고 5월 8일; Apple 공식 게시 7월. [논문](https://arxiv.org/abs/2605.08513), [Apple 저자 소개](https://machinelearning.apple.com/research/single-neuron-safety-alignment) | 저자들은 native MLP의 단일 좌표 개입으로 특정 평가 행동을 바꾸는 충분성 결과를 보고한다. | 단일 native neuron 조작의 행동 효과라는 큰 주장은 이미 선행한다. 이번 무해한 과제의 작은 분포 변화는 그 안전성 과제의 재현도 아니며, 우리 후보의 의미 기능 충분성을 입증하지 않는다. |

위 논문들의 모델·데이터·성공률을 우리 실험 수치로 사용하지 않았다. 날짜는 초고와 확인한 개정판·공식 게시일을 구별했다. Apple 저장소가 연결한 논문은 공식 저자 페이지에서도 확인했다. 이 문헌 목록은 모든 관련 연구를 포괄하지 않는다.

## 2. 현재 측정과 원전 사이의 주장 경계

다음 수치는 **본 저장소의 validation까지 확보한 실제 측정**이다. 원전의 결과와 별개이며, 이후 test·utility 결과를 이 문서의 근거로 소급하지 않는다. [분석 요약](../logs/summary/analysis.json), [Validation 요약](../logs/summary/validation.json), [동결 가설](../configs/hypotheses.json).

| 이번 관찰 | 실제 범위 | 아직 입증하지 않은 것 |
|---|---|---|
| H001/H002/H003 validation AUC=1.0, H004=0.97619 | 후보별 target 3개, other 42개의 사용자 본문 평균 activation | 어휘·언어명·문자·길이·지시 구문을 분리한 의미 기능, 정확한 행동 수행, 필요성·충분성 |
| 4개 후보의 category-selective KL signature CI가 모두 0을 포함 | +1 SD 후보 개입에서 projected-direction control의 seed 평균을 뺀 paired KL의 target-minus-other; stratified prompt bootstrap | 선택적 효과의 확인도, 효과가 엄밀히 0이라는 동등성 결론도 아님 |
| H001–H003 discovery category-profile Pearson **r=0.998583** | 선택된 두 좌표의 동일한 15개 범주 평균 | 독립 재현, 직접 연결·상호작용, 정보 전달, 같은 semantic feature |
| 반례 family AUC: H001 0.875, H002 1.0, H003 0.8125, H004 0.9375 | 후보별 high 4 / low 4 source families; positive와 paraphrase는 같은 family | 대표 입력 분포에서의 선택성·다국어 일반화·행동의 인과 검증 |

Validation signature의 실제 95% CI는 H001 `[−6.5813e−4, 2.0413e−4]`, H002 `[−7.5359e−4, 3.1162e−4]`, H003 `[−5.5042e−5, 3.2357e−4]`, H004 `[−4.9112e−5, 2.6887e−4]` nats다. 작은 양성 표본 때문에 부호와 구간이 불안정할 수 있다. 이 수치에서 새 의미 메커니즘을 선언하지 않는다.

**상관·필요성·충분성은 서로 다른 질문이다.** 높은 AUC는 이 데이터의 라벨을 activation에서 구분한다는 관찰이다. 해당 행동을 수행하는 조건에서 좌표를 제거했을 때 행동이 선택적으로 손실되는지, 행동이 없는 조건에서 좌표를 바꿨을 때 행동이 유도되는지는 각각 별도 개입·행동 지표가 필요하다. 한 좌표의 제거 효과가 작아도 중복 표현·위치·개입 강도·평가 민감도 같은 설명이 남는다. Zero→동일 clean 좌표의 기술적 restoration은 이러한 의미적 필요성·충분성 검사를 대신하지 않는다. 이 구분은 위 probing·knowledge intervention·single-neuron 원전과 우리 설계의 차이를 읽기 위한 방법론적 해석이다.

## 3. Dose·norm·post-normalization을 다룰 때

ActAdd의 Appendix F–G는 자연 forward 대비 개입 크기와 random-vector 비교를 이미 다룬다. 우리 분석의 범위는 **|dose|=1 SD에서 명목상 activation norm 또는 raw down-projection norm을 맞추고 BF16 실제 norm을 별도 기록한 것**이다. 사용자 본문 통계로 산출한 SD를 마지막 wrapper token 개입에 썼으므로, 위치가 일치하는 자연 변동 조건이라고 부를 수 없다. [방법론](METHODOLOGY.md).

Gemma 4 Unified의 `down_proj` 뒤에는 RMSNorm이 있다. 따라서 raw `W_down × delta` norm 일치는 최종 residual delta norm 일치를 보장하지 않는다. 이는 실제 [공식 HF forward 소스](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py#L466)와 [아키텍처 감사](ARCHITECTURE.md)에서 확인한 계산 위치의 차이다. 아직 post-normalization 교란을 계측한 새 결과는 아니다.

Lyra의 [고정 NOTES.md](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/NOTES.md)는 문맥 길이에 따른 해석 차이와, 인과 탐색한 뉴런 집합과 같은 층의 random 집합이 다른 학습 평가에서 비슷할 수 있다는 긴장을 기록한다. 이 자료는 탐색·반례 설계 참고용 소스 노트이며, 동료심사된 일반 결론이나 우리 측정으로 취급하지 않는다. 세부 코드·라이선스 감사는 [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md)에 분리했다.

## 4. 후속 연구를 제안할 때 필요한 선행 비교와 대조

| 가능한 후속 연구 | 이미 존재하는 선행 관점 | 이 pilot이 확립하지 않은 것 | 추가 문헌·실험 대조 | 신규성 |
|---|---|---|---|---|
| 번역·불확실성·프로그래밍의 의미와 표면 어휘 분리 | Gurnee의 probe 해석·허위 상관, Dai의 관계를 표현하는/하지 않는 문맥 비교 | `translate`, 언어명, `Python`, 질문 형태를 넘어선 의미 선택성 | 동일 어휘·상이 의미와 동일 의미·상이 어휘를 교차하고 길이·script·template를 맞춘 새 독립 families; multilingual concept-neuron 연구 추가 조사 | NOT_EVALUATED |
| 같은 token 위치에서 native coordinate·residual direction·SAE feature 비교 | Turner의 activation direction, Bricken의 dictionary feature, Kazemi의 native coordinate | 동등한 intervention budget에서의 우월성·의미 행동 효과 | 같은 layer·token·dtype·자연 범위에서 activation/raw-projection/post-norm 실제 delta를 구별; SAE reconstruction error·random controls·정답 행동 지표; normalization 관련 개입 연구 추가 조사 | NOT_EVALUATED |
| H001/H003 유사 profile의 중복·상보·상호작용 분리 | Sparse representation·composition·feature splitting에 관한 Gurnee/Bricken의 논의 | 높은 r로부터 같은 기능이나 회로를 추론하는 것 | 새 범주·연속 문맥의 covariance, 단독/공동 ablation과 요인별 matched controls, 독립 행동 성능, mediation·circuit-faithfulness 문헌 추가 조사 | NOT_EVALUATED |
| 강도에 따른 분포 변화·행동 손상·유용성의 관계 | Turner의 norm/random-vector 분석, Dai의 다른 지식 보존 평가 | 작은 KL과 greedy 무변화에서 기능 부재·품질 보존을 추론하는 것 | 사전 고정 dose별 norm 대조군, 충분한 길이·독립 prompt·다양한 생성, 행동 정확도와 일반 과제의 paired 평가; robustness·dose 연구 추가 조사 | NOT_EVALUATED |

이 목록은 기존 연구를 재현·구별할 실험 방향이다. 새 데이터나 Gemma 4라는 모델 이름만으로 논문 신규성을 판정하지 않는다. 추가 검색 범위·선행법 비교·인과 대조·불확실성이 갖춰진 뒤 novelty와 publication potential을 다시 평가해야 한다.


## 5. R5: 오류 피드백 아래의 조건부 모집 (설계 검토)

[Conditional Co-Ablation, CoAx](https://arxiv.org/html/2607.01940v3)는 primary 제거 뒤 ablation 효과 증가로 backup을 찾고, 손상/출력 변위 매칭 및 정규화 동결 대조를 다룬다(§2, §3.2, Appendix C.2.2 확인). 따라서 이중 차단이나 즉각적 보상 경로의 발견 자체는 R5의 새 기여가 아니다. R5는 primary 차단을 그대로 둔 상태에서, 같은 참조답과 문제의 오류 status 이력이 남은 좌표군에 대한 query-time 의존성을 바꾸는지를 추가로 묻는다. 최초성은 UNVERIFIED다.

[Hydra Effect](https://arxiv.org/html/2307.15771v1)와 [Explorations of Self-Repair](https://arxiv.org/html/2402.15390v1)는 즉각적 보상·정규화·sparse Anti-Erasure를 다룬다. [Multiple Recovery Paths in a Lesioned Transformer](https://aclanthology.org/2026.blackboxnlp-1.25/)는 readout/mask/encoder 재최적화를 사용한다(이 문헌은 공식 초록 확인, PDF 접근 실패). R5는 가중치와 primary mask를 고정하고 문맥의 오류 프로파일 정보를 조작한다. 정확도 회복만으로 알고리즘 재배치를 주장하지 않으며, query-only 추가 차단과 대응 대조를 통한 제한된 인과적 의존성으로 해석한다. [R5 설계·채택/수정 근거](../reports/ROUND5_DESIGN_KO.md).
