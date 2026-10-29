# Round 2 E/F 실행 전 등록

이 연구는 공개된 Round 1 결과를 보고 설계한 **새 연구**다. 최종 추천 목록의 5번(R2-E, 입력 취약도 공통 요인)과 6번(R2-F, H001/H003 인과적 대체)을 수행한다. 기존 validation/test와 후보·가설·요약은 변경하지 않는다. 실행 성공과 연구 가설 지지는 별도로 판정한다.

## 협업과 채택한 제안

사용자가 지정한 사이드바의 GPT Pro Deep research와 Claude Opus 5.5 Extra Research에 동일한 R1 근거와 E/F 과제를 전달했다. GPT의 완료 보고서 「R2-E / R2-F Deep Research 설계: 입력 취약도와 뉴런 간 Rescue의 식별 가능성」, Claude의 완료 보고서 「Gemma-4-12B-it 뉴런 후보 R2-E·R2-F 실험 설계: 입력 취약도 분해와 H001→H003 인과 대체가능성 검정」을 읽고 통합했다. 실제 모델 실행과 최종 방법 선택은 Codex가 맡는다. 대화 원문·화면은 로컬 `runs/round2_ef_collaboration/`에 보존한다.

- GPT: calibration 계수 고정, control 측정이 필요한 예측과 zero-probe 예측 구분, signed log-odds, standalone을 뺀 rescue 상호작용, 자연값 median rescue, 손상 부재 시 식별 불가를 채택했다.
- Claude: family 단위 집계, 선택지 교환, H001→H003 1단계 변화와 oracle block을 고정 rescue와 구분, 실제 post-norm 및 잔차 변화, identity·훅 순서·복원 점검을 채택했다.
- 예산 안에서 층당 native random 좌표는 1개, residual random 방향은 calibration 2개·확증 2개의 새 실현값을 쓴다. 이는 무작위 좌표 모집단에 대한 특이성 검정을 허용하지 않는다. 새로운 방향으로의 일반화는 보조 분석이다.
- rescue 분위수를 행동 효과로 검색하는 제안은 채택하지 않는다. H003 자신의 calibration 자연분포에서 지시별 median을 고정한다. copy에서도 copy median을 사용하므로, off-target 결과는 **동일한 raw clamp 값**의 전이를 검증하는 실험이 아니다.
- 이미 알려진 정규화·매개·입력별 steerability를 새 발견으로 주장하지 않는다. 선행연구와의 정확한 차별성은 미확정이다.

## 공통 입력·계측·예산

설정은 [round2_ef.yaml](../configs/round2_ef.yaml), deterministic manifest 생성기는 [round2_ef_corpus.py](../src/neuron_lab/round2_ef_corpus.py)다. E는 calibration 32·확증 64 family, 각 2개의 A/B 순서 변형이다. F는 calibration 16·확증 64 family, 각각 translate/copy × A/B 2순서다. 총 512개 prompt지만 독립 단위는 생성 원본 family다. 두 분할의 prompt shell과 nonce 단어는 다르다. A/B, 색상, 과제 규칙 등은 공유하므로 전혀 새로운 자연어 과제·도메인에 대한 일반화를 검증하지 않는다.

외부 지식 없이 주어진 작은 사전, 문자열 연산, 명시적 함의, 제공된 사실만 사용한다. 정답은 프로그램으로 결정된다. 사전 번역은 가상 언어→제공된 영어 의미이며 한국어 번역 능력을 평가하지 않는다. 답은 첫 토큰 A 또는 B다. 별도 thinking 없이 native chat template의 마지막 prefill 토큰에서 **관측과 개입을 정렬**한다. 본문 전체 위치나 자유 생성 행동은 미평가다. 강제 A/B 정답률과 unrestricted top-1 정답률을 구별한다.

고정 모델은 `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, BF16, eager attention, batch 1이다. 훅은 down_proj 입력의 곱 활성 좌표와 post-MLP RMSNorm 출력에 적용한다. post-norm Δ와 실제 BF16 residual-add 후 layer output Δ를 모두 기록한다. 모델 재로딩마다 기존 28개 BF16 smoke check와 새 정렬 훅 검사를 통과해야 한다. 재주입 일치와 같은 좌표 복원은 기술 검사이며 의미적 복구 증거가 아니다. KV cache 최적화는 사용하지 않는다.

예산은 단일 GPU, 합산 최대 13,500 model forward call, 합산 실행 10,800초(로딩은 별도 기록), run당 2GB다. 기본 조건 행은 E 6,336개 + F 4,480개 = 10,816개다. native random norm 보정과 BF16 검증은 추가 호출이며 전체 예산에 포함한다. 50회 처리량을 실측하고 정해진 상한을 강제한다. 다른 사용자의 프로세스를 종료하지 않는다.

## E: calibration 고정 예측

H001/H002/H003/H004 각각 ±0.5 calibration SD를 적용한다. 각 후보·부호에 대해 같은 층의 variance/write-norm 근접 random 좌표 1개와 post-norm norm을 맞춘 random residual 방향 2개를 비교한다. native random의 scalar는 write norm으로 초기화하고, 결과 logit을 사용하지 않는 최대 1회의 geometry 보정만 허용한다. actual BF16 norm 오차를 전부 기록한다. 방향 대조도 BF16 반올림 후 norm으로 맞춘다.

각 family에서 두 순서, 네 후보, 두 부호의 `abs(Y_intervened−Y_clean)`를 평균한 뒤 log(·+1e−6)를 취한 값을 예측한다. Y는 정답 대 오답 A/B logit 차다. S는 같은 prompt의 무작위 대조 semantic 변화 중앙값을 순서 간 평균한 log 값이다. 모든 회귀 계수와 표준화는 calibration 32 family에서 고정하며 ridge penalty=1, 절편은 무벌점이다.

- 기본 모형: |원래 semantic margin|, unrestricted top-1 margin, 전체 entropy, log 입력 길이, log 후보 post-norm dose, log 실제 후보 layer-output dose.
- 확장 모형: 기본 모형 + S.
- 1차 비교: 확증 64 family에서 확장 모형의 MSE 감소와 ΔR². 계수를 다시 적합하지 않는다. ΔR²는 확증 target 분산을 공통 분모로 사용한다.
- 성공: ΔR² 점추정 ≥0.05, family bootstrap 95% CI 하한 >0, MAE도 감소, 두 1차 검정 Holm p<.05, 효과가 큰 상위 10%·20% family 제거 시에도 ΔR²>0. 전부 0인 반응 또는 target 분산 0이면 NOT_IDENTIFIABLE.
- 보조: baseline 네 변수만 쓰는 모형, baseline+S, native-only S와 두 direction-only S, baseline에서 S를 예측해 연결한 zero-probe 모형, 상위 1·3·5 family 집중도.

1차 결과는 후보와 대조의 **측정된 geometry에 조건화된 control-probe-assisted 예측**이다. 후보 효과와 같은 forward에서 측정한 dose가 기본 모형에 들어간다. 새 입력에서 아무 개입도 없이 효과를 예측하는 주장이나, 입력 공통 요인의 인과적 식별로 해석하지 않는다. zero-probe 모형은 별도로 보고한다. 한 층당 random 좌표가 하나이므로 candidate-specific 의미 기능을 모집단 수준에서 확증하지 않는다.

## F: 손상과 복구의 상호작용

H001은 같은 family·선택지 순서의 반대 지시에서 관측한 자연 활성으로 교체한다(D). H003은 자신의 calibration 지시별 자연 median으로 clamp한다(R). 층 사이 raw scalar를 복사하지 않는다. zero H001은 별도의 2차 손상이다.

14개 조건을 모두 실행한다: clean, D, zero-H001, R, D+R, zero-H001+R, random-native R, D+random-native R, random-direction R, D+random-direction R, 같은 forward H001 clean 복원, joint zero-H001/H003, zero-H003, D+H003 prompt별 clean 값(oracle block).

random native R은 고정된 같은 층 좌표의 calibration 지시별 median이다. actual norm 매칭을 보장하지 않으며 비율을 보고한다. random residual R은 clean R과 같은 post-norm norm이고 clean/D world에서 **같은 vector**를 사용한다. D world의 실제 norm도 기록한다. native 특이성은 geometry가 다르면 제한적으로만 해석한다.

확증 translate 64 family가 1차군이다. 두 A/B 순서를 평균한다.

- 손상: `damage = mean(Y_clean−Y_D)`. 95% CI 하한이 **0.10 nats 초과**여야 rescue 해석 gate를 통과한다.
- 1차 상호작용: `I = (Y_D+R−Y_D)−(Y_R−Y_clean)`.
- 추가 gate: I의 95% CI 하한 >0.05 nats, Holm p<.05, candidate I−native-random I 및 I−direction I의 CI 하한 >0, 확률 척도 I의 평균 부호도 양수.
- 위 기준을 통과해도 기술할 수 있는 것은 이 개입 조건의 손상 의존적 보상 증거다. 일반적 기능 동등성이나 동일한 의미 단위라는 결론은 아니다. geometry와 copy 부작용도 함께 검토한다.
- 손상 gate 미통과 시 **NOT_IDENTIFIABLE**. 모든 등록 조건은 끝까지 실행하되 rescue fraction은 계산하지 않는다. 더 센 dose나 새 damage로 바꾸지 않는다.
- rescue fraction은 gate 통과 시에만 family 평균의 비율로 계산한다. bootstrap 분모가 0을 건너면 여전히 NOT_IDENTIFIABLE. family별 비율 평균은 금지한다.
- H001 변경 뒤 H003 자연 활성 변화(FS), oracle block의 회복, joint zero 비가산성, zero 손상의 I, standalone, copy군은 보조 결과다. oracle block은 고정 median rescue와 다른 반사실이다. 이를 순수한 매개 경로나 직접 연결의 증명으로 부르지 않는다.

## 통계·고정·실패

95% percentile family bootstrap 5,000회, 양측 family sign-randomization 10,000회(16개 이하에서는 정확 열거)다. 반복 부호·순서·direction seed는 독립 표본이 아니다. sign-randomization은 family contrast의 부호 교환 가능성 가정이 필요한 진단이며 연구 대상 자체를 무작위 표집한 것은 아니다. E의 MSE 감소와 F의 I, 두 1차 p에 Holm을 적용한다. calibration 불확실성을 다시 적합하는 구간이 아니라 고정 calibration에 조건부인 확증 구간이다. 나머지 보조 구간은 다중성 조정 없는 기술 통계다.

calibration 이전에 코드·분석식·manifest를 로컬 commit한다. 자연 활성로 SD/대조 좌표/median을 결정하고 calibration E 반응으로 회귀를 적합한다. 파라미터 JSON을 별도 commit한 뒤에만 확증을 실행한다. confirmation은 implementation SHA-256과 manifest hash 및 frozen JSON commit을 검사한다. 각 run에 실제 revision·환경·source hash·split·seed·device·dtype·시간·자원·raw hash를 보존한다. 결과가 null이어도 임계값이나 후보·가설을 바꾸지 않는다. 구현 오류로 수정이 필요하면 실패 run과 변경을 기록하고 새 등록을 한다.

기존 R1 주요 요약·가설·입력·카탈로그 8개 hash가 run 전후 같아야 한다. 합성 CPU 단위 검사는 실제 모델 측정과 분리한다. 데이터가 없는 수치는 NOT_EVALUATED로 둔다.

## 원전과 새 기여의 경계

[Analyzing the Generalization and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404)는 입력별 steerability 차이와 spurious bias를 이미 다룬다. [SPARK](https://arxiv.org/abs/2607.10296)의 susceptibility 보정, [Stress-Testing Random-Vector Controls](https://aclanthology.org/2026.blackboxnlp-1.73/)의 random 실현값·구현 민감도도 선행한다. E의 추가 질문은 고정된 네 native 좌표에 대한 새로운 family에서의 예측 잔여 정보다.

[Every Ablation Is a Dose](https://arxiv.org/abs/2610.02173)는 개입을 signed counterfactual dose로 해석하고 고정된 하류 gain으로 apparent self-repair를 설명한다. [The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510)는 patching 추정량이 다른 구성요소와의 상호작용을 포함할 수 있음을 보인다. F는 이 틀을 넘는 최초 주장을 하지 않으며, 기존 두 후보의 자연-support 제한·standalone 차감·손상 gate가 포함된 검증 사례로 위치시킨다.
