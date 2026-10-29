# 실험 방법과 해석 범위

작성 기준: 2026-10-29, 현재 `hooks.py`, `statistics.py`, `corpus.py`, `runtime.py`, `smoke.py`, `survey.py` 및 `configs/`의 구현. 이 문서는 **측정 정의와 검증 절차**다. 실행 여부와 PASS/FAIL, 표본 수·효과 크기는 각 `logs/summary/`와 run 기록으로 확인한다. 아래 검증 계획을 이미 통과한 결과로 읽으면 안 된다.

## 1. 모델·정밀도·뉴런의 정의

대상은 `google/gemma-4-12B-it`, 고정 revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`이다. native Hugging Face 모델을 BF16, eager attention, `eval()`과 inference mode로 실행한다. 초기 연구는 text-only이며 이미지·음성 경로에 대한 결론을 포함하지 않는다. `trust_remote_code=False`이고 cache 경로가 고정 revision 이름과 일치해야 한다. 계층 수·intermediate dimension·전체 좌표 수와 parameter dtype은 로딩한 config 및 실제 모듈과 대조한다.

계층 `l`의 MLP 입력을 `h`, gate/up/down projection을 각각 `G_l`, `U_l`, `D_l`, 모델에 정의된 activation function을 `φ`라고 쓰면 native 중간 활성은 다음과 같다.

```text
a_l = φ(G_l h) ⊙ (U_l h)       shape [batch, sequence, intermediate]
MLP raw output = D_l a_l
neuron (l,n) = a_l[..., n]
```

**본 연구의 뉴런은 `mlp.down_proj`에 실제로 입력되는 post-gating 중간 텐서의 한 좌표**다. gate 값, up 값, residual coordinate, SAE latent와 구별한다. 훅은 이 텐서를 직접 관찰하므로 Qwen의 SiLU 공식이나 다른 Gemma 세대의 normalization을 대신 계산하지 않는다. 공개 neuron ID는 0부터 시작하는 layer/neuron index와 model revision의 조합이다.

`runtime.py`는 실제 `*.layers.<l>.mlp.down_proj`를 찾고 계층 누락·intermediate width 불일치·BF16 이외 parameter를 거부한다. 다른 아키텍처가 같은 이름을 쓰더라도 이 검사를 통과했다는 이유만으로 기능이 같다고 해석하지 않는다.

## 2. 토큰 범위

입력은 고정 tokenizer의 chat template을 적용한 단일 user message다. `enable_thinking=False`, generation prompt 포함, 입력 최대 256 tokens이며 초과 입력을 조용히 자르지 않는다.

활성 통계는 **prefill의 user-content tokens만** 집계한다. `survey.content_mask`는 렌더링된 template에서 user 문자열의 문자 구간을 찾고 fast-tokenizer offset이 그 구간과 겹치는 token을 선택한다. chat wrapper·역할 marker·assistant generation은 통계에서 제외한다. tokenizer가 반환한 token IDs가 실제 모델 입력과 정확히 같아야 하며, 다르면 실패한다. 문자 경계와 겹친 token은 포함되므로 경계 token에 인접 공백 등이 일부 들어갈 수 있다. 이 mask는 다른 chat template이나 다중 message 데이터에 자동 일반화되는 정책이 아니다.

통계의 token position은 masking 이후의 순위가 아니라 **원래 chat token sequence 안의 0-based 위치**다. 현재 수집은 prompt당 batch 1이다. 보존한 단일 token의 decoded 문자열은 전체 문맥의 뜻을 대신하지 않으며, 다국어·subword 토큰은 전체 prompt와 함께 읽어야 한다.

개입의 위치 옵션은 통계 mask와 별개다.

| 옵션 | 실제 적용 범위 |
|---|---|
| `all` | 해당 forward 호출의 모든 position. prefill에서는 wrapper도 포함할 수 있다. |
| `last` | **매 forward 호출의 마지막 position**. prefill 마지막 token, KV-cache decode의 현재 token에 적용된다. |

`last`는 전체 생성 중 딱 한 번의 절대 index를 뜻하지 않는다. 이 위치의 logits 변화는 그 다음 token 선택에 영향을 준다. `use_cache=False` teacher forcing과 cached generation에서 동일한 문구만 보고 개입 범위가 같다고 가정하지 않는다. Run에는 사용한 위치 정책과 입력·생성 token 수를 함께 남긴다.

## 3. 스트리밍 활성 통계

한 계층·뉴런에서 prompt `p`의 선택된 token 집합을 `T_p`, token 수를 `N_p`, 활성값을 `a_pt`라 하자. 저장하는 통계에는 서로 다른 가중 방식이 있으므로 혼용하지 않는다.

### Token 단위 통계

```text
N = Σ_p N_p
μ_token = Σ_p Σ_t a_pt / N
σ_token = sqrt(Σ_p Σ_t (a_pt - μ_token)^2 / N)
frequency = Σ_p Σ_t 1[|a_pt| > 1e-6] / N
```

`mean`, `std`, `min`, `max`, `frequency`는 이 범위의 token 기준이다. `std`는 표본 표준편차가 아닌 **population standard deviation**이다. 긴 prompt는 token 단위 통계에 더 많이 기여한다. `frequency`는 양의 활성 비율이나 ReLU 발화율이 아니며, 위 절댓값 threshold를 넘은 비율이다. 부호와 크기가 다양한 gated MLP에서는 이 값이 거의 1이어도 의미 선택성이 높다는 뜻이 아니다.

모델 활성은 BF16이고 집계는 기본 FP32 parallel Welford 방식이다. 전체 activation tensor를 CPU에 계속 저장하지 않고 GPU에서 각 prompt의 moments/extrema를 합친다. min/max는 실제로 관측한 선택 token 전체의 extrema다. FP32 집계와 BF16 자체의 반올림 때문에 수학적 실수 연산과 완전히 일치하는 수치는 아니다. NaN/Inf, 비어 있는 mask, 중복 prompt ID는 유효한 통계로 게시하지 않는다.

### Prompt 균형과 범주 선택성

```text
μ_p = Σ_{t∈T_p} a_pt / N_p
μ_prompt = (1/P) Σ_p μ_p
σ_prompt = sqrt((1/P) Σ_p (μ_p - μ_prompt)^2)
μ_c = (1/P_c) Σ_{p∈c} μ_p
g_c = μ_c - (1/(C-1)) Σ_{d≠c} μ_d
c* = argmax_c g_c
selectivity_gap = g_c*
selectivity_z = g_c* / max(σ_prompt, 1e-8)
```

각 prompt의 길이와 관계없이 `μ_c`에서 같은 무게를 준다. 비교군은 다른 범주의 평균을 다시 **범주별 같은 무게**로 합친 것이다. `cross_prompt_std`는 위 `σ_prompt`다. 한 범주뿐이면 gap은 0으로 처리하고 후보를 뽑지 않는다. 후보 순위에서는 `σ_prompt < 1e-6`인 좌표를 제외하고, z 내림차순·layer/neuron 오름차순으로 정렬한다.

이 `selectivity_z`는 **설명용 표준화 분리도**다. 표준정규 검정통계량, p-value, 신뢰도 확률, 인과 효과의 크기가 아니다. 여러 범주 중 가장 큰 gap을 선택한 값이므로 discovery에서 낙관적으로 보일 수 있다. 낮은 방향의 활성이나 다의적 기능은 이 양의 최고범주 순위에 충분히 나타나지 않는다.

`survey.py`는 계층별 상위 8개 후보를 모아 재정렬하고 요약과 공개 catalog의 보존 개수를 제한한다. 이 편집은 전체 뉴런에 대한 다중비교 보정이 아니다. 전 계층에서 수십만 좌표를 검색한 뒤 가장 높은 점수의 좌표를 읽는 행위 자체가 선택 편향을 만들므로 후속 독립 검증이 필수다.

### Quantile과 evidence buffer

q05/q50/q95는 모든 token을 저장해 정확히 계산한 quantile이 아니다. `(seed, prompt_id, 원래 flat position)`의 deterministic hash priority가 가장 작은 **최대 64 token rows**를 보존하고 그 표본에 선형 보간 quantile을 적용한다. Hash를 균등 priority로 취급하는 bounded token reservoir다. 모든 neuron은 같은 sampled positions를 공유하며, 같은 corpus/seed면 계층 간에도 같은 position 표본을 갖는다. 이는 독립적인 neuron별 sampling이 아니다.

64개 중 q05나 q95는 소수의 관측에 의존하므로 꼬리 분포 추정이 특히 불안정하다. 긴 prompt는 이 token reservoir에 들어갈 기회가 많다. 따라서 **quantile은 token 가중**, category selectivity는 prompt 균형이라는 차이를 유지한다. 표본 ID·실제 표본 수와 추정 방법을 남기고, 이 작은 표본에서 매끈한 density나 정밀한 tail probability를 주장하지 않는다.

Top/low context buffer는 각 prompt에서 해당 neuron의 최대/최소 하나를 먼저 뽑고, 이를 prompt 사이에서 정렬해 기본 상위/하위 3개를 남긴다. **전체 token에서 순위가 가장 높은 3개를 그대로 저장하는 방식이 아니다.** 서로 다른 prompt의 극단을 보여주는 장치이며, 문맥·범주·token 위치·activation을 함께 보존한다. 최대값은 corpus 크기와 prompt 길이에 따라 커질 수 있어 보편적인 자연 활성 상한으로 쓰지 않는다.

### 계층 요약 그래프의 읽기

`std_p10/p50/p90`은 계층 안 **neuron별 token 표준편차의 분위수**이며 token 활성분포의 분위수가 아니다. `mean_abs_activation`이라는 현재 필드는 실제로 `mean_n(|μ_token,n|)`을 담는다. 전체 token의 평균 절댓값 `E|a|`와 구별해야 한다. `category_deviation[c]`는 `mean_n(|(μ_c,n - μ_prompt,n) / max(σ_token,n,1e-6)|)`다. 이 값은 계층별 관찰 패턴을 요약하며 특정 의미의 표현 강도나 인과 중요도 점수가 아니다.

## 4. Corpus와 분할

`corpus.py`의 직접 작성한 무해한 convenience corpus는 15개 범주 × 10개 원본, 총 150 prompt다. 각 범주에서 고정 index 0–3은 discovery, 4–6은 validation, 7–9는 test다. 기본 분할은 각각 60/45/45개다. 외부 benchmark 사본을 사용하지 않는다. 원본마다 `source_id`를 부여하고 향후 paraphrase·near-miss는 원본과 동일 split에 묶어야 한다. ID 중복·정규화 텍스트 중복·source group의 split 교차를 검사하지만 의미적으로 유사한 질문의 교차 누수를 전부 자동 판별하는 검사는 아니다.

기본 pilot는 discovery에서 범주별 첫 prompt 하나씩 15개와 계층 `[0,15,31,47]`을 사용한다. 범주당 하나라 category 내 분산을 추정할 수 없으며 후보의 의미적 안정성을 평가하는 표본으로 부족하다. 기본 survey는 discovery 60개, 모든 계층을 사용한다. 계층 전체를 계측하는 것이 언어·문맥·행동 공간 전체를 조사한다는 뜻은 아니다.

범주별 token 길이, 영어/한국어/일본어 문자, task 지시·format·표면 단어가 다르다. 높은 분리도가 의미가 아닌 길이·script·문법·instruction 차이에서 나올 수 있다. `long_context` 항목은 짧은 정보 회수 prompt이며 장문 benchmark가 아니다. `benign_refusal` 범주의 활성 차이는 실제 over-refusal 발생률 측정이 아니다. 정확도·말투·persona·안전 행동의 이름은 독립적으로 정의한 행동 지표 없이 neuron의 검증된 기능명으로 사용하지 않는다.

## 5. 개입 수식과 norm 대조군

선택한 position에서 원래 activation vector를 `a`, 단위 좌표를 `e_n`, 강도를 `α`, 배율을 `m`, replacement를 `r`이라고 쓴다.

| 개입 | 부동소수점 적용 전의 정의 |
|---|---|
| Additive | `a' = a + α e_n` |
| Negative additive | 같은 공식에서 `α < 0` |
| Multiplicative | `a'_n = m a_n` |
| Zero ablation | `a'_n = 0` |
| Replacement / restoration | `a'_n = r` |
| Random neuron | seed로 좌표 `j`를 뽑아 `a' = a + α e_j` |
| Random direction | seed로 정규벡터 `v`를 뽑아 `a' = a + α v/||v||₂` |

나머지 좌표는 보존하며 변경 tensor는 복사해 반환한다. Capture callback에는 **개입 전** detached clone을 주며 read-only hook은 입력을 바꾸지 않는다. 한 모듈에 lab hook을 중복 등록하지 못하게 하고 context 종료·예외에서 자기 handle을 제거한다. 기존 타 소유 hook은 자동 삭제하지 않는다.

BF16에서 실제 `a' - a`는 위 실수 공식의 delta와 다를 수 있다. 작은 α는 반올림에 묻히고 dense direction의 각 작은 성분은 서로 다르게 양자화될 수 있다. 따라서 실제 변경 norm, natural activation 대비 변화, 출력 변화량을 함께 보고한다. Add=0 또는 multiply=1의 정확한 no-op도 확인 대상이다.

**Activation-space norm 일치와 down-projection-space norm 일치는 다른 대조 조건이다.** 순수 coordinate add의 activation norm은 명목상 `|α|`, projection 직후 norm은 `|α| ||D_l[:,n]||₂`다. activation norm을 같은 `|α|`로 한 dense 무작위 방향의 projection norm `||D_l δ_random||₂`는 보통 다르다. `hooks.py`의 기본 random-direction API는 전자만 맞추며 후자를 자동 보장하지 않는다. 또한 random neuron의 column norm과 natural variance가 target과 다를 수 있고, 현재 seed 선택이 target 좌표와 우연히 같을 수도 있으므로 실제 선택 좌표를 남겨야 한다.

후속 matched-control 실험은 **무엇의 norm을 맞췄는지** 먼저 고정한다. 가능한 비교는 ① 동일 activation delta norm, ② 동일 raw-down-projection delta norm, ③ 자연 변동 대비 동일 표준화 강도다. 모든 조건을 동시에 만족한 것처럼 표현하지 않는다. 단일 뉴런의 residual contribution과 비교할 때는 같은 layer·position·dose 정책을 유지하고 `D_l δ`를 측정하거나 선형 계산해 맞춘다. Gemma의 post-MLP norm과 residual layer scaling 이후에는 `D_l δ`와 다른 비선형 변화가 생기므로 projection norm만으로 최종 residual norm까지 같다고 주장하지 않는다.

여러 seed의 무작위 neuron·direction, 작은 강도부터의 dose-response, 자연 q범위/관측 extrema, ablation, restoration, 별도 일반 과제의 품질 보존을 함께 비교한다. 이는 검증 설계이며, 어떤 대조군이 실제 실행됐고 어떤 norm이 일치했는지는 steering run metadata와 결과가 결정한다.

## 6. Smoke test가 보증하는 범위

`smoke.py`는 3개의 무해한 prompt에서 기본 생성, 전 계층 capture shape·유한값, read-only logits의 exact equality, hook 제거 후 equality, greedy 반복 생성 token equality, BF16 add 공식, restoration equality, ablation logits의 유한값을 검사하도록 작성되어 있다. 이 검사는 기계적 정확성과 재현성에 대한 smoke 절차다. 특정 neuron의 의미 역할이나 전체 행동 평가를 대신하지 않는다.

현재 smoke restoration은 같은 입력에서 얻은 자연 baseline 좌표를 별도의 forward에 재주입한다. 그 시점에는 선행 add/ablation hook이 제거돼 있으므로 **변경된 실행을 중간에서 rescue하는 causal mediation 실험이 아니다.** 일치하면 matched replacement와 기술적 baseline 복구가 작동한다는 제한된 증거다. 의미적 restoration을 주장하려면 다른 원인의 perturbation을 유지한 상태에서 target activation만 복구하고, 정해 둔 행동 지표가 matched/random 복구 대비 선택적으로 회복되는지 독립적으로 검사해야 한다.

`torch.equal`은 동일 환경·동일 입력에서의 엄격한 통과 기준이다. 모델·library revision, GPU, kernel, precision이나 batch 방식이 달라져도 항상 동일한 수치가 나온다는 보증은 아니다. smoke FAIL이면 worker는 후속 수집과 개입을 허용하지 않으며, 불일치 원인을 해결하기 전 확대하지 않는다.

## 7. 가설·독립 검증·불확실성

단계별 증거를 구별한다.

- **관찰:** user-content 활성값의 범주 분리, 극단 문맥, prompt 간 변동. 후보 선정 근거이며 인과 결과가 아니다.
- **출력에 대한 인과 효과:** 동일 입력에서 통제된 activation 개입이 logits·생성을 바꿈. 모델 내부 계산에 영향을 준다는 증거지만 그 자체로 기능을 명명하지 않는다.
- **행동에 대한 선택적 인과 효과:** 사전 정의한 행동 평가가 target에서 바뀌고 matched/random control·일반 품질 손상·대체 설명으로 충분히 설명되지 않음. 별도 validation이 필요하다.
- **반복 확인:** 다른 prompt family, 사전 동결된 test, 독립 설계·seed/환경에서 방향과 선택성이 유지됨. 단일 discovery 성공이나 설명 모델의 확신으로 대체할 수 없다.

가설은 discovery evidence만 읽고 경쟁 설명과 함께 동결한다. 이후 양성·음성·near-miss·paraphrase·OOD 예제를 설계하고 범주당 충분한 **독립 source group**으로 평가한다. validation 결과를 보고 가설을 수정했다면 수정 이력과 새 가설 ID를 남기고, 같은 validation을 다시 독립 test라고 부르지 않는다. test를 선택·튜닝에 사용한 뒤에는 새로운 holdout 없이는 확인적 주장으로 승격하지 않는다.

수십만 neuron과 여러 범주·dose·행동 지표 중 최고치를 고르는 탐색은 다중선택 문제를 가진다. 확인적 비교는 후보·주 지표·대조군·강도·방향을 사전에 고정하고, 충분한 표본으로 prompt/source-group 단위 paired 효과와 불확실성을 평가해야 한다. token을 독립 표본으로 간주해 신뢰구간을 과도하게 좁히지 않는다. bootstrap을 쓴다면 독립 prompt/source를 resample하며, 후보 다수의 확인적 검정을 할 때는 family와 보정 방식을 명시한다. 작은 표본·희소한 범주에서는 수치의 한계와 부호 불안정을 함께 기록한다. 현재 discovery z만으로 통계적 유의성을 선언하지 않는다.

현재 모델 생성은 greedy(`do_sample=False`)다. 같은 prompt에 seed만 바꿔 같은 출력을 반복해도 독립적인 생성 표본 수가 늘어나지 않는다. 모델 seed는 재현성 설정이고, reservoir/control seed는 선택·방향의 반복 안정성을 측정하는 변수다. 같은 의미의 프롬프트를 여러 seed로 실행하는 것은 새로운 prompt family 검증을 대신하지 않는다. `deterministic_algorithms(..., warn_only=True)`이므로 library 경고까지 확인해야 하며 전 환경의 완전 결정성을 보장하지 않는다.

Catalog 상태 `candidate`, `correlated`, `causally_supported`, `replicated`, `ambiguous`, `falsified`, `unknown`은 구체적 evidence와 run ID로 뒷받침한다. logits가 달라졌다는 이유만으로 행동명과 `causally_supported`를 부여하지 않는다. 기능 가설이 지지되지 않은 경우에도 측정된 개입 효과 자체는 별도로 기록할 수 있다.

## 8. 저장·예산·재현

Run ID, 고정 설정, Python/library/CUDA/GPU 정보, 핵심 source/config SHA256, 시작·종료 시각(KST), 시간·peak GPU memory, PASS/FAIL과 artifact hash를 기록한다. 로컬 SQLite는 run 상태를 추적하고, neuron 통계는 Parquet, 제한한 context/category 배열은 압축 NPZ, 공개 요약은 JSON으로 저장한다. 원본 가중치·cache·전체 activation·전체 생성 출력은 공개 Git에 넣지 않는다.

기본 예산은 smoke 최대 600초/3 prompt/16 생성 token, pilot 900초/15 prompt, survey 1800초/60 prompt다. 런타임은 free VRAM 34GB 미만이면 시작을 거부하고, stage 사이·prompt 사이에서 디스크 여유와 예약 GPU memory·run 용량을 점검한다. 이 점검은 각 forward 내부를 실시간 선점하는 장치가 아니므로 한 호출 동안의 순간 초과나 timeout을 완벽히 막지는 않는다. 다른 프로세스는 종료하지 않는다.

완료한 stage는 보존한다. 현재 수집기의 중간 progress 파일은 표본 처리 현황이며 accumulator 전체의 재개 checkpoint가 아니다. 중단된 stage는 동결 corpus로 처음부터 새 run을 실행해 prompt를 중복 누적하지 않는 방식이다. 실제로 저장된 source hash·의존성 버전·모델 revision·corpus digest·token 범위·stage 설정을 함께 보존해야 실험을 재현할 수 있다.

## 9. 첫 steering 주기의 사전 명시

Steering의 next-token 개입 위치는 마지막 chat-template wrapper token이다. Discovery의 사용자 본문 평균 activation과 token 범위가 다르므로 본문 선택성을 응답 생성 기능으로 바로 확장하지 않는다. 자연 마지막-token activation과 개입 후 값이 discovery 관측 범위를 벗어나는지도 별도로 기록한다. 랜덤 대조군은 |dose|=1 SD에서만 nominal activation/projection norm을 맞춘다. ±0.5/±2 SD sweep을 1 SD 대조군과 직접 크기 비교해 선택성을 주장하지 않는다. greedy 생성은 고정된 소수 prompt에 한정되며 restoration은 정렬된 next-token logits에서만 검증한다.
