# R10/R10b 실제 계측 수리 결과

A/B/C/E의 원래 미평가 상태를 과학적 반증으로 해석하지 않고, 실제 두 채팅과 계측을 수리해 새 데이터로 실행했습니다. 원래 R10 실패와 별도 R10b 결과를 분리합니다.

모델 revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. FP32_upcast_of_same_stored_BF16_values. 이번 수치는 실제12B 모델에서 생성한 응답의 집계이며, CPU 합성 테스트17개나 토큰 replay는 추가 모델 표본으로 세지 않습니다.

## 원래 시도와 별도 수리

R10 A 보정은 완결42응답에서 VALID30/TOKEN_CAP12였습니다. 원래 등록에 없던 추가 공학적 중단을 사용했고, 당시 모든 P0는 미실행입니다. 미완결1응답의75forward 진입은 INTERRUPTED 비용이며 TOKEN_CAP 응답으로 세지 않았습니다. 원래 실패와 소스는 보존했습니다.

R10b에는 간결 출력계약과 A의 직접 공지 anchor 명료화를 등록했습니다. R9와 R10의 사용·예약 세계/입력을 모두 배제하고,256토큰·retry0·기존 terminal parser·문자/선택지 교차를 유지했습니다. 질문별 futility는 새 출력 전에 등록했습니다. 후속 추가 문구·cap 수리는 하지 않았습니다.

추정 대상: Final selected option under this direct-output instruction and 256-token budget; not equivalent to R9/R10 behavior and no randomized causal estimate of wording improvement. 새 세계와 출력 계약이 함께 바뀌어 R10 대비 개선의 원인을 분리할 수 없습니다. 기초 정답 자격 실패, 최종답 판독 실패, 실제 main의 넓은 구간은 서로 다른 상태입니다.

## 실행 범위와 결과

| 질문 | 보정 완결 응답 | 보정 terminal 유효 | 보정 한 줄 준수 | 새 P0 응답(기초+main) | P0 실제 forward | 최종 상태 |
|---|---:|---:|---:|---:|---:|---|
| A | 78 | 76/78 | 0/78 | 0 (0+0) | 0 | NOT_EVALUATED; CALIBRATION_NOT_READY |
| B | 160 | 160/160 | 33/160 | 864 (288+576) | 88339 | Negative; P0_MEASURED |
| C | 4 | 0/4 | 0/4 | 0 (0+0) | 0 | NOT_EVALUATED; CALIBRATION_NOT_READY |
| E | 136 | 132/136 | 0/136 | 0 (0+0) | 0 | NOT_EVALUATED; CALIBRATION_NOT_READY |

보정 중단은 최종 예정 분모에서 기준 달성이 불가능하다는 등록 규칙에 따릅니다. 이 상한은 고정 문항 묶음의 산술적 판정이며 모집단 판독률이95% 미만이라는 통계 검정이 아닙니다. 중단 시까지의 관측 분모와 최종 예정 분모를 혼동하지 않습니다. Futility로 생성하지 않은 나머지는 NOT_EVALUATED이며 TOKEN_CAP 응답으로 세지 않습니다. Sentinel은 판독 자격만 검사하며 과제 정답으로 main 진입을 고르지 않습니다.

판독된 기본 답에서 알려진 오답0은 종료된 응답에 조건을 건 기술통계입니다. 절단된 응답까지 정답이라고 가정하거나 전체 기초 능력이 확립됐다고 해석하지 않습니다. 한 줄 준수와 terminal 판독도 분리합니다. 이 실험은 간결 출력 지시 아래의 최종 선택을 측정하며, 실제 응답이 모두 설명 없는 직접답이었다고 주장하지 않습니다.

### A: 관측 정보와 전략의 구현 가능성

보정 gate: **CALIBRATION_NOT_READY**. 기초 78/128행, 복합 sentinel 0/64행을 실제 생성했습니다.

| 기초 기능 | 관측 수 | 확인 정답 | 알려진 오답 | 미판독 | 가능한 최종 판독률 상한 | 가능한 최종 확인 정답률 상한 |
|---|---:|---:|---:|---:|---:|---:|
| code | 20 | 20 | 0 | 0 | 100.00% | 100.00% |
| order | 20 | 20 | 0 | 0 | 100.00% | 100.00% |
| payoff | 20 | 20 | 0 | 0 | 100.00% | 100.00% |
| signal_rule | 18 | 16 | 0 | 2 | 93.75% | 93.75% |

등록 futility 기능: signal_rule. 해당 main 가설은 NOT_EVALUATED이며 Negative가 아닙니다.

### B: 총합 보존 보상 재분배와 경로 선택

보정 gate: **CALIBRATION_READY**. 기초 96/96행, 복합 sentinel 64/64행을 실제 생성했습니다.

| 기초 기능 | 관측 수 | 확인 정답 | 알려진 오답 | 미판독 | 가능한 최종 판독률 상한 | 가능한 최종 확인 정답률 상한 |
|---|---:|---:|---:|---:|---:|---:|
| path_sum | 64 | 64 | 0 | 0 | 100.00% | 100.00% |
| total_comparison | 32 | 32 | 0 | 0 | 100.00% | 100.00% |

P0 gate **P0_MEASURED**, Main 생성 576행.
실제 main의 가설 분류는 **Negative**입니다.
Main terminal 판독 100.00%; 확인 정답 576/576, 알려진 오답 0, 미판독 0. 정확도 최악 경계 [100.00%, 100.00%].

| 등록 대비 | 식별 경계 | 탐색적 outer95 구간 | 독립 세계 |
|---|---|---|---:|
| total0_reverse0 | [0.0000, 0.0000] | [0.0000, 0.0000] | 12 |
| total0_reverse1 | [0.0000, 0.0000] | [0.0000, 0.0000] | 12 |
| total1_reverse0 | [0.0000, 0.0000] | [0.0000, 0.0000] | 12 |
| total1_reverse1 | [0.0000, 0.0000] | [0.0000, 0.0000] | 12 |
| primary_code0 | [0.0000, 0.0000] | [0.0000, 0.0000] | 12 |
| primary_code1 | [0.0000, 0.0000] | [0.0000, 0.0000] | 12 |

### C: 자료 수집 절차와 posterior 판단

보정 gate: **CALIBRATION_NOT_READY**. 기초 4/128행, 복합 sentinel 0/64행을 실제 생성했습니다.

| 기초 기능 | 관측 수 | 확인 정답 | 알려진 오답 | 미판독 | 가능한 최종 판독률 상한 | 가능한 최종 확인 정답률 상한 |
|---|---:|---:|---:|---:|---:|---:|
| Bayes_weights | 4 | 0 | 0 | 4 | 93.75% | 93.75% |

등록 futility 기능: Bayes_weights. 해당 main 가설은 NOT_EVALUATED이며 Negative가 아닙니다.

### E: 국소 자원 집합과 전역 배정 가능성

보정 gate: **CALIBRATION_NOT_READY**. 기초 136/160행, 복합 sentinel 0/64행을 실제 생성했습니다.

| 기초 기능 | 관측 수 | 확인 정답 | 알려진 오답 | 미판독 | 가능한 최종 판독률 상한 | 가능한 최종 확인 정답률 상한 |
|---|---:|---:|---:|---:|---:|---:|
| edge | 52 | 52 | 0 | 0 | 100.00% | 100.00% |
| reuse_rule | 32 | 32 | 0 | 0 | 100.00% | 100.00% |
| union_count | 52 | 48 | 0 | 4 | 93.75% | 93.75% |

등록 futility 기능: union_count. 해당 main 가설은 NOT_EVALUATED이며 Negative가 아닙니다.

## 비용·검증·해석 한계

R10b 기술50 + 보정44332 + P088339 = 132721 actual forward. 원래 R10의7792를 포함해 총140513회입니다. 한 응답은 여러 forward를 사용합니다.

R10b 완결 응답1242개, terminal 상태 {"VALID": 1232, "TOKEN_CAP": 10}, 한 줄 준수151개. 누적 실측 GPU시간 6624.007초. 최대 할당 메모리 48.402GB.

실제 BF16/FP32 훅 검증과 제한된 KV-cache/full-prefix spot check를 통과했습니다. CPU replay는 원시 token ID·최종답·입력 해시·실제 카운터·최초 futility 시점·동결된 통계를 재계산합니다. 모델 logit을 다시 실행한 독립 반복은 아닙니다.

정상 판독은 의미 정답이나 가설 양성이 아닙니다. 원래 가설의 특정 효과가 Negative여도 해당 유한 생성기·모델·출력 계약에 한정하며, 전역적 능력이나 내부 인과 회로의 부재를 뜻하지 않습니다. 기초 실패는 해당 측정 자격 실패이며 가설 반증이 아닙니다. 8–12세계 bootstrap은 탐색적이며 여러 대비의 구간은 다중검정 보정이나 동시95% 구간이 아닙니다. P1/P2와 봉인 확인은 미평가입니다.

[집계 JSON](../logs/summary/round10b_aggregate.json) · [원시 token CPU replay](../logs/summary/round10b_replay.json) · [사전등록](ROUND10B_REPAIR_PREREG_KO.md) · [원래 R10 실패](../logs/summary/round10_attempt1_abort.json)

## B의 퇴화 bootstrap에 대한 사후 민감도 점검

등록된 B Negative와 원래 primary 구간은 변경하지 않았습니다. 다만 12세계에서 효과가 모두0이라 bootstrap[0,0]이 된 것을 모집단 동등성 증명으로 해석하면 안 됩니다. 예시 IID 모집단에서10%세계만 대비+1이고 나머지는0이면 모집단 평균은10pp이지만,12개 모두0을 관측할 확률은 0.2824입니다.

추가 참고 모형에서 세계별 어떤 위반 사건도0/12라면 그 사건 확률의 단측95% 상한은 0.2209입니다. 이것은 실제 유한 corpus의 IID를 입증한 CI가 아닌 가정을 드러낸 사후 민감도입니다. 따라서 **관측12세계의 등록 규칙상 Negative와 모집단10pp 가설의 미확립을 구분**합니다. 새 모델 호출이나 후보 변경은 없습니다.

[사후 민감도 JSON](../logs/summary/round10b_uncertainty_sensitivity.json)

## 실제 결과 공동 재검토와 다음 실험

재검토 상태: **BOTH_FINAL_AGGREGATE_REVIEW_AND_CROSS_COMPLETE**. 전체 원시 생성문 없이 집계·효과·불확실성·한계를 기존 두 채팅에 전달했습니다.

- B Negative is the frozen sample-level decision. All576main responses in12worlds are correct and read out, so zero allocation contrast is a consequence of observed accuracy ceiling, not independent additional evidence for an internal algorithm.
- Empirical[0,0] identification interval is exact for the observed choices; bootstrap[0,0] is degenerate and does not bound a population10pp effect. Codex additionally computed explicit post-hoc IID-reference sensitivity: zero-event95upper0.220922, and a10percent positive-world mixture can produce12zeros with probability0.282430. No registered statistic/classification was overwritten.
- Choice invariance does not prove latent-value/margin/effort invariance. Numerical range, single model/FP32-upcast condition and concise-output instruction limit generalization. Actual prose remains common; do not claim pure direct answers.
- A/C/E remain NOT_EVALUATED. Known-wrong0 in parsed basic answers does not score censored answers as correct. C has zero confirmed answers. Omitted rows after futility are not token-cap responses.
- Old6x6 future-candidate example changed an obvious duplicated-singleton cue along with matching feasibility. Reject global-feasibility-only causal attribution from that example.
- Follow GPT control proposal: cross exclusive versus unlimited reuse with the SAME structural graph pairs, compare true-switch delay minus matched-sham delay, and separate correct-final/wrong-final/no-delivery. Generic EOS shortening is not a causal bottleneck.
- Codex implemented a CPU-only8x8 prototype after this feedback, with no duplicate primary rows, minimum3-task obstruction, identical row/column degrees, typed pairwise overlap histograms, fixed local rows/union and matched2x2-degree/4-edge switches. Exhaustive permutation and subset oracles agree. This is one structural family, not a corpus or model result; higher-order/component confounds and text/token matching remain. The peers have not reviewed this subsequent prototype implementation.
- Current R10b prompt/cap/parser/stop rules and all old R9/R10 data stay frozen. No renamed R10c, new GPU, sparse-neuron discovery, confirmation or paper Go. Future candidate is a different scientific endpoint, not a claim to repair the unmeasured A/C/E main.

| 추천순 | 다음 실험 | 목적·전제 |
|---:|---|---|
| 1 | 묻지 않은 전역 제약이 국소 답 전달에 미치는 선택적 영향: CPU 대조 corpus·선행 차별성 검증 | 1순위 CPU만. 단일8x8 존재 예시는 검증했으나 비동형 여러 구조·연결 성분·정확한 token 길이·중복/저차 단서와 sham을 더 맞춰야 한다. 독점/재사용의 switch-minus-sham 상호작용을 별도 새 연구의 사전 primary로 설계한다. 기존A/C/E main은 여전히 미평가. 대조쌍 또는 선행 차별성을 확보 못 하면 폐기; GPU Hold. |

### 다음 명제를 위한 CPU 구성 예시: 모델 결과 아님

명제 후보는 **국소 정답과 관련 사실을 고정한 상태에서, 묻지 않은 전역 제약이 국소 답 전달을 선택적으로 지연시키는가**입니다. 기존 A/C/E의 의미적 main을 평가했다고 주장하지 않으며, 형식·cap 재수리를 이름만 바꿔 재개하지 않습니다.

새8×8 조합 예시 한 구조군에서 exhaustive 배정 검사 80642회와 부분집합 검사를 교차해 구성 가능성을 확인했습니다. 추가 모델 호출0, CPU 0.0276초입니다.

| 예시 | 독점 배정 가능 | 무제한 재사용 가능 | 중복 행 | 최소 실패 업무 집합 크기 |
|---|---|---|---:|---:|
| base_infeasible | False | True | 0 | 3 |
| switch_feasible | True | True | 0 | 없음 |
| sham_infeasible | False | True | 0 | 3 |
| duplicate_but_feasible_auxiliary | True | True | 1 | 없음 |

기본·실제전환·sham은 행/열 degree, 고정된 질의 행·union, degree로 층화한 행/열 두 집합씩의 교집합 histogram이 같습니다. 실제전환과 sham은 각각2행·2열의4edge 항목을 바꾸며, 변경 endpoint의 degree도2/2로 같습니다. 기본·sham은 1개·2개 업무 조건은 모두 만족하지만3개 업무가 자원2개에만 연결되어 전체 배정이 불가능합니다. 중복 행 없이도 이런 장애가 생깁니다. 보조 예시는 중복된 다중 자원 행이 있어도 전체 배정이 가능함을 확인합니다.

**아직 한 구조군뿐입니다.** 이름만 바꾼 예시는 독립 표본이 아닙니다. 연결 성분 크기 등 더 높은 구조는 아직 맞추지 않았고, 실제 문장/token 길이의 동등성도 검증하지 않았습니다. 보조 중복 예시도 모든 primary 구조 통계를 맞춘 것은 아닙니다. 따라서 전역 가능성만의 인과 효과를 분리했다고 주장하지 않습니다.

다음 CPU 단계는 비동형 여러 구조·연결 성분·표면 단서·정확한 token 길이를 맞춘 corpus와 sham을 확보하고 원문 선행과의 차별성을 검증하는 것입니다. 확보 후에도 새 사전등록이 필요하며, 예측은 **전환−sham의 답 전달 지연이 독점 조건에서만 커지고 재사용 조건에서는 사라지는가**입니다. 정답 final·오답 final·미전달을 분리하며, 완결답만의 평균 길이를 비교하지 않습니다. Sham/재사용에서도 같은 효과이거나 국소 판독이 안 되면 종료하고 회로 patching으로 넘어가지 않습니다.

표현 길이·진행도 조절 자체는 이미 [Overclocking](https://arxiv.org/html/2506.07240v1), 모순으로 계산 비용을 늘리는 것은 [CODE](https://arxiv.org/html/2601.13112v1)와 인접합니다. 해가 없는 배정 문제와 서로 모순되는 사실은 다르지만, 그 차이만으로 신규성이 확립되지는 않습니다. 현재 GPU Go·논문 Go는0입니다.

[CPU 검증 JSON](../logs/summary/round10b_next_cpu_design.json) · [예산](../configs/round10b_next_cpu_design.yaml) · [구성·독립 oracle 코드](../scripts/check_round10b_next_cpu_design.py)
