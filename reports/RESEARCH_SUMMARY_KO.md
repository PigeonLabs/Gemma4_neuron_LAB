# Gemma 4 Neuron Lab — 연구 종합 보고서

실제 실행한 **첫 번째 유한한 연구 주기**의 기록이다. 구현 여부·실험 실행 PASS·과학적 가설 상태를 구별한다. 동결한 text-only 실험과 다음 연구 후보를 정리한다.

**후속 Round 2 E/F도 완료했다.** 10,816개 조건 행·11,568회 실제 BF16 forward. 5번은 ΔR²=0.20433의 control-probe-assisted 예측 개선, 6번은 손상 gate 실패로 NOT_IDENTIFIABLE다. 원 연구의 수치를 섞지 않고 [새 연구의 전체 결과](ROUND2_EF_RESULTS_KO.md)에 설계·통계·한계를 별도로 기록했다.

실제 BF16 모델에서 **48계층 × 15,360 = 737,280개 MLP 중간 좌표**를 확인했다. Discovery 60 prompts·계층당 1,022 user-content tokens를 관측했다. 모델 revision은 `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`이다.

종합 상태: **PASS**. First bounded, Codex-guided text-only research cycle; execution complete, semantic neuron mechanisms unresolved

초기 CP4 증거: [INITIAL_FINDINGS_KO.md](../reports/INITIAL_FINDINGS_KO.md).

## 1. 실제 구현한 기능과 검증 범위

| 기능 | 구현 산출물 | 실제 검증 |
|---|---|---|
| BF16 native adapter·capture·개입·hook 제거 | [구현](../src/neuron_lab/hooks.py) | PASS |
| GPU streaming 통계·Parquet/JSON/SQLite 기록 | [구현](../src/neuron_lab/statistics.py) | PASS |
| 강도 sweep·seeded norm controls·restoration | [구현](../src/neuron_lab/steering.py) | PASS |
| 동결 가설·반례·독립 분할 재분석 | [구현](../scripts/analyze_results.py) | PASS |
| 공개 neuron catalog | [구현](../catalog/public/neurons.json) | 4개 측정 기록 |
| 로컬 Explorer·Live Steering | [구현](../explorer/app.py) | [실행 QA 기록](../docs/EXPLORER_QA.md) |

Codex는 discovery 증거로 경쟁 가설과 반례를 작성했다. Hook API의 multiplicative/replacement 지원과 실제 모델에서 평가한 조건은 같은 범위가 아니다. 이미지·음성·외부 계측 engine·대표 행동 정확도는 별도 미평가 범위다.

## 2. 실행된 실험과 Run 기록

| 단계 | 상태 | Run ID | 원본 corpus / 추가 사례 | 조건–prompt rows | 시간(s) | Peak reserved(GiB) |
|---|---|---|---:|---:|---:|---:|
| smoke | PASS | `20261029T135946_smoke` | 3 / 해당 없음 | 해당 없음 | 5.9091 | 22.338 |
| pilot | PASS | `20261029T131907_pilot` | 15 / 해당 없음 | 해당 없음 | 2.2495 | 22.369 |
| survey | PASS | `20261029T132008_survey` | 60 / 해당 없음 | 해당 없음 | 17.268 | 22.719 |
| steering | PASS | `20261029T132730_steering` | 15 / 0 | 630 | 97.03 | 22.812 |
| validation | PASS | `20261029T133451_validation` | 45 / 40 | 3,740 | 381.92 | 23.258 |
| expanded | PASS | `20261029T135014_expanded` | 45 / 0 | 3,060 | 308.48 | 23.258 |
| utility | PASS | `20261029T135550_utility` | 16 / 해당 없음 | 생성 272 | 39.952 | 23.221 |

Smoke: 28/28 checks; read-only 최대 logit 차이 0. Activation의 표본 단위는 사용자 본문 token·prompt이고, 조건 row나 random seed를 독립 prompt로 세지 않는다.

## 3. 실패한 실행과 실패 이유

읽은 stage 요약에 `FAIL` 실행은 기록되어 있지 않다. 누락된 단계는 NOT_EVALUATED이며 성공으로 간주하지 않는다. 과학적 음성 결과·반례는 8–9절의 등록된 종합 판단과 구별한다.

## 4. 후보 뉴런과 경쟁 가설

| ID | Layer / neuron (0-based) | 동결 가설 | 경쟁 설명 | 카탈로그 상태 |
|---|---|---|---|---|
| H001 | 15 / 9,637 | 번역 과제 문맥에 반응하는 좌표일 수 있다. | 번역 동사·언어 이름의 표면 토큰; 외국어 문자와 문장 길이 | ambiguous |
| H002 | 22 / 7,859 | 주어진 정보로 답을 결정할 수 없는 문맥에 반응하는 좌표일 수 있다. | uncertain/know 등 표면 어휘; 질문·부정문 형태 | ambiguous |
| H003 | 26 / 719 | 후기 계층의 번역 과제 문맥 좌표일 수 있다. | 언어 이름 자체의 표상; 번역 요청의 어휘·구문 | ambiguous |
| H004 | 35 / 990 | 프로그래밍 과제 전반에 반응하는 좌표일 수 있다. | Python이라는 특정 어휘·동물 이름; 코드 문법·표면 언어 이름 | ambiguous |

가설 동결 KST: 2026-10-29T13:24:32.462142+09:00. 선택은 discovery에 한정했다. 관측 분리도는 범주 어휘·문자·길이·구문과 혼동될 수 있으며, 의미 기능의 인과 라벨이 아니다.

## 5. 정량 Activation 및 Steering 결과

Survey: 60 prompts / 15 categories / 계층당 1,022 tokens / 737,280좌표. Quantile 방법: uniform hash-priority reservoir, 64 user-token positions per layer. 범주 평균은 prompt 균형, token moments는 token 가중이다. Quantile의 작은 reservoir는 정밀한 꼬리 추정을 보장하지 않는다.

Baseline 대비 next-token KL(P_base || P_changed), nats. 평균 [95% percentile bootstrap CI]; random seed는 prompt 안에서 평균한 후 독립 corpus prompt를 resample했다. 반례 family는 이 CI에 섞지 않았다.

| 단계 / 후보 | 조건 | n prompts / rows | KL 평균 [95% CI] | 첫 token top-1 변화 비율 |
|---|---|---:|---|---:|
| 표 범위 | expanded · 가장 최근 완료한 분할 | — | — | — |
| expanded / H001 | −1 SD | 45 / 45 | 0.00055396 [0.00018539, 0.0010453] | 0.022222 |
| expanded / H001 | +1 SD | 45 / 45 | 0.0004433 [0.00015767, 0.00083447] | 0.022222 |
| expanded / H001 | Zero | 45 / 45 | 0.00066403 [0.00030306, 0.0011339] | 0 |
| expanded / H001 | Random neuron · projected | 45 / 135 | 0.00054369 [0.00024044, 0.0009432] | 0.014815 |
| expanded / H001 | Random direction · projected | 45 / 135 | 0.00040922 [0.00022121, 0.00062658] | 0.0074074 |
| expanded / H002 | −1 SD | 45 / 45 | 0.00039996 [0.00015385, 0.00070759] | 0.022222 |
| expanded / H002 | +1 SD | 45 / 45 | 0.00041985 [0.00016909, 0.00075279] | 0.022222 |
| expanded / H002 | Zero | 45 / 45 | 0.00036354 [0.00018554, 0.00058792] | 0.022222 |
| expanded / H002 | Random neuron · projected | 45 / 135 | 0.00043033 [0.00019938, 0.00072755] | 0.014815 |
| expanded / H002 | Random direction · projected | 45 / 135 | 0.00046293 [0.00025201, 0.00070382] | 0.022222 |
| expanded / H003 | −1 SD | 45 / 45 | 0.00041229 [0.00020026, 0.00067591] | 0.022222 |
| expanded / H003 | +1 SD | 45 / 45 | 0.00036833 [0.00013655, 0.00072399] | 0 |
| expanded / H003 | Zero | 45 / 45 | 0.00038219 [0.00017631, 0.00063222] | 0.022222 |
| expanded / H003 | Random neuron · projected | 45 / 135 | 0.00051418 [0.00025234, 0.000841] | 0.022222 |
| expanded / H003 | Random direction · projected | 45 / 135 | 0.00054705 [0.00027716, 0.00088531] | 0.0074074 |
| expanded / H004 | −1 SD | 45 / 45 | 0.00034672 [0.00013585, 0.00065409] | 0.022222 |
| expanded / H004 | +1 SD | 45 / 45 | 0.00041704 [0.00019363, 0.00068753] | 0.022222 |
| expanded / H004 | Zero | 45 / 45 | 0.00026324 [0.00011683, 0.0004378] | 0 |
| expanded / H004 | Random neuron · projected | 45 / 135 | 0.00026654 [0.00012975, 0.00044272] | 0.0074074 |
| expanded / H004 | Random direction · projected | 45 / 135 | 0.00026605 [0.00011448, 0.00046332] | 0.014815 |

조건 표는 가장 최근 완료한 분할에 한정했다. 이전 단계의 모든 조건은 [steering](../logs/summary/steering.json) · [validation](../logs/summary/validation.json) · [expanded](../logs/summary/expanded.json)에 보존되어 있다.
생성은 사전 선정한 소수 prompt의 greedy 출력에 한정했다.
개입 관찰치는 다음과 같다. 같은 prompt의 반복 조건은 새로운 독립 표본이 아니며, collapse proxy는 의미 정확도·일반 성능 검증이 아니다.
- steering: 고정 prompt 5개; 비-baseline 생성 비교 70개, token sequence 변화 비율 0. 정렬된 기술적 restoration logits exact: True.
- validation: 고정 prompt 5개; 비-baseline 생성 비교 100개, token sequence 변화 비율 0.08. 정렬된 기술적 restoration logits exact: True.
- expanded: 고정 prompt 5개; 비-baseline 생성 비교 100개, token sequence 변화 비율 0.05. 정렬된 기술적 restoration logits exact: True.

## 6. 무작위 대조군과 실제 norm

대조군은 |dose|=1 SD에서만 명목상 activation 또는 raw down-projection norm을 맞춘다. ±0.5/±2 SD와 직접 크기 비교해 선택성을 주장하지 않는다. BF16 반올림 뒤 실제 norm을 별도로 기록했다. 아래 비율은 candidate +1 SD / projected random-direction control의 실제 raw-down norm 비율이다.

| 후보 / 분할 | n prompts / control seeds | 비율 평균 [관측 min, max] |
|---|---:|---|
| H001 / discovery | 15 / 3 | 0.99964 [0.98996, 1.003] |
| H001 / validation | 45 / 3 | 0.99991 [0.97977, 1.004] |
| H001 / test | 45 / 3 | 1.0003 [0.99712, 1.0039] |
| H002 / discovery | 15 / 3 | 0.9962 [0.99049, 1.0022] |
| H002 / validation | 45 / 3 | 0.99652 [0.99024, 1.0021] |
| H002 / test | 45 / 3 | 0.99662 [0.98764, 1.0021] |
| H003 / discovery | NOT_EVALUATED / NOT_EVALUATED | NOT_EVALUATED |
| H003 / validation | 45 / 3 | 0.99832 [0.99462, 1.0031] |
| H003 / test | 45 / 3 | 0.99872 [0.99411, 1.0019] |
| H004 / discovery | NOT_EVALUATED / NOT_EVALUATED | NOT_EVALUATED |
| H004 / validation | 45 / 3 | 0.99426 [0.98647, 1.0119] |
| H004 / test | 45 / 3 | 0.99462 [0.9829, 1.0153] |

관측 통계는 user-content 평균이지만 next-token 개입은 마지막 chat-wrapper token에 적용한다. 생성에서는 매 forward의 마지막 token에 적용한다. Post-MLP RMSNorm 뒤 최종 residual norm까지 일치했다고 해석하지 않는다. Restoration은 zero로 손상한 동일 입력 좌표를 clean 값으로 되돌리는 기술 검증이며 의미적 mediation 증거가 아니다.

## 7. 독립 검증·사전 등록된 분포 효과·반례

Held-out AUC는 사용자 본문 prompt-mean activation의 고정 target-vs-rest 판별이다. 양·음 표본 수와 2-sided permutation p의 Holm 보정값을 함께 표시한다. 각 분할의 고정 4가설이 하나의 family다.

| 후보 / 분할 | n positive / negative | AUC | Holm p |
|---|---:|---:|---:|
| H001 / validation | 3 / 42 | 1 | 0.001999 |
| H001 / test | 3 / 42 | 1 | 0.001999 |
| H002 / validation | 3 / 42 | 1 | 0.001999 |
| H002 / test | 3 / 42 | 1 | 0.001999 |
| H003 / validation | 3 / 42 | 1 | 0.001999 |
| H003 / test | 3 / 42 | 1 | 0.001999 |
| H004 / validation | 3 / 42 | 0.97619 | 0.001999 |
| H004 / test | 3 / 42 | 1 | 0.001999 |

등록한 분포 signature는 target의 (candidate +1 SD KL − seed 평균 projected-direction KL)에서 other의 같은 값을 뺀 차이다.
95% stratified prompt bootstrap CI이며 의미 행동의 인과 효과를 뜻하지 않는다. Validation과 test 모두 CI 하한 양수인지 사전 조건을 평가한다.

| 후보 / 분할 | n target / other | Signature 평균 [95% CI], nats | CI 하한 > 0 |
|---|---:|---|---|
| H001 / validation | 3 / 42 | -0.00016915 [-0.00065813, 0.00020413] | False |
| H001 / test | 3 / 42 | -0.00035164 [-0.00095789, 0.00012168] | False |
| H002 / validation | 3 / 42 | -0.00018601 [-0.00075359, 0.00031162] | False |
| H002 / test | 3 / 42 | -0.0005749 [-0.0016413, 0.00024567] | False |
| H003 / validation | 3 / 42 | 9.5329e-05 [-5.5042e-05, 0.00032357] | False |
| H003 / test | 3 / 42 | -0.0010387 [-0.0033069, 0.00033641] | False |
| H004 / validation | 3 / 42 | 0.00010643 [-4.9112e-05, 0.00026887] | False |
| H004 / test | 3 / 42 | 7.1124e-05 [-0.00022034, 0.00042487] | False |

반례는 각 candidate에 소유된 사례만 집계했다. Positive 원본과 paraphrase는 같은 source_id family로 묶고, family 평균 high-vs-low AUC 및 별도 4가설 Holm p를 계산한다. Role 평균의 n은 prompt 수, family AUC의 n은 독립 source family 수다.

| 후보 | Role별 n / 평균 activation | Family n high / low | Family AUC / Holm p | 역전 / 동률 pairs |
|---|---|---:|---|---:|
| H001 | Positive 2/9.8118; Negative 2/0.35623; Near-miss 2/1.5636; Paraphrase 2/2.7315; OOD 2/1.6958 | 4 / 4 | 0.875 / 0.23388 | 2 / 0 |
| H002 | Positive 2/0.61533; Negative 2/0.004876; Near-miss 2/0.04617; Paraphrase 2/0.82607; OOD 2/0.82416 | 4 / 4 | 1 / 0.13593 | 0 / 0 |
| H003 | Positive 2/0.47626; Negative 2/0.0070454; Near-miss 2/0.36171; Paraphrase 2/0.64457; OOD 2/0.42879 | 4 / 4 | 0.8125 / 0.23388 | 3 / 0 |
| H004 | Positive 2/0.32538; Negative 2/-0.0061698; Near-miss 2/0.16886; Paraphrase 2/0.15924; OOD 2/0.50561 | 4 / 4 | 0.9375 / 0.2099 | 1 / 0 |

### 별도 post-validation utility 점검

원래 동결 test의 확인적 평가와 구별하는 별도 사례집이다. 상태 PASS, Run `20261029T135550_utility`, prompt 16개, 생성 272개 / 최대 12 tokens. Validation 이후 설계 flag: True. [원자료 요약·채점·한계](../logs/summary/utility.json).

표의 각 개입 cell은 정답 수/n 및 baseline 대비 paired accuracy Δ [95% CI]다. Baseline 출력은 후보 사이에 공유하므로 반복 표시를 독립 표본으로 세지 않는다. Accepted-answer exact score의 적용 범위이며 대표 benchmark·의미 기능의 인과 증명이 아니다.

| 후보 / group | Baseline correct/n | +1 SD correct/n; Δ [CI] | −1 SD correct/n; Δ [CI] | Zero correct/n; Δ [CI] | Projected RD correct/n; Δ [CI] |
|---|---:|---|---|---|---|
| H001 / general | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
| H001 / translation | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
| H002 / general | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
| H002 / translation | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
| H003 / general | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
| H003 / translation | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
| H004 / general | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
| H004 / translation | 8/8 | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] | 8/8; 0 [0, 0] |
Utility scope: Small authored post-validation utility check; not a general benchmark or semantic causal proof.; scoring: Exact match after strip().casefold() only; explicit accepted variants frozen before utility execution. No answer extraction or punctuation removal..
Utility control scope: Random-direction control matches nominal down-projection L2 using the actual selected layer weights; BF16 generation perturbation norms are not re-audited in this check..

## 8. 예상하지 못한 관찰과 등록된 종합 판단

- BF16 실제 모델의 737,280개 MLP 좌표를 확인하고 smoke 28/28을 통과했다. Discovery 60개 입력의 사용자 본문 1,022토큰/계층을 집계했다. [근거](../logs/summary/model_load.json) · [근거](../logs/summary/smoke.json) · [근거](../logs/summary/survey.json)
- 동결한 후보 4개의 독립 test 활성 AUC는 모두 1.0이었다. 각 후보의 양성은 3개, 음성은 42개인 자체 작성 편의 표본이다. 의미 기능을 확정하는 결과는 아니다. [근거](../logs/summary/analysis.json)
- 네 후보 모두 validation/test에서 norm-matched 방향 대조군 대비 범주별 개입 효과 차이의 95% CI가 0을 포함했다. 등록한 선택적 분포 효과 기준은 지지되지 않았으며 카탈로그 상태는 모두 ambiguous다. [근거](../logs/summary/analysis.json) · [근거](../catalog/public/neurons.json)
- 별도 동결 utility 16문제(일반 8·번역 8), 실제 272회 생성에서 모든 ±1SD·zero·대조군 조건은 16/16 정답이고 출력 변화는 0이었다. 이 작은 쉬운 표본의 ceiling 결과로 일반 성능 무손상이나 뉴런의 불필요성을 주장하지 않는다. [근거](../logs/summary/utility.json)
- 번역 후보 H001/H003의 discovery 15개 범주 평균 프로필은 Pearson r=0.998583였다. 후보 선택·범주 구성의 영향을 받는 기술적 상관이며 연결이나 상호작용의 증거는 아니다. [근거](../logs/summary/analysis.json)

Discovery category-profile 관계는 다음과 같다. 같은 15개 범주 평균 간 Pearson r의 기술 통계이며 선택된 후보와 범주 구성·어휘 혼동에 의존한다. 연결·상호작용·같은 기능·독립 재현의 증거가 아니다.
측정 pair: H001 / H002 r=-0.11484 (범주 평균 n=15); H001 / H003 r=0.99858 (범주 평균 n=15); H001 / H004 r=-0.023381 (범주 평균 n=15); H002 / H003 r=-0.12269 (범주 평균 n=15); H002 / H004 r=-0.041419 (범주 평균 n=15); H003 / H004 r=-0.03991 (범주 평균 n=15).

## 9. 반증된 가설과 보류한 라벨

카탈로그의 실제 상태: `ambiguous` 4개.
`falsified` 기록이 없는 경우에도 가설이 참이라는 뜻은 아니다. 반증·제한 해석은 8절의 등록된 종합 진술에만 근거하며, KL·AUC나 기술적 restoration만으로 의미 기능 라벨을 승격하지 않는다.

## 10. 다음 연구 후보와 추가 비용

후속 실험은 등록한 실제 관찰에서 도출한다. 비용은 제안자가 적은 **예상 GPU 초**이며 실제 소모량이 아니다.

| 우선순위 | 후보 | 검증할 문제 | 예상 GPU 초 |
|---|---|---|---:|
| 1 | 관측·개입 토큰 위치 일치 | 본문 평균 관측과 마지막 wrapper 개입의 불일치를 해소한다. 동일 prompt의 본문/마지막 사용자 token/wrapper/decode 범위에서 +1SD·zero·matched 방향을 비교하고 새 독립 평가를 등록한다. | 872 |
| 2 | 어휘·길이 대조와 반례 확대 | 번역 요청·언어명 언급·비번역 문장을 길이와 문자 체계별로 맞추고, 프로그래밍 요청에서 Python/Rust/R 이름과 코드 의도를 교차한다. 4대4 family 검증의 작은 표본을 늘린다. | 581 |
| 3 | 후기 정규화와 dose 비단조성 분리 | down-projection 뒤 RMSNorm의 실제 residual perturbation을 측정하고 norm을 맞춘다. 현재 작은 dose sweep의 비단조 평균을 normalization·BF16 rounding·prompt 변동 중 하나로 단정하지 않는다. | 436 |

## 11. 관련 선행 구현과 출처

이번 환경에서 upstream 프로젝트의 실험을 재현한 것은 아니다. 아래는 실제 소스·구조·라이선스를 확인한 설계 참고이며 upstream 수치는 본 연구 수치에 섞지 않았다. [고정 revision·라이선스·경로 감사](../docs/REFERENCE_AUDIT.md).

- [Lyra neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): steering-first 후보 탐색·대조군·연구 로그.
- [Apple ml-refusal-neurons](https://github.com/apple-aiml-research/ml-refusal-neurons): 단일 좌표 훅·평가 분리.
- [EleutherAI Delphi](https://github.com/EleutherAI/delphi): 활성 증거·설명 가설·독립 점수화 분리. SAE feature와 native neuron은 구별했다.
- [interp-engine](https://github.com/decoderesearch/interp-engine), [NNsight](https://github.com/ndif-team/nnsight), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): 계측 point·backend parity 참고. 본 실험의 기준선은 native HF eager다.

## 12. 남은 한계

직접 작성한 작은 편의 corpus, held-out 범주별 소수 양성, discovery의 수십만 좌표 선택 편향이 있다. Greedy 생성의 소수 prompt·짧은 길이와 반복 proxy는 대표 정확도·일반 품질·모든 문맥 안정성을 평가하지 않는다. 멀티모달·장문 benchmark·외부 engine parity·독립 모델 반복·행동의 의미적 인과 검증은 NOT_EVALUATED다.

[원전 논문 5편과 신규성·필요 대조 검토](../docs/RELATED_WORK.md). 후속 연구의 신규성은 미평가다.

- Discovery는 범주당4개 prompt, held-out 양성은 범주당3개다. 토큰이나 random seed를 독립 prompt 수로 계산하지 않았다.
- 관측은 사용자 본문 평균, 개입은 마지막 chat-template token이다. 자연 활성 범위와 위치가 다르며 음의 dose가 관측 범위를 벗어나는 경우도 있다.
- 유한 64-token reservoir quantile은 근사치이며 tail·rare activation의 모집단 추정이 아니다.
- 16개 utility는 별도 post-validation 편의표본이며 정답률이 ceiling이다. 장문·다중모달·대표 benchmark·다른 모델 seed·외부 backend는 미평가다.
- 범주별 상관·프로필 유사도·KL 변화는 의미 기능이나 뉴런의 인과적 필요성·충분성을 입증하지 않는다.
- 후속 후보의 신규성·논문 가능성은 NOT_EVALUATED다. 단일 뉴런 개입·random norm 대조 자체는 선행연구에 존재한다.

## 13. 실제 계산·저장 자원

GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition. CPU: AMD Ryzen 5 5600X 6-Core Processor. OS: Ubuntu 24.04.5 LTS. RAM 원기록: 65753988 kB.
Resource snapshot KST: 2026-10-29T14:00:28.657507+09:00; 프로젝트 disk free 407.282 GiB. 단일 시점 snapshot이며 stage peak와 구별한다.
모델 다운로드: PASS, 22.307 GiB / 652.38초. 실제 모델 로딩 6.4006초 / allocated 22.277 GiB.
표에 있는 PASS stage elapsed 합계: 852.81초. 설치·다운로드·모델 로딩·개발 시간을 제외하며, GPU 이용률을 적분한 계산 시간이 아니다.
실행 환경: Python 3.12.3, CUDA build 13.2; accelerate 1.15.0, huggingface_hub 2.3.0, matplotlib 3.11.2, numpy 2.5.4, pyarrow 26.0.0, torch 2.15.0, transformers 5.21.0.

## 14. 재현 방법과 원본·해시

프로젝트 루트에서 실행한다. HF cache를 프로젝트 디스크에 두고 deterministic cuBLAS 설정을 worker 시작 전에 적용한다. 다음 명령은 게시된 discovery 요약·고정 가설을 유지한 채 독립 평가를 재실행한다. Test를 후보 재선정에 사용하지 않는다.

```bash
export HF_HOME="$PWD/data/huggingface"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/prepare_model.py
for stage in smoke steering validation expanded shutdown; do
  PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit "$stage"
done
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli worker
PYTHONPATH=src .venv/bin/python scripts/run_utility.py --check
PYTHONPATH=src .venv/bin/python scripts/run_utility.py
.venv/bin/python scripts/analyze_results.py
.venv/bin/python scripts/analyze_results.py --check
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_steering_figures.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/build_report.py --check
.venv/bin/python scripts/build_readme.py
.venv/bin/python scripts/build_readme.py --check
```

Pilot·survey 수집도 다시 수행하려면 새 연구 사본에서 가설 동결 전에 `submit pilot`, `submit survey`를 실행하고 새 run ID와 가설을 등록한다. 기존 등록 가설 파일을 덮어쓰거나 원래 discovery 요약을 교체한 뒤 동일 연구로 분석하지 않는다.

가중치·전체 activation·원본 생성 로그는 Git에 올리지 않는다. 로컬 `runs/<run_id>/`의 metadata·Parquet·NPZ·JSONL과 SQLite를 공개 run ID·artifact SHA256으로 연결한다. 재실행 시 현재 게시된 summary 파일은 갱신되지만 이전 Run은 보존된다. 보고서의 해석은 기록된 synthesis.json에서 가져오므로 재현자가 수치 재계산 뒤 종합 진술의 근거도 검토해야 한다.

| 공개 재현 입력 | SHA256 |
|---|---|
| [configs/hypotheses.json](../configs/hypotheses.json) | `62dd2149c2c1f9f2825e945add634da9a244f6a23dda8d561e81f28f35c413d3` |
| [configs/validation_cases.json](../configs/validation_cases.json) | `f635187ea338bc702fe2abe8551d0b50f4774febfe87b0c967d75b08e5b00aeb` |
| [requirements.lock.txt](../requirements.lock.txt) | `4f434d4387f7a26cba053fac2310cd98d69bd63cb50bca8de50967a2b2a23b78` |
| [logs/summary/analysis.json](../logs/summary/analysis.json) | `b0a6529cc187eb2088be1cc87e20c50f3ce1bd960e43a94724c030ec2fdc30fe` |
| [logs/summary/synthesis.json](../logs/summary/synthesis.json) | `a22997673ce566c827f031285bca99cd7618398cd375e7b866afd6cd87e14cd9` |
| [logs/summary/utility.json](../logs/summary/utility.json) | `93f1485e160ee06375c3721b17ef397edfdf3eebee095b025cd2c2a9a81dbc4f` |
| [configs/utility.yaml](../configs/utility.yaml) | `986f74d4d49fa965bd1d9d88cf3dbc72f7504c1668dd973babc9b0b9843b269c` |

| 단계 | 원본 interventions.jsonl SHA256 |
|---|---|
| steering | `96a7045c210d37dfc4e6ed15f474e95c09486224e4854fa74c4643bec02dfd99` |
| validation | `61cb5c1c51d14046fa88385fca09cc819e8cd7f785052033ad9393791dd54c26` |
| expanded | `935bf63c4533d319addfe6374ed598dc63714b4ccec8f87b787060fc2054a0fd` |
| utility (utility_generations.jsonl) | `26fa569f2e4f28616d0f863936471ea4d958285df44b200071e8c0ebff0c84df` |

## 15. GitHub 체크포인트와 결과 링크

공개 저장소: [PigeonLabs/Gemma4_neuron_LAB](https://github.com/PigeonLabs/Gemma4_neuron_LAB). 원격 반영 여부는 publication receipt에 기록된 값만 표시한다.

| 체크포인트 | KST 날짜 | Commit | 원격 확인 |
|---|---|---|---|
| CP0 | 2026-10-29 | [d1d38357](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/d1d383578946b0d875cbe6561704b0e154430eb5) | 확인됨 |
| CP1 | 2026-10-29 | [cad4c697](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/cad4c6974c4c9d1fd0aeb8f661740cb4fd0d3c33) | 확인됨 |
| CP2 | 2026-10-29 | [cad4c697](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/cad4c6974c4c9d1fd0aeb8f661740cb4fd0d3c33) | 확인됨 |
| CP3 | 2026-10-29 | [c38728f9](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/c38728f9b89aca7b8113c66efca28b034b5d33fe) | 확인됨 |
| CP4 | 2026-10-29 | [df4c2c35](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/df4c2c35b0ddc4933af87f6a8087b03a658b56a0) | 확인됨 |
| CP5 | 2026-10-29 | [1d03f84e](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/1d03f84e587c999be43c55be7b2041498d1bd235) | 확인됨 |
| CP6 | 2026-10-29 | [3fb74111](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/3fb7411188f98a086f7dd6bda38e377700c8f213) | 확인됨 |
| CP7 | 2026-10-29 | [3fb74111](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/3fb7411188f98a086f7dd6bda38e377700c8f213) | 확인됨 |
| CP8 | 2026-10-29 | [3fb74111](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/3fb7411188f98a086f7dd6bda38e377700c8f213) | 확인됨 |

[실험 분석](../logs/summary/analysis.json) · [종합 근거](../logs/summary/synthesis.json) · [Neuron catalog](../catalog/public/neurons.json) · [방법론](../docs/METHODOLOGY.md) · [연구 로그](../docs/RESEARCH_LOG.md) · [초기 보고서](../reports/INITIAL_FINDINGS_KO.md)

![동결 가설의 held-out activation 판별](../figures/heldout_auc.svg)

![반례 role·source-family 관찰](../figures/counterexample_activation.svg)
