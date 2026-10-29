<!-- Generated from measured summary files by scripts/build_readme.py. -->
# Gemma 4 Neuron Lab

*Autonomous Discovery, Steering, and Causal Analysis of LLM Neurons*

## Project Overview

Gemma 4 12B의 MLP 뉴런을 BF16 활성값 수집, steering 개입, 무작위 대조군과 독립 검증으로 연구합니다. 뉴런은 `GELU(gate(x)) × up(x)` 이후, `down_proj` 이전의 중간 활성 좌표로 정의합니다. Python·PyTorch·Transformers, streaming 통계, SQLite·Parquet를 사용합니다. [연구 지침](MASTER.md)과 [방법론](docs/METHODOLOGY.md)에 재현 조건과 한계를 기록합니다.

## Research Status

기록 기준: **2026-10-29 KST**. 아래 `PASS`는 해당 실행·기술 검증 상태입니다. 뉴런 기능·인과성의 검증 상태는 카탈로그에서 별도로 표시합니다.

| 단계 | 실행·기록 상태 | 근거 |
|---|---|---|
| CP0 환경·저장소·참조 감사 | PASS | [기록](docs/REFERENCE_AUDIT.md) |
| CP1 BF16 모델·런타임 아키텍처 | PASS | [기록](logs/summary/model_load.json) |
| CP2 훅·Smoke Test | PASS | [기록](logs/summary/smoke.json) |
| CP3 활성 통계·시각화 | PASS · 전체 계층 | [기록](logs/summary/survey.json) |
| CP4 Steering·무작위 대조군 | PASS | [기록](logs/summary/steering.json) |
| CP5 독립 가설 검증 | PASS | [기록](logs/summary/validation.json) |
| CP6 확대 탐색·카탈로그 | NOT_EVALUATED · 카탈로그 있음 | 미측정·미기록 |
| CP7 반례·연구 후보 종합 | NOT_EVALUATED | 미측정·미기록 |
| CP8 최종 기록·공개 정리 | NOT_EVALUATED | 미측정·미기록 |

## Key Findings

- 실제 BF16 모델은 **48계층 × 15,360좌표 = 737,280개 MLP 뉴런**입니다. [런타임 아키텍처](logs/summary/model_load.json)
- Smoke Test **28/28** 검증을 통과했고, read-only 훅의 최대 logits 차이는 **0**였습니다. [기술 검증](logs/summary/smoke.json)
- 전체 계층 survey에서 **60개 프롬프트·15개 범주**, 계층당 **1,022개 user-content 토큰**을 측정했습니다. 후보 선택성은 discovery 상관 통계이며 기능은 아직 확정하지 않습니다. [활성 통계](logs/summary/survey.json)

## Experiment Results

**H001 · validation / validation 분할**의 실제 집계입니다. [전체 조건·후보 결과](logs/summary/validation.json)

| 조건 | n 프롬프트 / 개입 수 | 평균 KL · 95% CI (nats) | 다음 토큰 top-1 변화 |
|---|---:|---:|---:|
| Baseline | 45 / 45 | 0 [0, 0] | 0.0% |
| Additive +1SD | 45 / 45 | 0.0004809 [0.0001169, 0.0009797] | 4.4% |
| Negative −1SD | 45 / 45 | 0.0006925 [0.0001437, 0.001564] | 2.2% |
| Zero Ablation | 45 / 45 | 0.0002079 [5.336e-05, 0.0004003] | 0.0% |
| Activation Restoration | 45 / 45 | 0 [0, 0] | 0.0% |
| Random Neuron · projected | 45 / 135 | 0.0003755 [0.0001629, 0.0006532] | 0.7% |
| Random Direction · projected | 45 / 135 | 0.0003212 [0.0001586, 0.0005092] | 0.0% |

95% CI는 프롬프트 단위 percentile bootstrap입니다. 무작위 대조군의 seed는 프롬프트 안에서 평균하며, 대조군은 **+1SD 조건에서만** 측정했습니다. `projected`는 nominal down-projection L2를 맞춘 조건이며 실제 BF16 교란 norm도 별도로 기록합니다.

활성 통계는 **user-content 토큰**을 수집합니다. Steering은 **입력의 마지막 chat-wrapper 토큰**에 개입하며, 생성에서는 매 forward의 마지막 토큰에 개입합니다. 두 위치의 측정 범위를 구분해야 합니다. Restoration은 같은 입력의 clean 활성값으로 손상 좌표를 되돌리는 기술 검증입니다.

## Visualizations

![계층별 활성 변동](figures/layer_activation_variability.svg)

Discovery n=60, 계층당 1,022개 user-content 토큰. 뉴런 간 p10–p90은 관측 분포의 폭이며 신뢰구간이 아닙니다.

![계층·범주별 활성 패턴](figures/category_activation_heatmap.svg)

Discovery n=60, 계층당 1,022개 user-content 토큰. 프롬프트별 가중치를 같게 한 범주 평균의 표준화된 절대 편차입니다.

![후보 뉴런 선택성](figures/candidate_selectivity.svg)

Discovery n=60, 계층당 1,022개 user-content 토큰. 선택된 discovery 후보의 기술 통계이며, 유의성 검정의 z-score나 인과 증거가 아닙니다.

![후보 뉴런 활성 분포](figures/candidate_activation_histograms.svg)

Discovery n=60, 계층당 1,022개 user-content 토큰. 제한된 토큰 reservoir의 근사 히스토그램이며, 각 패널의 활성 축 범위가 다릅니다.

![steering dose H001](figures/steering_dose_H001.svg)

H001 additive dose sweep on 15 discovery prompts; observed mean next-token KL and 95% prompt-bootstrap intervals, projected-direction control measured only at +1SD. Baseline KL=0 omitted; log y-axis. Intervention is at the final chat-wrapper token.

![heldout auc](figures/heldout_auc.svg)

Target-vs-rest prompt-mean activation AUC for 4 discovery-frozen hypotheses, showing only measured validation only; test not evaluated. n=3 positive / 42 negative per candidate/split. No confidence interval is claimed; AUC was independently recomputed from pairwise prompt comparisons.

![counterexample activation](figures/counterexample_activation.svg)

User-content prompt-mean activation for 40 measured candidate-matched counterexample cases: two prompts per role for each of 4 hypotheses. Panel-specific activation scales; dotted lines indicate related original/paraphrase pairs, which are not independent replicates.

[그림 출처·표본 수·해시](figures/metadata.json)

## Interesting Neurons

가설과 검증 상태는 측정 카탈로그의 기록을 그대로 표시합니다.

| Layer | Neuron | 기능 가설 | 상태 | 측정 효과·근거 |
|---:|---:|---|---|---|
| 15 | 9637 | 번역 과제 문맥에 반응하는 좌표일 수 있다. | ambiguous | +1SD KL=0.0004809 [카탈로그](catalog/public/neurons.json) |
| 22 | 7859 | 주어진 정보로 답을 결정할 수 없는 문맥에 반응하는 좌표일 수 있다. | ambiguous | +1SD KL=0.0003824 [카탈로그](catalog/public/neurons.json) |
| 26 | 719 | 후기 계층의 번역 과제 문맥 좌표일 수 있다. | ambiguous | +1SD KL=0.00034 [카탈로그](catalog/public/neurons.json) |
| 35 | 990 | 프로그래밍 과제 전반에 반응하는 좌표일 수 있다. | ambiguous | +1SD KL=0.0001437 [카탈로그](catalog/public/neurons.json) |

카탈로그 상태: `ambiguous` 4.

## Research Timeline

| KST 날짜 | 체크포인트 | Commit | 원격 확인 |
|---|---|---|---|
| 2026-10-29 | CP0 | [d1d38357](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/d1d383578946b0d875cbe6561704b0e154430eb5) | 확인됨 |
| 2026-10-29 | CP1 | [cad4c697](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/cad4c6974c4c9d1fd0aeb8f661740cb4fd0d3c33) | 확인됨 |
| 2026-10-29 | CP2 | [cad4c697](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/cad4c6974c4c9d1fd0aeb8f661740cb4fd0d3c33) | 확인됨 |
| 2026-10-29 | CP3 | [c38728f9](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/c38728f9b89aca7b8113c66efca28b034b5d33fe) | 확인됨 |
| 2026-10-29 | CP4 | [df4c2c35](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/df4c2c35b0ddc4933af87f6a8087b03a658b56a0) | 확인됨 |

## Reproduction

실제 실행 환경: Python 3.12.3 · torch 2.15.0 · transformers 5.21.0. GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.

모델 revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. 정확한 의존성은 [lock 파일](requirements.lock.txt), 계산·저장 예산과 실행 조건은 [configs](configs)에 있습니다.

프로젝트 루트에서 환경·고정 revision 다운로드:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install -e ".[test,explorer]"
.venv/bin/python scripts/prepare_model.py
```

단일 GPU worker에 smoke → pilot → survey 순서로 작업을 제출합니다.

```bash
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit smoke
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit pilot
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit survey
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit steering
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit validation
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit expanded
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli worker
```

측정 요약에서 그림·README를 재생성하고 수치 동기화를 검사합니다:

```bash
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/build_readme.py
.venv/bin/python scripts/build_readme.py --check
.venv/bin/python -m pytest -q
```

가중치·전체 활성 텐서·원본 생성 로그는 로컬에 보존합니다. 공개 요약의 해시와 run ID로 원본 산출물을 연결합니다.

## Research Log

[연구 기록](docs/RESEARCH_LOG.md) · [참조 구현 감사](docs/REFERENCE_AUDIT.md) · [아키텍처](docs/ARCHITECTURE.md) · [방법론](docs/METHODOLOGY.md) · [실험 계획](docs/EXPERIMENT_PLAN.md)

## Limitations & Future Work

자체 작성 소규모 text-only 코퍼스의 결과입니다. Discovery 후보 선택성은 인과적 기능을 뜻하지 않으며, next-token KL·top-1 변화는 의미적 정확도나 일반 성능 보존을 대신하지 않습니다. 설계된 생성 품질 평가는 빈 출력·반복 proxy로 제한되며 내용 품질은 별도 평가가 필요합니다. 무작위 대조군의 여러 seed는 독립 모델 반복 실험이 아닙니다.

검증 범위를 넓히려면 개입·관측 토큰 위치를 맞춘 대조, 더 다양한 독립 원본, 의미적 행동 평가와 반복 실험이 필요합니다. 이미지·음성·대규모 장문 일반화는 이 기록에서 검증하지 않았습니다.
