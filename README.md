# Gemma 4 Neuron Lab

*Autonomous Discovery, Steering, and Causal Analysis of LLM Neurons*

Gemma 4 12B의 MLP 뉴런을 실제 활성값과 개입 실험으로 연구합니다. BF16 계측, streaming 통계, 무작위 대조군, 독립 검증을 통해 재현 가능한 근거를 축적합니다. 연구 지침은 [MASTER.md](MASTER.md)에 있습니다.

## Research Status

2026-10-29 KST — BF16 모델 로딩과 실제 모델 훅 검증을 완료했습니다. 다음 단계는 활성 통계입니다.

| 단계 | 상태 | 근거 |
|---|---|---|
| CP0 환경·저장소·공식 자료 감사 | PASS | [연구 로그](docs/RESEARCH_LOG.md) |
| CP1 BF16 모델 로딩·런타임 아키텍처 | PASS | [실제 로딩](logs/summary/model_load.json) |
| CP2 훅·Smoke Test | PASS | [28/28 검증](logs/summary/smoke.json) |
| CP3 활성 통계·시각화 | NOT_EVALUATED | 미측정 |
| CP4 Steering·대조군 | NOT_EVALUATED | 미측정 |
| CP5 독립 가설 검증 | NOT_EVALUATED | 미측정 |
| CP6 확대 탐색·카탈로그 | NOT_EVALUATED | 미측정 |
| CP7 연구 후보·반례 | NOT_EVALUATED | 미평가 |
| CP8 최종 정리 | NOT_EVALUATED | 연구 진행 중 |

## Key Findings

실제 모델은 48계층 × 15,360좌표 = **737,280개 MLP 뉴런**, 11,959,730,176개 BF16 파라미터입니다. 3개 프롬프트에서 read-only 훅의 logits 차이 0, greedy 생성 동일, 개입 수식·손상 후 복원·훅 제거 검증을 포함한 **28/28 smoke 검증**을 통과했습니다. peak GPU allocated는 약 **22.33GiB**입니다. 뉴런 기능은 아직 미평가입니다. [검증 원본](logs/summary/smoke.json)

공식 모델 ID `google/gemma-4-12B-it`의 공개 접근을 확인했으며, 모델 revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`로 재현 조건을 고정했습니다. 모델 메타데이터와 실제 모듈 구조가 일치했습니다.

## Experiment Results

Baseline, additive/negative steering, zero ablation, restoration, random neuron/direction control 모두 **NOT_EVALUATED**입니다. 미측정 수치나 예시 발견을 게시하지 않습니다.

## Visualizations

측정 완료 후 요약 JSON에서 실제 그림을 생성합니다.

## Interesting Neurons

카탈로그는 아직 비어 있습니다. [공개 카탈로그](catalog/public/README.md)

## Research Timeline

| KST 날짜 | 체크포인트 | 내용 |
|---|---|---|
| 2026-10-29 | CP0 시작 | 빈 원격 저장소와 main 기본 브랜치 확인, 실제 GPU 접근 및 의존성 준비 |

## Reproduction

Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.21.0을 사용했습니다. 의존성은 [requirements.lock.txt](requirements.lock.txt), 설정은 [configs](configs)에 있습니다. `PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit smoke` 후 같은 환경에서 `python -m neuron_lab.cli worker`를 실행합니다. 가중치와 원시 결과는 GitHub에 올리지 않습니다.

## Research Log

[연구 기록](docs/RESEARCH_LOG.md) · [참조 구현 감사](docs/REFERENCE_AUDIT.md) · [아키텍처](docs/ARCHITECTURE.md) · [실험 계획](docs/EXPERIMENT_PLAN.md)

## Limitations & Future Work

소규모 smoke 단계까지 실행했습니다. 개별 뉴런 기능·인과성·일반화는 아직 평가되지 않았습니다. 첫 연구 주기는 text-only, 소규모 독립 분할, 제한된 개입 강도로 시작합니다.
