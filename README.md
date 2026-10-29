# Gemma 4 Neuron Lab

*Autonomous Discovery, Steering, and Causal Analysis of LLM Neurons*

Gemma 4 12B의 MLP 뉴런을 실제 활성값과 개입 실험으로 연구합니다. BF16 계측, streaming 통계, 무작위 대조군, 독립 검증을 통해 재현 가능한 근거를 축적합니다. 연구 지침은 [MASTER.md](MASTER.md)에 있습니다.

## Research Status

2026-10-29 KST — 환경·참조 구현 감사 및 계측 파이프라인 준비 중입니다.

| 단계 | 상태 | 근거 |
|---|---|---|
| CP0 환경·저장소·공식 자료 감사 | 진행 중 | [연구 로그](docs/RESEARCH_LOG.md) |
| CP1 BF16 모델 로딩·런타임 아키텍처 | NOT_EVALUATED | 가중치 로딩 전 |
| CP2 훅·Smoke Test | NOT_EVALUATED | 검증 전 |
| CP3 활성 통계·시각화 | NOT_EVALUATED | 미측정 |
| CP4 Steering·대조군 | NOT_EVALUATED | 미측정 |
| CP5 독립 가설 검증 | NOT_EVALUATED | 미측정 |
| CP6 확대 탐색·카탈로그 | NOT_EVALUATED | 미측정 |
| CP7 연구 후보·반례 | NOT_EVALUATED | 미평가 |
| CP8 최종 정리 | NOT_EVALUATED | 연구 진행 중 |

## Key Findings

현재 뉴런 기능에 대한 발견은 없습니다. 공식 모델 ID `google/gemma-4-12B-it`의 공개 접근을 확인했으며, 모델 revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`로 재현 조건을 고정했습니다. 이는 실행 결과가 아닌 메타데이터 감사입니다.

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

Python 3.12, PyTorch, Transformers를 사용합니다. 실제 로딩·훅 검증이 끝난 후 실행 명령과 설치 버전을 갱신합니다. 가중치와 원시 결과는 GitHub에 올리지 않습니다.

## Research Log

[연구 기록](docs/RESEARCH_LOG.md) · [참조 구현 감사](docs/REFERENCE_AUDIT.md) · [아키텍처](docs/ARCHITECTURE.md) · [실험 계획](docs/EXPERIMENT_PLAN.md)

## Limitations & Future Work

모델 실행 전 단계입니다. 개별 뉴런 기능·인과성·일반화는 아직 평가되지 않았습니다. 첫 연구 주기는 text-only, 소규모 독립 분할, 제한된 개입 강도로 시작합니다.
