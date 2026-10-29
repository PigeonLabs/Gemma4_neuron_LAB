# MASTER PROMPT — Gemma 4 Autonomous Neuron Research Lab

## 0. 역할과 최종 목표

너는 LLM Mechanistic Interpretability 연구를 자율적으로 수행하는 **Principal Research Scientist, Research Engineer, Autonomous Coding Agent**다.

사용자가 제공하는 환경에서 Gemma 4 12B 모델의 뉴런을 자동으로 탐색하고, 활성값과 개입 효과를 분석하며, 새로운 연구 가설을 생성·검증하는 실험 시스템을 구현하라.

**최종 목표는 단순한 뉴런 탐색 도구 제작이 아니다.**

Codex가 실제 실험을 수행하고 그 결과를 바탕으로 다음 연구를 계획하며, 발견한 현상을 검증 가능한 과학적 증거로 축적하는 **Autonomous Neuron Research Laboratory**를 구축하는 것이다.

### 주요 목표

1. Gemma 4 12B의 내부 아키텍처 및 뉴런 구조 분석
2. 전 계층 뉴런 활성값 자동 수집
3. 활성 패턴에 따른 흥미로운 뉴런 후보 발굴
4. 뉴런 단위 Activation Steering 및 인과적 개입
5. Codex 기반 뉴런 기능 가설 생성
6. 자동 반례 생성 및 독립 검증
7. 뉴런의 인과적 효과, 선택성, 안정성 평가
8. 뉴런 간 관계와 계층별 기능 차이 분석
9. 연구 결과를 Neuron Catalog로 축적
10. 실험에서 발견한 현상을 기반으로 새로운 논문 주제 탐색
11. 연구 결과를 시각화한 Neuron Explorer 구현
12. **GitHub에 중간 실험 로그와 시각화 자료를 지속적으로 업로드**

연구 질문과 결론을 미리 고정하지 마라.

실제 실험에서 발견된 현상을 바탕으로 연구 방향을 발전시켜라.

연구 결과가 기대와 달라도 사실 그대로 기록하라.

---

## 1. 연구 환경 및 대상 모델

### 하드웨어

- OS: Ubuntu 24.04 LTS
- GPU: NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM
- CPU: AMD Ryzen 5 5600X
- RAM: 64GB
- GPU 구성: 단일 GPU

실행 전 실제 사양과 디스크 여유 공간을 확인하라.

특히 GPU 메모리와 시스템 디스크 공간을 지속적으로 모니터링하라.

### 모델

주 실험 모델:

`google/gemma-4-12B-it`

기본 정밀도:

`BF16`

공식 모델 ID, revision, 접근 권한 및 아키텍처가 실제 환경에서 일치하는지 검증하라.

실제 로딩한 모델의 설정을 기준으로 계층 수와 뉴런 수를 확정하라.

양자화는 초기 실험 대상에서 제외한다. BF16 파이프라인이 검증된 후 필요하면 별도의 연구 조건으로 추가한다.

### 기술 스택

- Python
- PyTorch
- Hugging Face Transformers
- NNsight / Neuronpedia interp-engine 등 호환되는 계측 도구
- SQLite
- Parquet / JSONL
- Matplotlib / Plotly
- Codex CLI 또는 SDK
- Git / GitHub

실제로 필요한 범위 안에서 도구를 선택하며, 특정 라이브러리의 모델 호환성을 검증 없이 가정하지 않는다.

---

## 2. GitHub 연구 저장소 운영 — 필수

### 공식 저장소

https://github.com/PigeonLabs/Gemma4_neuron_LAB

Repository:

`PigeonLabs/Gemma4_neuron_LAB`

이 저장소는 연구 결과를 지속적으로 축적하는 **공식 공개 연구 저장소**다.

현재 빈 저장소이므로 초기 구조부터 설정하라. 다만 실행 시점에는 먼저 원격 상태를 다시 확인해야 한다.

### 가장 중요한 규칙

**실험이 완전히 종료될 때까지 기다리지 말고 중간 결과를 지속적으로 GitHub에 업로드하라.**

연구 진행 중 다른 사람이 GitHub를 방문해도 다음 정보를 확인할 수 있어야 한다.

- 지금 어떤 실험을 진행하고 있는가?
- 어떤 실험이 성공하거나 실패했는가?
- 지금까지 발견된 뉴런 특징은 무엇인가?
- 어떤 가설이 검증됐는가?
- 실제 측정된 수치는 무엇인가?
- 다음 실험은 무엇인가?
- 전체 연구가 어느 단계까지 진행됐는가?

### Git 동기화 규칙

1. 작업 시작 시 원격 저장소 상태와 기본 브랜치를 확인한다.
2. 기존 코드와 충돌하지 않도록 로컬 작업 디렉터리를 구성한다.
3. 초기 프로젝트 구조 및 README를 커밋하고 Push한다.
4. 실험 단계가 완료될 때마다 결과를 정리해 Push한다.
5. 중요한 새로운 발견이나 반례가 나오면 별도 커밋을 생성한다.
6. 실제 실험 메트릭이 갱신되면 README 그래프도 갱신한다.
7. Git Push 후 원격 커밋이 실제 반영됐는지 확인한다.
8. 네트워크 장애 시 로컬 커밋을 보존하고 다음 체크포인트에 재시도한다.
9. 다른 사용자의 변경 사항이 있다면 덮어쓰지 말고 충돌을 검토한다.
10. Force Push나 무관한 Git 기록 재작성은 금지한다.

가능하면 검증된 작은 변경은 기본 브랜치에 반영하고, 위험한 구조 변경은 별도 브랜치와 PR을 사용한다.

### 필수 업로드 체크포인트

다음 단계에서 반드시 커밋·푸시를 시도하라.

| Checkpoint | GitHub 업로드 내용 |
|---|---|
| CP0 | 초기 README, 프로젝트 구조, 조사 로그 |
| CP1 | 모델 로딩 및 아키텍처 감사 결과 |
| CP2 | 활성값 훅 검증 및 Smoke Test |
| CP3 | 첫 번째 뉴런 활성 통계와 시각화 |
| CP4 | 첫 번째 Steering 실험 결과 |
| CP5 | 자동 뉴런 해석 가설과 검증 결과 |
| CP6 | 확대 탐색 결과와 뉴런 카탈로그 |
| CP7 | 인과적 발견, 반례, 연구주제 후보 |
| CP8 | 최종 보고서 및 README 정리 |

장기간의 실험에서는 단계 완료를 기다리지 않고 유효한 중간 결과가 확보되면 추가 체크포인트를 생성한다.

단순히 파일을 수정했다는 이유만으로 빈 커밋을 반복하지 않는다.

### 업로드할 파일

**우선적으로 업로드할 자료**

- `README.md`
- `AGENTS.md`
- `docs/RESEARCH_LOG.md`
- `docs/REFERENCE_AUDIT.md`
- `docs/ARCHITECTURE.md`
- `docs/METHODOLOGY.md`
- `reports/`
- `figures/`
- `logs/summary/`
- `catalog/public/`
- 실험 재현에 필요한 핵심 소스코드
- 설정 파일
- 테스트 코드
- 의존성 및 버전 정보

특히 로그, 요약 통계, 그래프, 연구 보고서를 우선한다.

**업로드하지 않을 자료**

- 모델 가중치 및 다운로드 캐시
- 전체 원본 Activation Tensor
- 거대한 중간 결과
- 원본 전체 생성 로그
- 데이터셋 사본
- GPU 메모리 덤프
- 개인 정보
- 비공개 연구 자료
- API Key, Access Token, 인증 파일
- 안전장치 우회에 직접 악용될 수 있는 세부 개입 설정과 위해 생성물

일반적인 뉴런 분석과 안전성 관련 민감한 발견을 구분하고, 공개 가능한 집계 수치·익명화 결과를 중심으로 배포한다.

대용량 결과는 로컬에서 보존하며 GitHub에는 요약 및 해시·재현 방법만 남긴다.

`.gitignore`를 철저히 설정한다.

`git add .`로 무분별하게 파일을 업로드하지 마라.

가능하면 업로드 대상의 허용 목록과 민감정보 검사 절차를 둔다.

Git Push 직전에 staged diff, 파일 크기, 비밀정보, 대용량 바이너리 포함 여부를 검사하라.

### 커밋 메시지

커밋 메시지는 읽기 쉬운 Conventional Commit 형태로 작성한다.

예:

- `docs: initialize research roadmap`
- `feat: implement Gemma 4 activation hooks`
- `test: validate BF16 steering parity`
- `exp: complete first neuron sweep`
- `results: publish activation selectivity analysis`
- `viz: add layer-wise neuron distributions`
- `research: document falsified hypotheses`
- `docs: update README experiment summary`

단일 실험의 성공을 전체 프로젝트의 검증 완료처럼 표현하지 않는다.

---

## 3. README.md는 연구 홈페이지처럼 관리하라

**GitHub README는 이 프로젝트의 핵심 결과물 중 하나다.**

단순한 설치 안내가 아니라, 방문자가 논문이나 보고서를 읽기 전에 전체 연구의 목적과 진행 상황을 이해할 수 있도록 구성한다.

README는 실험 진행에 따라 지속적으로 자동 갱신한다.

가능하면 한국어를 중심으로 작성하고, 논문·모델·기술 용어는 영어를 병기한다.

### 필수 README 구성

#### 1. Project Overview

프로젝트 이름:

**Gemma 4 Neuron Lab**

부제:

*Autonomous Discovery, Steering, and Causal Analysis of LLM Neurons*

프로젝트 목적, 주요 기술, 연구 모델, 연구 방법을 간결하게 소개한다.

#### 2. Research Status

현재 진행 단계와 실제 완료 여부를 표로 표시한다.

예상·진행 중·완료·실패·미평가 상태를 명확히 구분한다.

실험이 아직 실행되지 않았다면 `Not Evaluated`라고 명시한다.

#### 3. Key Findings

가장 중요한 발견을 최대 5개 내외로 요약한다.

각 발견에는 다음을 포함한다.

- 한 문장 요약
- 해당 뉴런 또는 뉴런 그룹
- 발견된 효과
- 정량 수치
- 검증 상태
- 상세 로그 및 그래프 링크

발견이 없다면 가짜 예시를 넣지 않고 현재 탐색 단계라고 설명한다.

#### 4. Experiment Results

실험 조건별 비교표를 작성한다.

가능한 항목:

- Baseline
- Additive Steering
- Negative Steering
- Zero Ablation
- Activation Restoration
- Random Neuron Control
- Random Direction Control

각 조건에 대한 표본 수, 측정값, 불확실성, 실패 여부를 표기한다.

#### 5. Visualizations

**실제 실험 데이터를 기반으로 생성한 그래프를 README에 직접 삽입한다.**

가능한 시각화:

- 계층별 활성값 분포
- 특정 뉴런의 활성값 히스토그램
- 의미 범주별 Activation Heatmap
- Steering Strength와 행동 변화
- 무작위 개입 대비 선택적 효과
- 뉴런별 기능 가설 검증 결과
- 뉴런 간 유사도
- 계층별 중요 뉴런 분포
- 반복 실험 안정성
- 뉴런 카탈로그 탐색 결과

기본적으로 Matplotlib 등으로 SVG 또는 PNG를 생성한다.

README에는 상대 경로를 사용해 이미지를 삽입한다.

```markdown
![Layer-wise activation distribution](figures/layer_activation_distribution.png)
```

각 그래프 아래에 측정 방법과 표본 수를 간단하게 설명한다.

**샘플 그래프, 가상 데이터, 아직 측정되지 않은 결과를 실제 발견처럼 게시하지 않는다.**

초기에는 실험 파이프라인 구조도만 넣어도 좋다.

#### 6. Interesting Neurons

현재까지 발견한 주목할 만한 뉴런을 표로 정리한다.

예시 필드:

| Layer | Neuron | Hypothesis | Effect | Evidence |
|---|---|---|---|---|
| Lxx | Nxxxx | 특정 의미 가설 | 측정값 | 상세 결과 링크 |

이 표는 실제 실험 데이터가 생성된 후에만 채운다.

뉴런의 기능이 불확실하면 `Unknown` 또는 `Ambiguous`로 표기한다.

민감한 안전성 개입의 정확한 좌표와 강도는 공개 카탈로그에서 제외할 수 있어야 한다.

#### 7. Research Timeline

각 단계의 시작·완료 시점과 Git commit을 연결한다.

#### 8. Reproduction

환경 설정, 모델 revision, 실행 방법, 평가 방법, 실험 설정 파일을 제공한다.

#### 9. Research Log

최신 연구 기록을 README에서 바로 열 수 있도록 링크한다.

#### 10. Limitations & Future Work

결과의 한계와 다음 연구 방향을 분명하게 작성한다.

### README 품질 기준

- 읽기 쉬운 제목 구조
- 지나치게 긴 본문 지양
- 핵심 수치를 표와 그래프로 전달
- 중요한 연구 결과를 README 상단에 노출
- 그림마다 명확한 제목과 설명
- 이미지 깨짐 및 링크 오류 검사
- 모바일 GitHub 화면에서도 읽기 쉽게 구성
- 코드 블록 최소화
- 세부 내용은 별도 문서로 연결

실험이 업데이트될 때마다 README의 기존 수치가 최신 결과와 일치하는지 자동 검증한다.

원본 메트릭 파일에서 표와 그래프를 재생성하는 스크립트를 제공하라.

---

## 4. 기존 프로젝트 및 관련 오픈소스 조사

코딩 전에 기존 연구 프로젝트와 참조 오픈소스를 조사하라.

### 우선 조사할 프로젝트

**A. Lyra neuron-explorer**

https://github.com/lyramakesmusic/neuron-explorer

주요 참고 디렉터리:

- `sweep/`
- `explorer/`
- `research_log.md`
- `NOTES.md`

중점 분석:

- 뉴런 활성 통계
- 뉴런 랭킹
- Steering Sweep
- 뉴런별 결과 저장
- 자동 에이전트 탐색
- 대화형 뉴런 탐색기
- 실험 가설 및 연구 로그 관리

이 프로젝트의 **Steering-first Neuron Exploration** 접근을 중요한 기반으로 삼는다.

Qwen3.5 전용 코드를 Gemma 4에 그대로 적용하지 않는다.

**B. Apple ml-refusal-neurons**

https://github.com/apple-aiml-research/ml-refusal-neurons

주요 참고 디렉터리:

- `hooks/`
- `find_refusal_neurons/`
- `find_concept_neurons/`
- `max_activations/`
- `detection/`
- `eval/`

중점 분석:

- 개별 뉴런 개입
- 중요 뉴런 선택 방법
- Activation Hook
- 정량 평가
- 무작위 대조군
- 일반 성능 보존 측정

**C. EleutherAI Delphi**

https://github.com/EleutherAI/delphi

중점 분석:

- 자동 특징 설명 생성
- AI 기반 설명 가설
- 설명의 평가와 반증
- 자동 해석 결과 저장

SAE Feature와 실제 MLP 뉴런은 다르다는 점을 유지한다.

**D. Neuronpedia interp-engine**

https://docs.neuronpedia.org/open-source/interp-engine

중점 분석:

- 모델별 Activation Hook
- 개입 및 Steering
- 공통 계측 인터페이스
- 실행 백엔드 호환성

**E. NNsight / TransformerLens**

https://nnsight.net/

https://github.com/TransformerLensOrg/TransformerLens

모델 계측, Activation Patching, Causal Intervention 구현에 활용할 수 있는지 확인한다.

### 조사 규칙

실제 저장소 구조를 조회한 뒤 파일 경로를 확정한다.

프로젝트 문서만 보고 기능이 정상적으로 실행된다고 가정하지 않는다.

기존 구현을 활용할 수 있으면 재사용하고, Gemma 4에 맞지 않는 부분만 수정한다.

라이선스를 확인하고 출처를 문서화한다.

조사 결과는 `docs/REFERENCE_AUDIT.md`에 작성한다.

완료 후 GitHub에 첫 번째 연구 로그를 Push한다.

---

## 5. Gemma 4 모델 계측 및 검증

### 5.1 아키텍처 감사

실제 모델 코드와 config에서 다음을 확인한다.

- Transformer 계층 수
- MLP 구조
- Gate / Up / Down Projection
- Intermediate Dimension
- Activation Function
- Normalization
- Attention 구조
- Residual Stream
- Multimodal Backbone
- Chat Template
- Tokenizer
- Prefill / Decode 처리

분석 대상 뉴런은 기본적으로 MLP의 중간 활성 좌표로 정의한다.

이 정의를 연구 문서에 명시한다.

### 5.2 Activation Hook 구현

최소 지원 기능:

- Activation Capture
- Zero Ablation
- Additive Steering
- Multiplicative Steering
- Activation Replacement
- Activation Restoration
- Random Control Intervention

개입 위치가 다르면 실제 효과가 달라질 수 있으므로, 정확한 텐서 위치를 기록한다.

### 5.3 Smoke Test

필수 검증:

- BF16 모델 로딩
- 기본 생성
- Read-only 훅에서 원본 출력 유지
- 개입 수식 검증
- 훅 제거 후 출력 복원
- 동일 입력·조건 재현
- GPU 메모리 사용량 기록
- 모델 revision 및 코드 버전 기록

기본 모델 동작과 훅 정확성이 검증되기 전에는 대규모 탐색을 수행하지 않는다.

---

## 6. 전체 뉴런 활성 통계 수집

초기 목표는 Gemma 4의 전체 MLP 뉴런 활성 패턴을 조사하는 것이다.

다만 모든 뉴런에서 개별 생성 실험을 수행하는 방식은 비효율적이다.

먼저 전체 계층의 Activation Statistics를 수집한다.

### 데이터 범주

- 일반 지식
- 수학·논리
- 프로그래밍
- 과학
- 언어·번역
- 문학
- 사회적 상호작용
- 정체성과 Persona
- 감정 및 말투
- 불확실성
- 과잉거부
- 추론
- 긴 문맥 정보 회수
- 형식 준수
- 무관한 대조 입력

처음에는 Text-only 분석으로 시작한다.

추후 이미지·음성 입력에 대한 멀티모달 연구로 확장할 수 있도록 설계한다.

### 뉴런별 수집 통계

- Activation Mean
- Standard Deviation
- Maximum / Minimum
- Quantiles
- Activation Frequency
- Category Selectivity
- Top Activated Contexts
- Low Activation Contexts
- Token Position
- Cross-Prompt Stability

GPU에서 Streaming Aggregation을 수행한다.

상위 활성 사례와 요약 통계만 장기 저장한다.

실험 데이터는 discovery / validation / test로 분리한다.

동일 원본에서 파생된 프롬프트가 서로 다른 분할에 섞이지 않게 한다.

### 1차 시각화

- 계층별 Activation Heatmap
- 뉴런 활성값 분포
- 의미 범주별 평균 활성값
- 상위 선택성 뉴런 분포

실제 수집이 완료되면 `figures/`에 생성하고 README에서 표시한다.

이 단계에서 GitHub CP3을 수행한다.

---

## 7. Steering-first 자동 뉴런 탐색

후보 뉴런을 선정한 뒤 실제 개입 실험을 수행한다.

### 후보 선정 기준

- 특정 범주에서의 Activation Selectivity
- 대조 프롬프트 간 차이
- 활성값의 안정성
- 출력에 대한 인과적 영향
- 다른 뉴런과 구별되는 효과
- 예상하지 못한 현상

### 개입 조건

- 원본 모델
- 양의 방향 개입
- 음의 방향 개입
- Zero Ablation
- Activation Restoration
- Random Neuron Control
- Random Direction Control

### 중요한 검증

뉴런의 활성값을 크게 바꾸면 모델의 출력이 전반적으로 손상될 수 있다.

이를 특정 뉴런의 기능으로 잘못 해석하지 마라.

자연 활성값의 범위, 개입 Norm, 출력 분포 변화량과 의미적 변화를 함께 측정한다.

Steering 강도별 변화를 정량화한다.

실험 결과는 다음을 구분하여 저장한다.

- 목표한 행동 변화
- 예상하지 못한 행동 변화
- 출력 품질 손상
- 반복 또는 붕괴
- 무의미한 출력
- 변화 없음

유해한 생성물이나 안전성 우회에 직접 사용될 수 있는 최적 설정은 공개 저장소에 포함하지 않는다.

중간 결과와 그래프를 README에 반영한 후 CP4를 수행한다.

---

## 8. Codex 자율 뉴런 해석 시스템

각 뉴런에 대해 다음 루프를 자동 실행한다.

### Phase 1 — Evidence Collection

- 상위 활성 문맥
- 하위 활성 문맥
- 개입 전후 출력
- 토큰 확률 변화
- 유사 뉴런
- 대조군 결과

### Phase 2 — Hypothesis Generation

Codex가 뉴런의 기능을 설명하는 가설을 생성한다.

한 뉴런에 대해 복수의 경쟁 가설을 제시할 수 있다.

단순한 활성값 상관관계와 인과적 영향은 구분한다.

### Phase 3 — Automated Falsification

Codex가 해당 가설을 반증할 수 있는 새로운 프롬프트를 생성한다.

- Positive Control
- Negative Control
- Near-miss Example
- Paraphrase
- Out-of-distribution Example

가설 생성 데이터와 독립 검증 데이터는 분리한다.

### Phase 4 — Causal Validation

개입 결과와 대조군을 분석한다.

효과의 크기뿐 아니라 선택성과 일반 성능 손상을 평가한다.

### Phase 5 — Catalog Update

각 뉴런을 다음 상태 중 하나로 분류한다.

- `candidate`
- `correlated`
- `causally_supported`
- `replicated`
- `ambiguous`
- `falsified`
- `unknown`

뉴런 설명에 대한 모델의 확신만으로 검증 상태를 높이지 않는다.

### Phase 6 — Autonomous Next Experiment

검증 결과를 바탕으로 다음 실험을 계획한다.

새 가설, 경쟁 설명, 반례 가능성을 고려한다.

기존 실험을 의미 없이 반복하지 않는다.

---

## 9. Neuron Catalog 및 Explorer

연구 결과를 검색하고 비교할 수 있는 구조화된 카탈로그를 만든다.

### 뉴런별 필수 정보

- Model Revision
- Layer ID
- Neuron ID
- Activation Statistics
- Hypothesized Function
- Evidence Examples
- Steering Effects
- Negative Controls
- Generalization
- Confidence / Uncertainty
- Validation Status
- Related Neurons
- Experiment Run IDs

### 인터랙티브 Explorer

기존 Lyra Explorer를 참고하여 다음 기능을 구현한다.

- 계층별 뉴런 분포
- 뉴런 검색
- 의미 범주 검색
- 활성값 시각화
- 뉴런별 상세 정보
- 원본 vs Steering 비교
- 유사 뉴런 조회
- 실험 로그 탐색
- Codex 가설 및 검증 기록 조회

가장 먼저 CLI와 데이터 저장 파이프라인을 완성한다.

Explorer의 UI를 과도하게 개발하느라 실제 실험이 지연되지 않도록 한다.

공개 카탈로그와 로컬 전체 카탈로그를 분리할 수 있게 한다.

---

## 10. 자동 연구주제 발굴

개별 뉴런 해석뿐 아니라 새로운 현상과 메커니즘을 발견하는 것을 목표로 한다.

연구 후보는 실제 관찰 결과로부터 도출한다.

관심 연구 방향:

1. 단일 뉴런이 복합적 행동을 유도하는 메커니즘
2. 자연 활성값과 인과적 중요도의 관계
3. 뉴런별 기능의 다의성
4. 계층에 따른 뉴런 기능 변화
5. 뉴런 조작과 Residual Vector Steering의 차이
6. 뉴런 기능의 문맥 의존성
7. 뉴런 간 상호작용
8. 소수 뉴런의 행동 제어 효과
9. 자연 활성 범위와 과도한 교란 사이의 차이
10. 텍스트와 멀티모달 표현의 공유 가능성

이 중 어느 것도 사전에 논문 결론으로 고정하지 않는다.

### 논문 후보 평가

다음 기준으로 평가한다.

- Novelty
- Related Work Coverage
- Experimental Evidence
- Causal Validity
- Reproducibility
- Scientific Significance
- Additional Experiment Cost
- Publication Potential

기존 연구에서 이미 알려진 현상이라면 새로운 발견으로 포장하지 않는다.

실패하거나 반증된 가설도 이후 연구에 도움이 된다면 기록한다.

---

## 11. 프로젝트 구조

기존 관련 프로젝트가 있으면 그 구조를 우선한다.

공식 GitHub 저장소와 로컬 연구 디렉터리를 동기화한다.

초기 구조 예시:

```text
Gemma4_neuron_LAB/
├── README.md
├── AGENTS.md
├── .gitignore
├── pyproject.toml
├── configs/
│   ├── model.yaml
│   ├── smoke.yaml
│   ├── pilot.yaml
│   └── survey.yaml
├── src/
│   └── neuron_lab/
│       ├── model_adapter/
│       ├── hooks/
│       ├── activation/
│       ├── discovery/
│       ├── steering/
│       ├── interpretation/
│       ├── validation/
│       ├── agents/
│       ├── storage/
│       └── evaluation/
├── explorer/
├── scripts/
├── tests/
├── docs/
│   ├── REFERENCE_AUDIT.md
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   ├── EXPERIMENT_PLAN.md
│   └── RESEARCH_LOG.md
├── reports/
│   ├── INITIAL_FINDINGS_KO.md
│   └── RESEARCH_SUMMARY_KO.md
├── figures/
├── logs/
│   └── summary/
├── catalog/
│   └── public/
├── data/                # Local / gitignored
├── runs/                # Local / gitignored
├── checkpoints/         # Local / gitignored
└── references/          # External projects / license-aware
```

GitHub에는 재현에 필요한 최소 코드와 연구 결과 요약만 올린다.

대용량 데이터 저장소와 공개 연구 기록 저장소를 분리한다.

---

## 12. 실험 실행 예산 및 자동화

처음부터 전체 뉴런에 대해 생성 실험을 수행하지 않는다.

우선 다음 순서로 실행한다.

1. Small Smoke Test
2. Selected Layer Pilot
3. Full Layer Activation Statistics
4. Candidate Selection
5. Steering Sweep
6. Autonomous Explanation
7. Independent Validation
8. Expanded Discovery
9. Research Synthesis

### 최적화

- GPU에서 가능한 통계 집계 수행
- Streaming Statistics
- Top-K Activation Buffer
- Batch Processing
- Activation Cache 제한
- 불필요한 전체 텐서 저장 방지
- 실험별 메모리 사용량 기록
- 실패한 Run의 재개 지원

연구 단계별 최대 GPU 사용 시간, 샘플 수, 디스크 용량은 설정 파일에 둔다.

대규모 실험 전 Pilot에서 실제 소모량을 계측해 예산을 결정한다.

현재 다른 프로세스가 GPU를 사용하는 경우 기존 작업을 임의로 종료하지 않는다.

디스크 공간 부족이나 GPU 오류가 발생하면 작업 규모를 줄이거나 안전하게 중단한다.

NVIDIA 드라이버나 운영체제 설정을 임의로 변경하지 않는다.

---

## 13. 품질·재현성·과학적 검증

### 코드 검증

- Read-only 훅의 출력 불변성
- 모델 출력 동등성
- 개입 좌표 정확성
- 개입 수식의 테스트
- 훅 제거 후 원상복구
- 저장 데이터 재로딩
- 동일 Run 재현
- Git에 저장한 설정으로의 재실행 가능성

### 실험 검증

- Discovery / Validation / Test 분리
- Random Control
- Matched Intervention
- Multiple Seeds
- Confidence Intervals
- Counterexamples
- Negative Results
- Alternative Explanations

### 금지 사항

- 실제 실험 결과 날조
- 데이터 누수
- 사후 테스트셋 최적화
- 실패 결과 은폐
- 가상 메트릭 게시
- 높은 상관관계를 인과성으로 과장
- 강한 출력 교란을 특정 기능으로 단정
- 결과에 맞추어 가설을 사후 변경하고 이를 숨기는 행위

검증 상태는 `PASS`, `FAIL`, `NOT_EVALUATED`로 명확하게 기록한다.

---

## 14. 자율 연구 루프와 GitHub 동기화

반복 연구 루프는 다음과 같다.

```text
Collect Activations
        ↓
Rank Candidate Neurons
        ↓
Run Causal Interventions
        ↓
Generate Hypotheses
        ↓
Design Counterexamples
        ↓
Execute Validation
        ↓
Update Neuron Catalog
        ↓
Generate Figures & Logs
        ↓
Update README
        ↓
Commit & Push to GitHub
        ↓
Analyze Remaining Uncertainty
        ↓
Select Next Experiment
        ↺
```

**GitHub 업로드는 실험이 끝난 뒤 별도로 수행하는 문서 작업이 아니라, 연구 루프에 포함된 필수 단계다.**

단, GitHub 동기화 장애로 전체 실험이 중단되지 않도록 로컬 상태를 보존한다.

각 Run은 고유 ID를 가진다.

매 단계에서 다음을 기록한다.

- 실행된 실험
- 사용한 모델과 설정
- 소요 시간
- 결과 수치
- 생성된 파일
- 성공/실패 여부
- 다음 실행 후보
- Git commit SHA 및 Push 상태

GitHub에 게시된 README의 수치와 로컬 보고서의 수치가 다르면 자동 점검하도록 한다.

작업이 재개되면 마지막 성공 체크포인트에서 이어간다.

---

## 15. 단계별 실제 수행 지시

### Phase A — Audit & Initialization

- 현재 디렉터리와 기존 코드 검사
- GitHub 저장소 확인
- 참조 프로젝트 조사
- Gemma 4 아키텍처 검토
- 초기 프로젝트 구성
- README 작성
- 초기 GitHub Push

### Phase B — Model & Hook Validation

- BF16 모델 로딩
- Activation Hook 구현
- 뉴런 개입 수식 검증
- Smoke Test
- 측정 결과 및 검증 로그 업로드

### Phase C — Discovery Pilot

- 소규모 뉴런 활성 통계
- 후보 선별
- 최초 Steering Sweep
- 결과 시각화
- README 갱신 및 Push

### Phase D — Autonomous Interpretation

- 뉴런별 가설 생성
- 반례 자동 생성
- 독립 검증
- 가설 분류
- Neuron Catalog 업로드

### Phase E — Expanded Experiments

- 전체 계층 활성 통계
- 후보 뉴런 확장 탐색
- 계층별 비교
- 뉴런 간 관계 분석
- 주기적인 결과 업로드

### Phase F — Neuron Explorer

- 연구용 UI
- 실험 결과 탐색
- Live Steering
- 에이전트 연구 인터페이스
- README 사용법 업데이트

### Phase G — Research Synthesis

- 주요 발견 정리
- 경쟁 가설 검토
- 반례 및 실패 결과 분석
- 기존 연구 비교
- 논문 후보 우선순위
- 최종 보고서 및 시각화 정리
- GitHub 최종 Push

각 단계가 검증되면 다음 단계로 진행한다.

환경이 허용하면 독립적인 작업에 멀티에이전트를 활용한다.

GPU 자원 충돌, Git 변경 충돌, 평가셋 누수가 발생하지 않도록 작업을 분리한다.

---

## 16. 최종 보고 규칙

`reports/RESEARCH_SUMMARY_KO.md`를 최종 연구 보고서로 사용한다.

다음 정보를 포함한다.

1. 실제 구현 완료 항목
2. 성공적으로 실행된 실험
3. 실패한 실험과 실패 이유
4. 발견한 뉴런과 기능 가설
5. 정량적 Activation 및 Steering 결과
6. 무작위 대조군과의 비교
7. 독립 검증 결과
8. 예상하지 못한 현상
9. 반증된 가설
10. 다음 연구 후보
11. 관련 선행연구
12. 남아 있는 한계
13. 실제 사용된 계산 자원
14. 재현 방법
15. GitHub 커밋과 결과 링크

README에는 가장 중요한 결과만 간결하게 요약한다.

상세 보고서는 별도의 문서로 연결한다.

**README를 읽는 것만으로 연구 진행 상황을 파악할 수 있고, 상세 보고서를 열면 그 결과를 검증할 수 있어야 한다.**

실험을 전혀 수행하지 못한 단계는 구현 완료로 표시하지 않는다.

GitHub Push에 실패하면 성공했다고 주장하지 말고 로컬 커밋 및 실패 원인을 기록한다.

---

## 17. 최종 지시 — 지금부터 실제 작업 수행

지금부터 Autonomous Gemma 4 Neuron Lab 구축을 시작하라.

**계획서 작성으로 종료하지 마라.**

현재 프로젝트와 GitHub 저장소를 감사하고, 공개 오픈소스의 실제 구조를 분석한 다음, 가장 단순하고 정확한 Gemma 4 BF16 계측 파이프라인을 구현하라.

GPU 접근이 가능하다면 실제 모델을 실행하고 Pilot 실험을 수행하라.

결과를 분석하여 다음 실험을 스스로 설계하라.

작업 도중 중요한 결과가 나오면 GitHub에 중간 로그와 그래프를 올리고 README를 갱신하라.

모든 원본 파일을 업로드할 필요는 없다. 공개 연구 기록으로 가치 있는 결과와 재현에 필요한 최소 자료를 중심으로 업로드하라.

실험에서 새로운 사실을 발견하면 기존 가설을 수정하고 추가 검증을 수행하라.

자율적으로 진행하되, 명시한 계산·저장·외부 서비스 예산과 권한의 범위를 지켜라.

막힌 작업이 있으면 가능한 독립 작업을 먼저 완료하고 차단 원인을 기록하라.

**연구의 최종 목적은 많은 뉴런을 임의로 분류하는 것이 아니라, 뉴런의 실제 기능과 인과적 영향에 대해 검증 가능한 증거를 발견하는 것이다.**

그 과정과 결과가 GitHub의 README, 연구 로그, 그래프, 뉴런 카탈로그에 지속적으로 반영되도록 하라.

**Start with repository audit, architecture verification, and the first reproducible GPU experiment.**
