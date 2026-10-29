# Research Log

## 2026-10-29 KST · CP0 · Audit & Initialization

- 작업 지시: `MASTER.md`를 읽고 실제 실험을 시작했다.
- 로컬에는 MASTER.md만 있으며 기존 유효한 Git checkout은 없다. 루트 `.git`는 환경이 제공하는 읽기 전용 빈 디렉터리다. 이를 변경하지 않고 별도 로컬 Git 메타데이터와 연결된 GitHub API로 동기화한다.
- GitHub connector로 `PigeonLabs/Gemma4_neuron_LAB`를 조회했다. 공개, size=0, default branch=main, branches=[]이며 push 권한이 있다. 기존 코드는 없다.
- Ubuntu 24.04.5 LTS, Ryzen 5 5600X, RAM 약 62GiB를 확인했다. 프로젝트 볼륨 여유 약 436GiB, 시스템 볼륨 약 70GiB다.
- 샌드박스 안의 nvidia-smi는 드라이버 통신 실패였다. 승인된 외부 읽기 전용 조회는 성공했다: RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887MiB total, 95,576MiB free, driver 580.178.04, compute app 없음. OS나 드라이버는 변경하지 않았다.
- 기본 Python과 번들 Python에 torch/transformers가 없다. 프로젝트 `.venv`를 생성하고 의존성을 설치 중이다. HF 캐시와 가중치는 프로젝트 볼륨에 배치한다.
- 공식 HF metadata에서 모델 공개 접근(gated=false)과 revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`을 확인했다. config 및 upstream 코드를 독립 감사 중이다.
- 참조 프로젝트 조사와 아키텍처 조사는 분리된 에이전트로 수행한다. GPU 실험은 메인 프로세스 하나만 담당한다.
- 실제 BF16 모델 실행, neuron 통계, steering 결과는 아직 `NOT_EVALUATED`다.
- 다음: 의존성 검증 → 모델 다운로드·BF16 로딩 → 정확한 런타임 hook 좌표 확인 → CP1/CP2.

### GitHub 동기화

초기 게시 시도와 검증된 SHA는 `logs/summary/publication.json`에 기록한다. API로 게시한 커밋도 동일한 공개 Git 기록에 남는다.

## 2026-10-29 KST · CP1/CP2 · 실제 BF16 로딩·훅 검증

- 공식 가중치 23,951,746,871 bytes와 추론 파일 7개를 약 652.4초에 다운로드했다. 파일 접근 제한이나 HF 인증키는 필요하지 않았다.
- PyTorch 2.15.0+cu132와 driver 580.178.04에서 sm_120 GPU BF16 행렬 연산이 성공했다. Transformers 5.21.0 순수 HF eager를 사용했다.
- 실제 모델 로딩 약 6.56초. 모든 파라미터 BF16, 11,959,730,176개 파라미터. 48개 down_proj의 입력 폭 15,360으로 737,280개 좌표를 런타임에서 확정했다.
- 실제 3개 무해한 입력의 smoke 28/28 PASS: 48계층 shape와 finite, read-only logits exact, greedy generation exact, 반복 exact, BF16 좌표 개입 수식, zero 손상 후 baseline 좌표 복원, 개입 hook 제거 후 baseline exact, handle 제거. 약 6.15초, peak allocated 23,978,810,368 bytes. 이는 의미 기능 검증이 아니다.
- 핵심 CPU 단위 테스트 95개 PASS. source/config hash와 환경을 각 Run에 보존했다.
- Run: `20261029T131735_smoke`, [실제 수치](../logs/summary/smoke.json), [로딩 감사](../logs/summary/model_load.json).
- 다음: 15개 입력/4계층 pilot → 통과 시 60개 discovery 입력/48계층 survey.

## 2026-10-29 KST · CP3 · 全 계층 Discovery

- Pilot `20261029T131907_pilot`: 15개 입력, 4계층, 61,440좌표, 사용자 본문 275 tokens/계층. PASS.
- Survey `20261029T132008_survey`: 60개 입력(15범주×4), 48계층, 737,280좌표, 사용자 본문 1,022 tokens/계층, 약 17.27초. PASS.
- 모든 좌표의 moment 통계를 Parquet로, bounded top/low와 범주 배열을 로컬 NPZ로 보존했다. 전체 activation tensor는 저장하지 않았다. 64-token reservoir quantile은 근사치다.
- 상위 후보는 번역 범주에 집중했다. L15N9637 discovery z=3.9922, 범주 평균 8.8053. 이 점수는 검정통계량이 아니며 어휘·문자·과제 유형을 분리하지 못한다.
- Codex가 실제 상·하위 문맥을 읽고 4개 경쟁 가설을 `configs/hypotheses.json`으로 동결했다. Primary H001=L15N9637(번역), H002=L22N7859(불확실성). 확장 H003=L26N719(번역), H004=L35N990(프로그래밍 vs Python 어휘).
- 40개의 새 validation 반례를 작성했다. positive/negative/near-miss/paraphrase/OOD를 분리했고 paraphrase family는 같은 validation split에 둔다. 아직 validation/test 측정은 수행하지 않았다.
- 실제 discovery 요약에서 그림 4개를 생성했다. 최초 pilot 그림 2개를 시각 검사했고, survey 갱신 그림은 별도 점검한다.
- 개입은 마지막 chat-template token에 적용되므로 본문 평균 활성과 위치가 다르다. 대조군은 1SD에서만 norm 비교한다. 이를 방법론에 추가했다.
