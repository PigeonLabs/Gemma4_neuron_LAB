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
