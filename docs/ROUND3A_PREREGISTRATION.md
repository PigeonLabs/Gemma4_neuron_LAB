# R3-A: 내용·답변 기호·규칙 수행 효과 분리 — 실행 전 등록

사용자가 추천 1번의 실행을 요청했다. 이전 R2 확증을 보고 설계한 **새 연구**이며, 원 R2 가설·분석·원시는 변경하지 않는다. 당시 제안은 `configs/round3_plan.yaml`에 보존한다. 이번 구체적 실행 규칙과 예산의 원본은 `configs/round3a.yaml`이다. 새 단순 생성 자료만 사용하며, R2의 원본·확증을 새 검정에 재사용하지 않는다. R3-B는 실행 범위가 아니다.

## P0를 먼저 측정하는 이유

작은 native 좌표 효과를 BF16의 출력 반올림과 구분해야 한다. 서로 다른 새8 family(두 과제×두 난이도 각2개), 규칙2×라벨배정2의32변형에서 clean+고정4좌표의 ±0.5SD를 측정한다. SD는 P0의 BF16 clean 좌표 통계로 정하고 두 dtype에서 같은 nominal 절대량을 쓴다. Native 좌표는 `MLP.down_proj` 입력이며 개입 위치는 마지막 prefill chat-template 토큰이다. 각 run은 revision·seed·source hash·장치·dtype·원시 SHA·자원 사용을 기록한다.

동일한 저장 가중치를 BF16으로 로딩하고, 원래 smoke28검사와 정렬 hook34검사를 통과한 뒤 BF16을 측정한다. **같은 프로세스에서 같은 파라미터/부동 버퍼 값을 FP32로 올린 후** FP32를 측정한다. 별도 FP32 가중치를 다운로드하지 않는다. 두 dtype마다 read-only·동일 forward 좌표 복원·훅 제거의 logits exact를 검사한다. 이는 native FP32 학습 가중치에 대한 비교가 아니며, BF16에서 저장된 값의 계산 정밀도 비교다. 실제 전달 delta와 post-norm geometry도 기록한다.

P0의 부호 있는 효과는 고정 내용축 m=c(logit_first−logit_second)의 d=(m_plus−m_minus)/2다. 각 좌표의32개 paired 효과에서 아래 **모든** 조건이 충족되어야 수치상 eligible이다.

- 모든 기술 동등성/finite 검사 통과, 양 dtype에서 actual nonzero delta 비율≥95%.
- FP32 |d|≥0.02 nats인 입력이 최소8개. 그 입력에서 BF16/FP32 부호 일치율≥80%.
- 전체32개 signed d의 Spearman≥0.70. 평균 절대 오차≤max(0.05 nats, FP32 평균 |d|의50%).

이는 결과를 보기 전에 작동 가능한 숫자로 정한 **측정 품질 기준**이며 통계적 기능 검정이 아니다. 개별32변형을32개의 독립 family로 검정하지 않는다. 최소1개 좌표가 통과해야 discovery로 진행하고, primary는 통과 좌표 중에서만 고른다. 모두 실패하면 `STOPPED_AT_NUMERICAL_QUALIFICATION`로 종료한다. Discovery/calibration/confirmation은 `NOT_EVALUATED`다. 실패 뒤 dose를 키우거나 기준을 바꾸어 같은 연구를 재개하지 않는다. 해당 위치·dose·과제에서 BF16으로 식별하기 어렵다는 뜻이며 뉴런의 기능 부재를 뜻하지 않는다.

P0 기본576 forward, 검증 포함 cap1,000회·1,800초, raw cap2GB다. 전체 모델은 하나만 실행하며 FP32 변환 후 메모리가80GB를 넘거나 여유4GB 미만이면 중단한다. 다른 사용자의 프로세스·드라이버·OS 설정은 건드리지 않는다.

## P0 통과 뒤의 독립 R3-A

사전 생성한 discovery16/calibration24/confirmation64 family에서 2규칙×2라벨배정×AB/CD 총8변형을 같은 family 안에 둔다. Reverse/sort 및 제공 사전 번역/copy의 두 과제와 easy/hard를 모두 유지해 recipe를 사후로 거르지 않는다. AB/CD는 각 분할에서 모두 측정하므로 미관측 답변 토큰 전이라고 부르지 않는다.

Discovery에서는 수치 eligible 후보 중 candidate−같은 층 native 패널평균의 C/G/L 절댓값이 가장 큰 후보1개를 고른다. 두 코드 평균에 대해 max(0.05 nats, 해당 후보 P0 signed 오류99백분위의3배)를 넘는 신호가 없으면 등록 gate 실패로 종료한다. Discovery doses는 discovery clean SD로 정한다. 새 calibration은 별도의 자연 SD·대조·전역 geometry 배율을 정하고 primary 후보/규칙을 변경하지 않는다. Calibration 완료 후 분석식·β·대조·dose·SESOI를 commit하고 confirmation을 시작한다.

고정 내용축의 C=mean(d), G=mean(r*d), L=mean(c*d), J=mean(r*c*d)를 분해한다. G는 단순 confidence scaling도 흉내 낼 수 있어 calibration에서 d=C+G*r+L*c+J*r*c+β*m_clean을 좌표/코드별 OLS로 적합하고 β를 고정한다. 1차량은 confidence-adjusted C/G/L의 후보−native6평균 family 대비이며 Holm3보정을 쓴다. 방향2개는 보조 대조다. 작은 native 패널의 순위로 뉴런 모집단 p값을 만들지 않는다.

규칙 대 확신 해석에는 clean 정확도≥.75, calibration/confirmation 각각 오답 변형을 포함한 source family≥8, full-rank 및 표준화 condition number≤50이 필요하다. 이를 충족하지 못하면 규칙 대 확신은 `NOT_IDENTIFIABLE`이다. 결과를 보고 확증 입력을 거르지 않는다. AB/CD의 방향 재현, 고정 대조 대비 실용 크기, family CI 및 Holm을 함께 평가한다. 정답 선택 변화가 없으면 logit 효과까지만 해석한다.

전체 R3-A는 P0 별도, 최대21,000 forward·7,200초다. 구체적인 주 실행 코드는 P0 통과 후 discovery를 시작하기 전에 추가로 등록·commit한다. P0 코드·수치 판정과 P0 원시 자료를 소급 변경하지 않는다. 공개는 요약·그림·코드·설정의 명시적 허용목록과 비밀정보/크기 검사 후 수행한다. 전체 prompt manifest·activation·logit·생성문은 로컬에만 둔다.
