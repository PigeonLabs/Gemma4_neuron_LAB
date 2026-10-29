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

## 2026-10-29 KST · CP4 · 첫 Steering Sweep

- 재시작한 worker에서 BF16 smoke 28/28을 재검증했다.
- `20261029T132730_steering`: Primary 후보 H001/H002, discovery 15 prompts, 630개 조건 실행, 약 97.03초, PASS. ±0.5/±1/±2 SD, zero, 손상 후 복원, activation/projected norm 대조군 각각 seed3을 측정했다.
- H001 +1SD mean KL=0.00018743 (prompt bootstrap95%CI [0.00001293,0.00046780]), H002=0.00014524 ([0.00000343,0.00041060]). 이는 next-token 분포 변화이지 번역·불확실성 기능의 인과 증거가 아니다.
- ±1SD에서 next-token top1 변화는 15개 입력 모두 0이었다. 소수 사전 선택 생성 입력에서도 해당 조건의 greedy 출력 변화가 없었다. 강한 행동 제어는 관찰하지 못했다.
- 모든 정렬 restoration 조건의 clean-baseline logits가 정확히 복원됐다.
- 다음: 고정 가설 4개의 validation45+별도 반례40 실행, 이후 untouched test45 실행. 이 결과를 보고 test 조건을 바꾸지 않는다.

## 2026-10-29 KST · CP5 · 독립 Validation과 반례

- `20261029T133451_validation`: corpus45 + 새로운 반례40 =85 prompts, 3740조건 평가, 약381.92초, PASS. 반례의 관련 paraphrase는 독립 corpus CI에서 제외했다.
- 동결4후보의 target3/rest42 AUC: H001/H002/H003=1.0, H004=0.976190. 2000회 permutation의 Holm보정 p는4후보 모두약0.001999였다. 737280 discovery 좌표 전체의 유의성 주장은 아니다.
- +1SD KL의 대조군 대비 category interaction은4후보 모두95%CI에0을 포함했다. 등록한 distribution-selectivity 기준은 validation에서 지지되지 않았다. 의미 기능은 모두 ambiguous로 유지했다.
- 반례는 후보별10문장/8source family다. family AUC H001=.875,H002=1,H003=.8125,H004=.9375, Holm보정 p는 모두>.05로 표본이 작다. 이를 의미 기능 확정으로 해석하지 않는다.
- 그림 재계산 검사가 초기에는 관련 반례를 corpus 집계에 포함해 scope 불일치를 검출했다. validator를 명시된 role=corpus 범위로 수정했고, 원시측정·요약은 변경하지 않았다. 45+40 분모 검사와 raw mean 검사를 통과했다.
- 다음: untouched test45, 이어서 별도로 동결한 소규모 utility 정답 평가. 원래 test 설정은 변경하지 않는다.

## 2026-10-29 KST · CP6 · Untouched Test와 추가 Utility

- `20261029T135014_expanded`: untouched corpus test45, 고정4후보, 3060조건, 약308.48초, PASS. 원래 test 조건과 가설은 변경하지 않았다.
- 네 후보의 target3/rest42 활성 AUC는 모두1.0이었다. +1SD의 norm-matched random-direction 대비 category interaction은 네 후보 모두95%CI에0을 포함했다. validation과 test를 모두 통과해야 한다는 등록 기준을 충족하지 못했다.
- Test에서 ±1SD의 top1 변화는 후보·조건에 따라0 또는1/45였다. 첫 discovery sweep의 무변화를 모든 입력으로 일반화하지 않는다.
- 추가 utility는 validation 이후 별도 설계하고 test 시작 전13:48:26KST에16문제와 정답을 동결했다. `20261029T135550_utility`: general8+translation8, baseline 공유16회+개입256회=실제272회 생성, 약39.95초, PASS. ±1SD/zero/projected random direction의 모든4후보에서16/16 정답·출력변화0·paired정답률Δ0이었다. 작은 편의표본의 관찰이며 일반 성능 무손상이나 의미 인과 기능의 증거가 아니다.
- 공개 frozen evidence4개만으로 가중치·private전체후보·파생분석을 제외한 /tmp 환경에서 모든 완료단계의 AUC/CI/반례/norm/프로필 재계산이 일치했다. 좌표·SD·revision·summary hash 변조 거부도 통과했다.
- sourcegroup32개 반례family와 원래150개 corpus 분할을 구별했다. 카탈로그4개는 모두 ambiguous다.
- 다음: 선행연구와 경쟁 설명을 반영한 연구 후보·한계 종합, 최종보고서와 공개상태 검사.

## 2026-10-29 KST · CP7/CP8 · 연구 종합·최종 검증

- 원전 논문5편을 확인해 native단일뉴런·SAE·norm대조·상관/인과 선행기법과의 관계를 정리했다. 후속3연구의 신규성은 NOT_EVALUATED이며 새발견·논문성과로 포장하지 않았다.
- 실제 test측정시간/조건 수를 기준으로 토큰위치대조·어휘길이대조·후기정규화 연구의 예상비용을 계산했다. 추정값은 실제 미래소모량과 구별한다.
- 세단계 steering/validation/test 총7430조건 평가, 별도 utility272회 생성. 기능가설4개는 모두 ambiguous다.
- 그림8개 실제PNG를 모두 시각검사했다. Source/SVG/preview hash와 QA 상태를 보존했다.
- 최종 artifact 검사19개 PASS: 공개 허용목록·비밀/크기, finiteJSON/SVG, Markdown링크, 등록hash, corpus분할, 모든 rawartifact, 분석/README/보고서 최신성, 공개subset CPU재계산, utilityconfig, 최종완결성·검사중불변성.
- CPU 회귀시험104개 PASS. 다른GPU작업·OS·드라이버는 변경하지 않았다.
- Explorer localhost8501 및 검증된GPUworker를 사용 가능 상태로 남긴다. Worker는 새 연구를 반복하지 않으며3600초 유휴 후 종료한다.
- 최종보고서: reports/RESEARCH_SUMMARY_KO.md. GitHub반영확인은 publication receipt와검증된SHA를 따른다.
