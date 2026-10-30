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

## 2026-10-29 — Round 2 신규성 중심 재분석·실험 설계

- 사용자 요청에 따라 실제 사이드바 GPT Pro Deep research와 Claude Opus 5.5 Extra Research의 완료 보고서를 읽고, 새 진단·가까운 원전을 전달해 교차 검토했다. AI 응답은 제안 자료이며 측정 증거로 사용하지 않았다. Claude의 상세 파일 접근 제한은 보고서에 명시했다.
- 기존 raw SHA와 행 수를 검증한 CPU 재집계에서 test 다음 토큰 변경 39개 조건행 모두가 원본 1개에 집중됨을 확인했다. 후보/무작위 대조가 함께 포함돼 있다. Margin·의미 정답 변화는 저장되지 않아 near-tie나 정답 전환으로 단정하지 않았다.
- 4v4 반례 family의 70개 라벨 할당을 추가 열거했다. 완벽 AUC의 exact 양측 최소 p=.02857, 4가설 Holm 첫 단계=.11429다. 기존 2,000회 Monte Carlo 결과는 보존했다.
- 검출/작동 해리·위치 일치·정규화 희석·단순 gate/up 분해는 가까운 선행과 중복돼 중심 기여에서 내렸다. 내용 편향/확신 증폭/규칙 수행 이득의 구별, 정보 부재×응답 규칙의 branch 인과 분업을 1·2순위로 설계했다.
- 보고서: [NOVELTY_IDEAS_ROUND2_KO.md](../reports/NOVELTY_IDEAS_ROUND2_KO.md). CPU 재집계: [round2_diagnostics.json](../logs/summary/round2_diagnostics.json). 계획 초안: [round2_plan.yaml](../configs/round2_plan.yaml).
- 새 GPU 추론·훈련은 실행하지 않았다. Round 2는 DRAFT_NOT_EXECUTED / NOT_FROZEN이며, 원본별 새 분할·실제 코드·예산·가설을 고정한 뒤 별도로 실행한다. 기존 test는 이번 가설의 탐색 근거가 되었으므로 새 confirmation에 재사용하지 않는다.

## 2026-10-29 · 추천 5번 R2-E / 6번 R2-F 실제 실행

- 사용자가 지정한 실제 GPT Pro Deep research와 Claude Opus 5.5 Extra Research의 E/F 설계 보고서가 모두 완료된 것을 확인하고 읽었다. 설계 채택·수정점은 [실행 전 등록](ROUND2_EF_PREREGISTRATION.md)에 기록했다. 위의 아이디어 검토 entry는 실행 이전 시점의 기록이며, 여기서는 사용자가 고른 5·6번만 새 연구로 수행했다. A/B 초안은 실행하지 않았다.
- 사전등록 로컬 commit `1a59405`, 자연 통계·예측 회귀 고정 commit `07535a0`. Manifest 512개 입력은 생성 원본 family와 A/B 순서를 보존하고 R1과 별도 분할했다. 두 번의 모델 로딩마다 BF16 smoke 28개·새 정렬 훅 34개 검사 전부 PASS. 기존 R1 주요 파일 8개 hash 불변.
- `20261029T154249_round2_ef_calibration`: E 32/F 16 family, E 2,112 + F 896 조건 행, 검증·geometry 보정 포함 3,296 forward. 본 run 209.94초, 검증 포함 budget 시간 219.06초, peak reserved 24.411GB. SD·random 좌표·H003 지시별 자연 median은 행동 검색 없이 결정했다. E 회귀만 calibration 반응으로 적합했다.
- `20261029T154727_round2_ef_confirmation`: E/F 각각 64 family, E 4,224 + F 3,584 조건 행, 8,272 forward, 본 run 543.65초, peak reserved 24.023GB. 전체 10,816 조건 행·11,568 forward로 등록 예산 이내 완료했다. 임계값·후보·분할·분석식 변경 없음.
- E: 고정 기본 모형 R²=-.06306, 대조 S 추가 R²=.14127, ΔR²=.20433, 95% family CI [.08981,.29211], Holm p=.0019998. MAE .34281→.32149. 상위 10%·20% family 제거 시 ΔR²=.25847/.29645. 등록 범위 내 지지지만 실제 개입 geometry와 새 입력의 대조 probe가 필요하다. Zero-probe 예측 개선·보편 공통 원인·의미 행동 개선은 입증하지 않았다. 확증 E top-1 변경은 모든 arm에서 0이었다.
- E 사후 기술 진단: MSE 감소 약91.92%가 문자열 과제에서 발생했다. Task 평균 제거 후 후보–대조 log 반응 상관 .60199. 1차 검정과 분리한 [사후 진단](../logs/summary/round2_ef_diagnostics.json)이며 추가 확증 주장으로 사용하지 않았다.
- F: 자연 counterfactual 손상 .01139 nats, CI [-.00788,.03145]로 하한>.10 gate 실패. I=.000355, CI [-.02847,.02873], Holm p=.98290. 모든 14개 조건을 끝까지 측정했지만 rescue fraction은 계산하지 않고 **NOT_IDENTIFIABLE**로 판정했다. 동등성·대체 불가능·기능 부재의 증거로 표현하지 않았다. 기술 복원 max logit error=0.
- E native random norm 오차 ±10% 이내85.06%, residual direction100%. 후보 활성 calibration min–max 이탈19.24%. F native/candidate R norm 비율0.5–2 이내94.92%. 제한된 random 좌표·BF16 해상도·task shell 조건부 일반화의 한계를 보고했다.
- [실측 보고서](../reports/ROUND2_EF_RESULTS_KO.md), [요약 JSON](../logs/summary/round2_ef_analysis.json), [공개 family 재계산](../scripts/replay_round2_ef.py), 실제 그림2개를 생성했다. 추가 GPU 실험은 수행하지 않았다. 다음 우선순위는 새 코드·뉴런·과제로의 probe 전이 검증이다.
- GitHub 사전등록 `4f47837e0158d960b6583851f4e58b126c393db8`, calibration freeze `bcc89aa2492c19fec69fe7336c4e78e2b4e796f6`를 각각 원격 main 재조회로 확인했다. 생성 manifest·tensor·전체 logit·비공개 연구 채팅은 공개 목록에서 제외했다. 이후 게시 상태는 [R2 receipt](../logs/summary/round2_ef_publication.json)를 따른다.
- 실측 결과 요약을 외부 ChatGPT 채팅에 전송하는 추가 검토는 자동 승인 검토에서 구체적인 비공개 payload 승인 부재를 이유로 거절됐다. Send 완료로 기록하지 않았고 다른 경로로 우회하지 않았다. 결과 해석·후속 추천에 대해 두 외부 모델이 재검토했다고 주장하지 않는다.
- 원시 재계산·공개 family replay·보고서/그림/README 최신성, CPU 111개 시험과 전체 artifact 검사19개를 통과했다. 결과29개 파일(899,686bytes)을 `aa1b92fc45b1dd72e3410d303f50296c948c062f`에 게시하고 원격 main SHA 일치를 확인했다. 이전 아이디어 파일은 이후 사용자가 제공한 공개 지침에 따른 이 새 checkpoint에 포함했으며, 이전 거절 기록은 보존했다. 별도로 거절된 외부 채팅 결과 전송은 계속 보류했다.


## 2026-10-29 · 승인된 실측 요약의 GPT·Claude 재검토와 R3 초안

- 사용자가 두 채팅에 실측 요약을 전달하도록 명시적으로 승인한 뒤, 기존 GPT Pro Deep research와 Claude Opus 5.5 Extra 채팅에 같은 요약과 설계상 한계를 전송했다. GPT의 새 완료 Deep Research 보고서와 Claude의 완료 Extra 후속 답변을 실제로 읽었다. 앞 entry의 전송 거절은 과거 상태로 보존하며, 현재 검토는 완료다. Claude가 새 장시간 웹 조사를 했다는 주장은 하지 않는다.
- 두 모델 모두 후보를 먼저 실행한 E의 dose 매칭으로 후보 관측 전 예측을 입증할 수 없고, F의 손상 gate 실패로 중복성·동등성·기능 부재를 판정할 수 없다는 데 동의했다. GPT는 candidate-blind transfer를 1위, Claude는 signed token/rule 분해를 1위로 제안했다.
- 사용자의 Novelty 우선 조건에 따라 최종 1위는 내용·답변 기호·규칙 정합 효과의 부호 분해(R3-A), 2위는 고정 probe의 새 생성기·좌표 전이 예측(R3-B)으로 정했다. 정답 라벨 교환의 항등적 부호 반전, 단순 confidence 증폭, 적은 대조의 순위 p값을 피하는 판정식을 추가했다. AB/CD는 양쪽 모두 calibration하는 두 코드 재현이며 미관측 토큰 전이로 부르지 않는다.
- 피드백의 92% 오차 감소를 과제 간 설명률로 해석한 부분, 활성값의 cal min/max 이탈19.2%를 dose 이탈로 표현한 부분, 한 predictor R²=.00033을 S 값으로 부른 부분은 채택하지 않았다. 새 전이 검정 실패를 기존 E의 원인 확정으로 해석하지 않는다.
- P0 BF16/FP32 수치 검증을 선행 조건으로 분리했다. R3-A 기본18,624/cap21,000 forward, R3-B 기본6,624/cap8,500, P0 기본576/cap1,000은 제안 예산이다. 새 GPU 추론은 전혀 실행하지 않았으며 모든 R3 결과는 NOT_EVALUATED다. R2 동결 분석·가설·원시 수치는 변경하지 않았다.
- [추천 보고서](../reports/ROUND3_RECOMMENDATIONS_KO.md), [구조화 검토 요약](../logs/summary/round2_ef_peer_review.json), [예산·설계 초안](../configs/round3_plan.yaml)에 근거와 성공·중단 기준을 기록했다. 검토 원문·전송 payload·화면은 로컬 runs에 보존하고 공개에서 제외한다. 두 AI의 검토는 독립적인 모델 재현이 아니다.
- 완료 검증: 기존 동결 R2 분석의 raw 재계산·공개 family replay, CPU 회귀시험111개, artifact 검사19개, 새 설계의 CPU/토크나이저 검사12개, 모든 보고서·README 최신성 및 private 검토 증거 SHA 일치가 PASS다. CPU toy 검사는 새 모델 실험 결과와 구별한다. 공개는 아래 명시적 파일 목록과 비밀정보·크기 검사를 거친 checkpoint로 수행한다.


## 2026-10-29 · 추천 1번 R3-A: 실제 P0 비교 후 등록 gate에서 중단

- 사용자의 1번 실행 요청에 따라 `configs/round3a.yaml`과 [실행 전 등록](ROUND3A_PREREGISTRATION.md)을 새로 고정했다. 로컬 commit `1a6f3a6`, 공개 사전등록 `ffef6fe93534177896f81997374a141fc5e78858`의 main 반영을 재조회한 뒤 추론을 시작했다. 이전 추천 초안은 원문 그대로 보존했다. R2와 다른 새864변형을 생성했고, 현재 측정에는 P0의8 family×규칙/기호4변형=32개만 사용했다. 나머지 분할의 모델 출력은 측정하지 않았다.
- `20261029T190213_round3a_p0`: 동일 BF16 저장 값을 FP32로 올리는 순차 단일 프로세스 비교. 576개 조건 행, smoke/generation·훅 검증 포함699 forward, 본 run54.24초·모델 로딩6.51초 별도. Peak allocated47.997GB/reserved49.543GB, 관측 최소 GPU 여유49.718GB. P0 cap1,000회/1,800초 이내다. 모델 revision·seed·정렬 위치·실제 환경·source/raw hash를 기록했다.
- 원 BF16 smoke28개, 정렬 hook34개와 두 dtype의 read-only/동일-forward 좌표복원/훅제거 exact·finite 검사가 모두 PASS였다. ±.5 P0 clean SD의 실제 전달 delta는 양 dtype/네 좌표에서100% nonzero였다. 측정 장치 실패로 중단한 것이 아니다.
- 평균 절대 signed 효과 BF16/FP32: H001 .044800/.000199, H002 .041870/.000133, H003 .039124/.000144, H004 .036865/.001019 nats. Signed Spearman은 각각 .321/.096/.189/−.378로 사전 기준≥.70을 모두 미달했다. FP32 |d|≥.02인 변형은 네 좌표 모두0/32로 기준≥8개를 미달했다. 이 하위집합의 부호 일치율은 해당 없음이며0%로 표현하지 않았다. 32개 변형은8 family의 파생이며 독립 n=32의 검정이 아니다.
- 수치 eligible 후보0개로 `STOPPED_AT_NUMERICAL_QUALIFICATION`. Discovery/calibration/confirmation은 NOT_EVALUATED다. Dose 증가·gate 완화·후보 교체로 같은 연구를 계속하거나 추천2번을 자동 실행하지 않았다. 각 dtype clean 정답은28/32이고 후보256개 개입 행의 top-1 변경은0이었다. 본 과제의 자유 생성 행동은 측정하지 않았다.
- 이 위치/dose/과제에서 정밀도에 안정적인 의미 효과를 확인하지 못했다. BF16의 실제 인과적 변화를 모두 무작위 잡음이라고 하거나, 뉴런 기능 부재·FP32의 참값·다른 조건의 무효를 주장하지 않는다. 전체 dtype 비교이므로 효과 차이의 발생 구간은 아직 국소화하지 못했다. R2 원시 수치/등록 판정은 보존하며 정밀도 일반화나 의미 기제 해석은 보류한다.
- [실측 보고서](../reports/ROUND3A_RESULTS_KO.md), [실측 요약](../logs/summary/round3a_p0.json), [원시 재계산](../logs/summary/round3a_p0_analysis.json), 실제 수치 그림을 생성했다. Prompt manifest·activation/logit cache·원시 행은 로컬 runs에만 보존한다. 정밀도 의존성이 어느 연산 구간에서 커지는가는 별도 신규 연구 가설이며 미측정이다.
- 완료 검증: CPU toy/회귀시험116개, 전체 artifact 검사19개, P0 원시 재계산31개, 모델/원시/cache 없이 별도 /tmp 사본에서 수행한 공개 paired gate 재계산, 보고서·SVG·README 재생성 검사가 모두 PASS다. 실제 PNG를 시각 검사했다. 이는 의미 기능 가설 지지가 아니다.


## 2026-10-29 · P0 실측의 GPT Pro·Claude 재검토와 R4 초안

- 사용자의 새 명시적 요청에 따라 P0 실측 요약·공개 근거·정확한 nominal/actual delta 한계를 기존 GPT Pro와 Claude Opus 5.5 Extra 채팅에 전송했다. GPT의 웹 검색을 포함한 10분30초 완료 답변과 Claude의 완료 답변을 읽었고, Claude에는 통계·원전·실제 tied head 구현 문제를 되물어 완료 수정본까지 확인했다. 새 장시간 Claude Research나 별도 신규 GPT Deep Research 문서가 생성됐다고 주장하지 않는다. 두 답변은 독립적인 모델 재실행이 아니다.
- 두 모델 모두 R3-A 중단을 지지했다. P0는 수치 자격 미통과이며 의미 기능 반증·FP32 참값·R2 소급 무효화로 표현하지 않는다. Prefix trajectory, 전달 벡터, 중간 연산과 terminal readout을 아직 분리하지 못했다.
- 자체 순위는 달랐다. GPT: 독립 P→Q 전이예측 → 공통 반응의 수치적 양방향 이전 → donor별 작은 규칙 경로. Claude 수정본: 에너지 맞춘 검출 연합 → placebo/readout 동반 감사. 사용자의 Novelty 우선 조건으로 최종 순위는 수치적 공통 패턴의 양방향 검증(R4-A), 같은 쓰기 에너지의 검출 연합(R4-B), 미관측 개입 예측(R4-C, 조건부)으로 정했다.
- 비율 CI가1을 포함하면 동등하다는 판정, full12B FP64 인증, k 증가만으로 분산 기제라고 하는 해석을 채택하지 않았다. Local FP64 audit와 전체 출력 정확도 보증을 구별한다. 비단조 곡선은 관측 grid crossing만 보고하며, 넓은 CI는 미결정으로 둔다. 단순 내용/답변기호 분리와 정밀도 floor에 직접 가까운 선행을 확인해 기존 signed 제안을 강등했다.
- 실제 설치된 Gemma4 구현에서 tied input embedding/head와 final tanh softcap을 확인했다. 기존 편집 cache에는 final hidden이 없어 FP32 head 비교에 재실행이 필요하다. G0는 functional readout 복사본과 명시적 dtype을 사용하는 선행 검증이며 논문 Novelty와 분리했다.
- [새 추천 보고서](../reports/ROUND4_RECOMMENDATIONS_KO.md), [검토 요약·원전 audit](../logs/summary/round3a_peer_review.json), [설계·예산 초안](../configs/round4_plan.yaml)에 가설·대조·성공/반증/식별불가를 기록했다. 모든 R4는 DRAFT/NOT_EVALUATED/미등록이고 신규성은 UNVERIFIED다. 이번 검토 중 새 GPU 추론은 전혀 실행하지 않았다. P0 원시 수치·사전 gate는 불변이며 R3-A는 계속 중단 상태다. 원문·전송문·화면은 로컬 runs에만 보존한다.
- 완료 검증: 원 P0 raw 재계산31개·공개 paired replay, CPU 회귀시험116개, 전체 artifact 검사19개, 새 보고서/README 최신성·새 초안 예산 산술·원전/검토 증거 hash 일치를 통과했다. 새 보고서는 요약 JSON과 계획 YAML에서 재생성했다.


## 2026-10-29 · R5 선택과 실제 GPT·Claude 공동 설계

- 사용자가 새 제안 2번(지속적인 뉴런 차단 아래 문맥 피드백에 따른 기능적 모집)을 선택하고 기존 GPT Pro Deep research와 Claude Opus5.5 Extra Research의 협업 설계를 요청했다. 두 사이트에서 실제 Research 모드를 선택해 같은 초안을 전송했고 새 완료 보고서와 교차 비판 후 최종 답변을 모두 읽었다.
- CoAx의 conditional co-ablation과 매우 가까운 선행을 확인했다. 새 목표는 같은 물리적 차단에서 동일한 문제·참조답에 붙은 오류 status만 바꿀 때 나타나는 query-time causal dependence 차이다. 새 물리적 회로/알고리즘 생성·형식적 mediation·특권적 introspection은 주장하지 않는다.
- 제공 nonce 표의 직접/두 단계 조회, 네 응답 코드의 질의별 교환, code-map-only guard, reference-corrected status-only history, 교차 차단 이력, query-only 2차 차단을 채택했다. 좌표 선정은 discovery에 한정하고 모든 과학적 단계는 같은 저장값의 FP32-upcast로 계획했다.
- 3 physical 상태×5 history에 세 no-history baseline을 더한 18개 조건으로 산식을 바로잡았다. 출력 Fisher 변위에 맞춘 확인 대조 재선택, norm replay가 모든 대안을 없앤다는 해석, TOST만이 방향 가설 기각 방법이라는 제안을 채택하지 않았다.
- [설계 보고서](../reports/ROUND5_DESIGN_KO.md), [계획·예산](../configs/round5_plan.yaml), [실제 협업 기록](../logs/summary/round5_design_peer_review.json), [CPU 검증](../logs/summary/round5_design_verification.json)을 남겼다. CPU 합성/토크나이저 검사와 미래 forward 산술을 실제 모델 결과와 분리했다. R5 GPU 추론은 아직 0회, 전체 확증 미등록이며 손상·회복·모집은 모두 NOT_EVALUATED다. R2·R3 실측과 중단 판정은 보존했다.

## 2026-10-29 · R5 실제 실행과 차단 자격 gate 중단

- 사용자의 실행 요청으로 새 technical/discovery manifest와 후보 순위·중단 기준을 고정했다. [실행 등록](../reports/ROUND5_EXECUTION_REGISTRATION_KO.md)을 공개 `33715757411b706c99c44e8fd078c161f0b2be0d`에 게시한 뒤 원격 11개 파일 내용을 다시 읽어 검증하고 추론을 시작했다. 코드 변환 witness 추가 408회는 관측 전 수정이며 전체 조건부 예산 31744회는 원래 cap 32000회 안이다.
- 실제 `20261029T234656_round5_discovery`에서 1680 full-prefix forward를 수행했다. BF16/FP32 기술 검증 48회와 discovery 1632개 관측이며 380.83초, peak allocated 48.23GB / reserved 49.58GB였다. 모델 로딩은 별도다. 단일 GPU 프로세스·고정 파라미터·use_cache=False를 확인했고 종료 뒤 GPU 계산 프로세스가 남지 않았다.
- Healthy 24질의씩 직접/양쪽/코드 변환 100%, 두 단계 87.5%로 준비 기준을 통과했다. 층 15/22/26/35 × 크기 32/128 × 활성 대비 두 방향의 16집합을 모두 평가했다. 모든 차단의 양쪽 표 정답은 24/24로 손상 0pp, 적격 쌍 0개였다. `STOPPED_AT_DISCOVERY_LESION_GATE`, 이 bounded search에서 NOT_IDENTIFIABLE이다.
- 차단 전달·같은 forward 복원·제거·readout parity는 통과했다. 이번 후보 선정법과 범위로 모집 검정에 필요한 손상을 만들지 못했다. 적응 불가능성·뉴런 기능 부재·모집 가설의 반증을 주장하지 않는다. 차단 크기·층·기준을 사후 변경하지 않았고 calibration/confirmation·피드백 회복·query-time 모집·norm replay는 NOT_EVALUATED다.
- [실측 보고서](../reports/ROUND5_RESULTS_KO.md), [실측 JSON](../logs/summary/round5_discovery.json), [CPU 재생](../logs/summary/round5_replay.json), [검증 기록](../logs/summary/round5_execution_verification.json), 전체 후보 그림을 생성했다. 1632개 raw 관측을 CPU에서 다시 계산했고 기존 연구 파일 38개와 등록 source 11개의 해시를 확인했다. 추론 전 CPU 단위/회귀시험 130개, 결과 이후 artifact 검사 19개와 별도 R5 replay/report/README/등록 검사가 통과했다. 그림 PNG를 실제 시각 검사했다. 원시 활성값·입력·출력 행은 로컬 runs에만 둔다.

## 2026-10-30 · R5 실측의 GPT Pro·Claude 재검토와 R6 추천

- 사용자의 명시적 요청으로 두 기존 웹 채팅에 같은 R5 실측 요약과 공개 근거를 전송했다. GPT Pro의 웹 검색을 포함한 완료 답변(10분12초)과 최종 답변(2분55초), Claude Opus 5.5 Extra의 완료 검토·해석 수정·최종 교차 비판을 읽었다. 이번에 새 독립 Deep Research 문서를 생성했다고 하지 않는다. 두 모델은 원시 Gemma 추론을 독립 재현하지 않았다.
- 같은 R5 discovery의 저장 logits 1632행을 CPU로 재계산했다. Dual healthy 정답 margin은 8.38~13.07nats, 중앙11.19였다. 정확도 손상0pp여도 L15_K128_composition의 dual 평균 절대 margin 변화는 .3316nats였다. L26_K128_composition은 direct/code-only margin을 각각 평균 .5450/.5264nats 낮췄다. 이는 선택에 사용한 8 episode의 사후 기술 진단이며 새 확증·의미 병목의 증거가 아니다. 기존 R5 gate와 중단 판정은 변경하지 않았다.
- 추가 진단을 두 채팅에 전달하고, 후기 표 행 상태의 출처 혼합·donor 답/문자 복사·cue 어휘 전달·항등 복원을 실제 rescue로 취급하는 문제를 되물었다. 최종 순위는 1) 내용·코드와 분리된 출처 선택 병목, 2) 백업의 정답 교정과 대체 출처 추종 분리(적격 경로 발견 뒤 조건부), 3) 작은 중간값의 미관측 후속 함수 재사용이다. 큰 site 탐색과 native 압축 자체는 공통 진단 도구이며 별도 novelty로 세지 않는다.
- 받는 문제의 두 답과 donor의 선택/미선택 답 및 코드 문자를 구분하도록 보완했다. 이런 충돌 회피는 무제약 완전 요인설계가 아니므로 허용 셀 안의 균형 무작위화로 명시한다. Query G 손상·다른 세계로의 선택 전이·독립 same-selector donor 복원을 함께 요구하고, cue 제외·새 cue 표현·code-only 연속 지표·고정 무작위군·postnorm 기하 대조를 둔다. G 복원은 query mask를 실제 donor 값으로 다시 여는 별도 arm이며 지속 zero라고 하지 않는다.
- [추천 보고서](../reports/ROUND6_RECOMMENDATIONS_KO.md), [검토 근거](../logs/summary/round6_peer_review.json), [계획·예산 초안](../configs/round6_plan.yaml), [사후 margin 진단](../logs/summary/round6_margin_diagnostics.json)을 생성했다. 첫 pilot은 호출 슬롯2352+reserve148=cap2500 제안이다. 전체 확증 arm manifest와 비용은 아직 미완성이다. 모든 R6는 DRAFT/NOT_REGISTERED/NOT_EVALUATED, 신규성은 UNVERIFIED이며 새 GPU 추론은0회다. 기존 실측·등록 50파일을 해시로 보존했다. 원 대화·전송문·스크린샷은 로컬 runs에만 둔다.

## 2026-10-30 · R6 세 질문과 별도 A2 실제 실행

- 위 entry는 설계 당시 상태다. 이후 A 선택 병목·B 백업 교정·C 중간값 재사용을 등록·실행했고, A의 별도 명시 경로 A2 연구도 원래 연구와 구분해 등록했다. 실제 합계 744 forward(science608 + technical136)다. 네 run 모두 기술 검사는 통과했으나 등록 과제 준비도 또는 선행 자격에서 중단했다.
- 실제 인과 병목·백업 모집·재사용 중간값을 발견했다고 하지 않는다. Calibration/confirmation 출력은 보지 않았다. 원래 A와 A2를 하나의 성공 실험으로 합치지 않았다. [실측·중단·환경](../reports/ROUND6_EXECUTION_RESULTS_KO.md), [독립 CPU replay](../logs/summary/round6_execution_replay.json)를 보존했다.

## 2026-10-30 · R7 자율 연구 재설정과 다섯 P0

- 기존 R2-E/F, R3, R5/R6 원시 자료를 재감사했다. R2-E의 dose 사후 매칭 및 string_rule에 약91.92% 이득 집중, R2-F 손상 자격 실패, R3 네 후보 numerical qualification 실패, 초기 top1변경39행의 단일 원본 집중을 확인하고 원 판정을 보존했다.
- 다섯 필수 선행의 방법·결과와 추가 근접 원전을 읽었다. 두 실제 연구 채팅의 독립 검토와 교차검토 뒤 A 논리 영향 범위, E 교란 계열 예측, B 질의 전 carrier 경로, C 취소 이력 잔류, D 선행 문맥 순서를 독립 질문으로 선정했다. A는 원격 b0ea725d2ec7fd539272847bc9d9a28b198a98ca, B–E는 3598ad29a70092677e2a6eafae398f8f86806d35에 최초 출력 전에 공개했다.
- 실제 Gemma-4-12B-it revision707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7에서 4278 forward(science4108 + technical170)를 수행했다. 각 재로딩의 BF16/FP32 훅·readout 기술 검사를 통과했고 GPU는 한 프로세스만 실행했다. Science는 동일 BF16 값을 FP32로 상향한 모델이며 참값 모델이 아니다. 실제 Unified config의 PLE와 KV sharing 수는 모두0이다.
- A: 준비도 중단/Inconclusive. R1 anchors48행에서 지시된 규칙과 보존된 R0 예시가 모순되는 **우리 설계 오류**를 발견·공개했다. 순수 규칙 실행 실패나 논리 영향 부재로 해석하지 않는다. 원본 입력·등록·중단을 보존했다.
- B: 특정 봉인·47층 token 궤적 교환 프로토콜의 Negative. Null graph와 identity 검사는 통과했지만 봉인 후 full-vocabulary 코드 형식이 붕괴했다. 원문 반복 접근은 허용되어 단일 compact state 기억이나 자연적 저장 부재를 주장할 수 없다. C: 등록 합성 특이 증폭 Negative이나 취소 이력의 old-value logit 흔적은 남았다. D: inverse 준비도 중단/Inconclusive이며 본 priming은 미측정이다. E: 6독립 보정 세계로 고정한 예측기의 20세계 prospective ΔR²는 −.099841 CI[−.210115,.0149589], 등록 Negative다. Position 사후 양성을 골라 주가설을 바꾸지 않았다.
- 4108 원시 행의 oracle·코드·head/softcap·margin·토큰/개입 scope·세계 단위 통계와 기존91개 보호 파일을 독립 CPU replay했다. B의 query-blind prefix cache48쌍은 최대 차이0, E의 target 이전 동결 예측80개도 검산했다. 독립 GPU 재현과 P1/P2는 수행하지 않았다. [실측 보고서·그림](../reports/ROUND7_EXECUTION_RESULTS_KO.md).
- 결과 공개 checkpoint2e4e3ab184a5eb9ec2121845846e5d5df0d6ecd5의 28허용 파일을 원격 SHA 재조회·fetch·개별 내용 해시로 확인했다. 원시 prompts·activation·채팅 전체는 로컬에만 보존했다.

## 2026-10-30 · R7 사후 독립 심사·교차검토·후보 갱신

- 같은 핵심 실측 패킷을 실제 GPT Pro와 Claude Opus5.5 Extra에 전달했다. GPT 독립 답변18m21s·교차답변7m17s와 Claude의 두 완료 답변을 수집했다. Pro 웹 검토는 별도 Deep Research 보고서가 아니다. GPT 첫 긴 첨부의 서버 오류 후 전체 핵심 증거를 입력란에 펼쳐 실제 재검토를 받은 과정을 구분했다.
- Claude의 B 코드표 차단, C 정확도 미측정, unrelated값 우연일치 편향 방향, E 상관 부호에 따른 이분법은 실제 토큰249개 고유 encoding·1024행·추가 CPU 통계와 맞지 않아 교차검토에서 철회됐다. C에서는 512쌍 중 top1변경7·정답변경6이고 현재 obsolete 값 선택은0행이다. 따라서 행동 변화 전무라고도 말하지 않는다.
- 실제 렌더링을 감사하니 B의 12latent draw는 11고유 main view였다. 원래 draw 가중 결과와 11view 동일가중 사후 민감도를 함께 기록했다. 미사용 P2에 B 전체view1개·D prompt112개·E prompt6개의 중복을 발견했다. 실제 사용 E calibration–P0 prompt 중복은0이다. 예약 입력을 조용히 교체하거나 독립 확증으로 재사용하지 않는다.
- GPT는 후속1순위 ‘값–계보–권한’의 Go를 LLMCAPLENS·CoRe 확인 후 Hold로 낮췄다. Claude는 내부 유사도·철회 표지·직렬화의 자체 세 제안을 모두 철회했다. C/E의 논문화 Hold 대 No-Go 의견 차이는 원시 결과 보존과 현재 추가 투자 No-Go로 판단했다. 새 명제 없이 dose·층·prompt를 바꿔 구제하지 않는다.
- 다음 다섯 제안은 신규성 심사상 Hold2/No-Go3이며 실행 Go0이다. 실행 가능한 다섯 새 연구를 선정했다고 하지 않는다. [실제 사후 심사 및 후보별9항목](../reports/ROUND7_REVIEW_AND_NEXT_KO.md), [누적 논문 순위](../reports/PAPER_CANDIDATE_RANKING_KO.md), [27편 원전 확인 범위](../logs/summary/round7_literature_audit.json)를 기록했다. R7은 종결했고 자율 연구 목표는 계속 열려 있다.
- 향후 연구용 입력 계약은 모순 deterministic anchor, 관측 세계 중복, 실제 token 분할 중복을 첫 출력 전에 거부한다. 실측 당시 CPU시험153개와 후속 인프라4개를 분리했으며 현재157개 통과다. 이를 새 실제모델 실험으로 세지 않는다. README 수치·그림·후보 상태는 JSON에서 재생성한다.

## 2026-10-30 · 후속 선정 감사와 실패 종료 조건 적용

- R7 종료 때 남은 S1 값·계보·권한과 S2 오류 상보성 Hold를 원전 방법·결과와 대조했다. SPLIT, Knowing Is Not Choosing, Automatic Model Selection, Overthinking the Truth 및 Models Take Notes의 방법을 추가 확인했다. ATT는 공식 NeurIPS 워크숍 초록만 확인했으며 OpenReview 본문은 접근 검증 화면으로 미확인이다. NeuroTaint가 정확히 같은 값 대조를 이미 했다는 주장과 ATT가 base LLM 내부 정책 판정을 검사했다는 주장을 신규성 기각의 확정 근거로 쓰지 않았다.
- 같은 핵심 패킷으로 GPT Pro와 Claude Opus5.5 Extra의 실제 독립 답변을 받은 뒤, 양측 주장·원전 정정·Codex 반례를 같은 교차 패킷으로 전달했다. 마지막 사실 검산도 두 채팅에 실제 전달·수집했다. GPT의 독립16m08s·교차4m12s·사실검산2m02s와 Claude의 완료 답변을 보존했다. Pro 웹 검토이며 별도 Deep Research 보고서는 아니다. 원문·첨부·화면은 로컬 runs에만 두고 공개 JSON에는 해시와 판단 범위를 남겼다.
- S2의 oracle을 검산했다. truth/기존 판정/새 판정의 모든 쌍별 결합표·정확도75%·공동오류율12.5%가 같은 두 32개 항목 분포에서, 추가 판정의 Bayes 정답률은87.5%와75%로 달랐다. 쌍별 상보성만으로 추가 정보의 정답 기준을 정할 수 없다. 기초적인 합성 확률 반례이며 새 정리나 실제12B 결과, CAVEAT/SPLIT의 선언된 모형에 대한 반증이 아니다.
- GPT가 제안한 계산 공유 G1은 결과 재사용과 입력 복사 후 재계산이6144개 제안 관측에서 같다는 CPU 반례로 검토했다. GPT는 Go를 철회했다. Claude가 남긴 같은 입력·다른 표 수정안도 결과 저장과 표 복사 후 재계산이20736개 관측과 입력 probe에서 같았다. 다른 입력 probe는 이 두 특정 모형 일부를 구분할 수 있으나 원래 설계 밖이며, 입력 특이 행 복사와 결과 저장을 일반적으로 분리하지 않는다. Claude도 마지막 조건부 수정안을 철회했다.
- GPT의 마지막 지적대로 답값 일치만으로 실제 logit 일치를 주장하지 않았다. 최초 답값-only 감사본을 로컬에 보존하고, 동일 값→4코드 logit/softcap readout을 두 반례에 명시한 가능 구성을 추가했다. 합성 logit·양성 margin 패턴까지 같지만 **실제 Gemma head를 측정·복제한 것이 아니다.** 모든 미래 대비의 식별 불가능성을 증명하지도 않는다.
- H1 취소값 직접 읽기/후속 오염과 H2 형식 cue 경로도 새 독립 예측과 기여를 확보하지 못했다. H1의 작은 분모 비율과 비배타적 mask, H2의 기존 filter/note 경로 적용 한계를 기록했다. S1/S2/G1/H1/H2는 현재 모두 No-Go다. 검토한 다섯 제안 각각9항목을 보존했으며 실행 가능한 새 연구 다섯 개를 선정했다고 하지 않는다. 새 Gemma forward는0회, 새 P0/P1/P2는 모두 NOT_EVALUATED다. 이를 실제 여덟 번째 모델 라운드로 세지 않는다.
- 원래 목표 §4E는 “다만 더 이상의 유효한 연구질문이 없거나, 자원이 부족하거나, 실험이 계속 식별 불가능해진다면 중단 사유를 명확하게 기록하라”고 허용했고 §8은 여러 라운드 후 논문 기여 미발견의 정직한 보고를 요구했다. 반복된 선행 자격·식별 실패와 현재 후보들의 신규성/경쟁 기제 구별 한계로 이 **실패 종료 분기**를 적용한다. 논문 수준의 새 주제 발견에는 실패했다. 고정 라운드 수·자원 부족이나 가능한 모든 질문의 소진을 주장하지 않는다.
- 요청한 R6 세 실험+별도A2의744회, R7 다섯 P0의4278회와 기존 양성·음성·중단 기록은 불변이다. 157개 CPU 시험(4.26s), R7 원시4108행과170기술 호출·91보호파일·B cache48쌍·E prospective80예측 재검산을 통과했다. 최종 GPU 계산 프로세스 조회는0개였다. 새 논문 명제·기제 병목을 발견했다는 뜻이 아니다.
- [최종 실패 종료 보고서](../reports/AUTONOMOUS_DISCOVERY_OUTCOME_KO.md), [전체 요구사항·종료 근거 감사](../logs/summary/autonomous_discovery_completion_audit.json), [실제 협업 기록](../logs/summary/round8_peer_review.json), [후보 목록](../reports/PAPER_CANDIDATE_RANKING_KO.md)을 유지한다. 재개에는 경쟁 기제들이 다른 결과를 내는 새 사전 대비와 구별되는 기여가 필요하며, 기존 dose·층·prompt를 양성이 나올 때까지 조정하지 않는다. 이번 종료 자료도 명시적 허용 목록·비밀정보/크기 검사 후 공개하고 원격 SHA와 각 파일 내용을 다시 조회한다.

## 2026-10-30 · 사용자 재활성화에 따른 R9 연구 재개

- 새 목표 첨부7ff9ce66-bca1-4a17-a5fd-0fc25bec99df의 본문은 이전 전체 자율 연구 목표와 같은 SHA418276d9…42d2다. 사용자의 재개 지시로 새 방향의 조사를 시작했다. 이전 실패 종료·기각과 실제 측정의 판정은 보존하며 이를 현재 목표의 성공으로 재사용하지 않는다. 재개 시 작업 트리가 깨끗했고 공식 원격5e06005d…7c15 및 과학 코드·기존 결과139파일을 확인했다.
- 두 기존 채팅에 새 독립 질문 다섯 개와9항목을 요청했다. GPT의 실제 Deep research 요청·재시도 및 긴 Pro 요청은 오류로 종료돼 완료로 세지 않는다. 새로고침 뒤 짧은 동일 핵심 증거 재진술·기존 채팅 자료·공개 보고서 링크로 일반 Pro가 실제 응답을 생성하는 상태를 확인했다. Claude에는 같은 전체 핵심 패킷을33행 pasted attachment로 보내 실제 Opus5.5 Extra Research를 시작했고, 별도 Research 패널의 source 수 증가와 Stop response를 확인했다. 아직 완결 새 제안이나 교차검토를 수집했다고 하지 않는다. 원문·화면은 로컬 runs에 보존한다.
- 산술 원전2508.02513의2–4절에서 digit-specific Fisher 선택·여러 층 activation interchange·carry 유발 donor·single/multi-digit tokenization을 이미 검사함을 확인했다. 단순 자릿수/자리올림 회로 발견은 신규성 후보에서 제외한다. Early Word Reconstruction 연구는 공식 초록만 읽었고 본문 미확인 상태다. 검색 결과 부재는 신규성 근거로 쓰지 않는다.
- [재개 기록](../reports/ROUND9_REOPENING_KO.md), [현재 진행·보호 해시](../logs/summary/round9_progress.json), [선정 단계 설정](../configs/round9_loop.yaml)을 추가했다. 새 모델 연구 선정0·실제 Gemma forward0, P0 미등록·미평가/P1·P2 NOT_EVALUATED다. GPU는 후보·대비·정확한 예산·입력 계약을 출력 전에 등록한 뒤에만 실행한다.


## R9 사전등록 — 두 실제 교차검토 완료, 새 P0 출력 전

GPT Pro 독립 원전검토31m44s 및 실제 교차10m30s, Claude Opus5.5 Extra Research19m21s 및 실제 교차답변을 회수했다. GPT 별도 Deep Research 오류/compact recovery, Claude241sources 표시와 원전 독서 범위의 차이를 공개 기록했다. 상대 새 의견은 두 독립 답변 완료 뒤에 처음 공유했다. 양측 순위·중립 예시 Go 이견은 보존한다.

Codex는 A정보 분할/B보상 재배분/C수집 절차를 우선, D고차 확률/E자원 경쟁을 엄격한 작은 선별로 선택했다. 논문 Go0개다. 실제 입력 개입에서 경쟁 예측을 구별하는 P0이며 뉴런/내부 알고리즘을 확정하지 않는다. 9항목 후보·비용·중단 기준은 reports/ROUND9_P0_PREREG_KO.md와 configs/round9_P0.yaml에 기록했다.

8개 독립 CPU oracle/분할 테스트 PASS(0.83s), authoritative 입력11900개·최대313토큰, 새 분할 중복검사 PASS. 과학3612＋공유 기술34=최대3646회, hard cap3700회/14400초/단일 GPU. 이 기록 시점 actual Gemma forward0; BF16/FP32 계측 검증 전 main을 실행하지 않는다. R1–R8 보호 파일139개 해시 일치.


## R9 실제 P0 — 1790 forward, 4 Inconclusive / 1 Negative

사전 공개0a1d8ab3a4992435e6b6b572a2c9531061d66b93 뒤 동일 동결 모델의 단일 GPU 실행을 마쳤다. P1/P2는 NOT_EVALUATED다.

A896회: anchor128/128, main 유효 코드99/768, To 시작669/768로 format gate 중단. B96회: path-sum47/96으로 main 미측정. C96회: likelihood-weight22/96, 유효 코드48/96으로 main 미측정. D540회: anchor60/64, main/reference format100%, 저차 정상·고차만 실패0/8 family로 등록 Negative. E128회: edge63/64, union11/64(유효한11답은 모두 정답)로 main 미측정. 과학1756+공유 기술34=1790회다.

CPU 재검산은 원시 행·token 입력·소스·카운터·등록 분석과 일치했다. 보호139개 파일을 유지했다. B의 route-choice cue와 YES/NO anchor의 잠재적 지시 충돌, A/C/E의 format confound, D의 같은 family 내 반복 입력을 사후 진단으로 기록했고 주 판정은 변경하지 않았다. 현재 추가 GPU는 중단하고 실측 후 GPT/Claude 독립·교차 비판을 요청한다.


### R9 실측 후 CPU 추가 감사

실제 tokenizer template hash가 모델 load와 일치하고, 저장11900개 입력 모두 enable_thinking=False의 빈 thought를 닫는 suffix를 포함했다. 해당 flag 무시/열린 thought 판독 설명은 CPU 입력 직렬화와 맞지 않는다. 새 model forward0이며 To 발생의 원인을 입증한 것은 아니다.

C의96개 확률 anchor는 정답이 모두 첫 번째 분수 선택지였다. A/B codebook 교환으로도 이 위치 shortcut을 제거하지 못하므로, 통과했더라도 충분한 Bayes 준비도 증거가 아니다. 현재는 gate 실패로 main미측정이며 원래 분류는 그대로다. 이 결함은 사후 설계 진단이지 새로운 과학 가설/양성 결과가 아니다. 이후 교차 패킷에 양쪽 동일하게 추가할 사실로 기록한다.

### R9 실측 후 실제 공동 심사와 현재 탐색 종료

GPT Pro의 독립10m02s·교차6m22s와 Claude Opus5.5 Extra Research 독립5m29s(UI276sources)·일반 Extra 교차 답변을 모두 실제 회수했다. 두 독립 답변 완료 전에는 새 상대 의견을 공유하지 않았다. GPT는 공개 코드·등록을 읽었지만 로컬 raw를 재현하지 않았고, Claude는 등록·YAML·코드에 접근하지 못했다. 별도 Deep Research 성공이나 UI source 수만큼의 본문 독서를 주장하지 않는다. 원문·화면은 로컬 보존, 공개에는 해시·범위·이견·정정을 기록했다.

핵심 이견은 D였다. 실행 전 정의에서 저차 실패가 P0_negative에 포함됨을 제시한 뒤 Claude는 D를 준비도 실패로 재분류하자는 의견을 철회했다. 등록한 공동 패턴0/8의 Negative와 저차 정상 eligible0인 조건부 고차 오류율의 추론 불가를 분리한다. B47/96의 chance 동등성, A의 To를 설명 기제로 해석하는 주장도 채택하지 않았다. C22/48은 형식 조건부 기술통계로 계산 가능하나 전체 능력의 증거가 아니다.

실행 전 CPU 검산은 수학적 정답과 분할을 확인했으나 B의 지시 층위·C의 정답 위치 결함을 놓쳤다. 사후에 원본을 수정하지 않고 해석 guard로 A/B/C/E primary estimate/CI를 null로 막았다. R9 소스·측정20개와 이전139개 보호 해시를 유지한다. 새 모델 호출은 계속0회다.

양측 사후 대안과 원전 방법·결과를 재검토했지만 새 경쟁 설명을 가르는 실행 가능한 독립 명제를 확보하지 못했다. source/copy와 martingale의 일반 대비는 가까운 본문으로 확인했고, 나머지 일부 인용 본문은 UNVERIFIED로 남겼다. 읽지 않은 논문으로 정확한 중복을 확정하지 않았다. 형식 수리를 새 연구로 세거나 숫자를 채우기 위해 새5개를 선택하지 않는다.

사용자 목표 §4E·§8의 현재 탐색풀 실패 종료 조건을 적용한다. R9 실제1790회와4 Inconclusive/1좁은 Negative를 보존하고 P1/P2는 NOT_EVALUATED다. 논문 가치가 검증된 새 주제 발견에는 실패했다. 자원 고갈·고정 반복 횟수·모든 미래 질문의 소진·미평가 가설의 거짓을 종료 이유로 삼지 않는다. [실측 후 심사·후속 선별·종료 근거](../reports/ROUND9_REVIEW_AND_NEXT_KO.md)와 [계산 가능한 판단](../logs/summary/round9_final_decision.json)을 유지한다.


## 2026-10-30 · R10 A/B/C/E 계측 수리 사전등록

사용자가 네 질문은 가설 음성이 아니라 미평가라며 계측·설계 수리와 새 데이터 재실험을 명시했다. R9의 종료는 당시 투자 판단으로 보존하고, 이번에는 같은 질문의 별도 수리 평가 R10을 수행한다. D와 기존 실측은 변경하지 않는다.

공개 자료 기반 GPT Pro 독립6m39s와 Claude Opus5.5 Extra Research3m45s(UI219sources), 실제 양쪽 교차(GPT8m22s·Claude 일반Extra)를 수집했다. 초기 비공개 초안 및 교차문 전송은 자동 심사에서 각각 차단됐고, 공개 자료 대안과 사용자의 명시 승인 뒤 허용 범위로 진행했다. 가중치·전체 원시 생성·인증정보는 전송하지 않았다. 의견 일치를 증거로 세지 않으며 원문·접근 한계·철회·Codex 판단은 별도로 보존한다.

A/B/E의 main 성공 일관성을 gate로 요구하는 순환 제안과 C의 posterior→likelihood ratio/4지선다 변경, E union→전역feasibility 변경은 채택하지 않았다. 기초 승패/순서/신호규칙·합계/숫자비교·우도정규화/장치규칙·edge/union/재사용규칙만 자격 gate다. 모델은 전체 어휘에서 greedy 생성하고 고유 최종 FINAL 선언을 판독한다. 내부 제어토큰은 지우지 않으며 EOS 경계·cap·미판독을 구분한다. 의미 null을NO로 바꾸지 않고 세계별 worst-case bounds와 탐색적 bootstrap을 사용한다.

GPU 접근이 없는 sandbox 시도는 model load 전0회 실패로 기록했다. 허용된 GPU 접근의 기술검증은 실제58회(훅34·cache16·생성계수8) PASS, 과학 출력0개다. 등록한 새 calibration/P0에는 여전히 출력0개이며12개 CPU 시험이 PASS다. 입력검사가 반복 B 합계 anchor를 출력 전에 거부해 실제 경로 문맥을 보존하도록 수정했다. 새 입력11040개는 전부 고유하고R9 사용/예약11900입력과 겹치지 않는다. 기존 보호189파일을 검증했다.

새 P0는 A/E8세계·B/C12세계, calibration각4세계, 확인각16세계(미사용)다. 보정768+P0최대3872=4640응답, 각256token, 최대1187840generation forward+공유기술64, 실제125만call/8시간/run4GB/단일GPU cap을 동결했다. source·입력·파서·gate를 P0 출력으로 변경하지 않는다. 수리 성공·가설 결과·논문 기여는 별개이며 P1/P2는NOT_EVALUATED다.


## 2026-10-30 R10 첫 calibration 실패와 별도 R10b 등록

원래 R10 실제 A 보정은 완결42응답(VALID30/TOKEN_CAP12)에서 필수 그룹95% 판독 기준 달성이 불가능해져 자기 GPU 프로세스만 추가 공학적 중단했다. 이 futility 중단은 원래 사전등록된 규칙이 아니었음을 기록한다. 7609완결 forward+75미완결 진입+50공유 기술+58별도 기술=7792회. 모든 P0는 미평가다. 실패·원시 행·소스·입력은 보존했다.

실제 GPT Pro 웹과 Claude Opus5.5 Extra의 새 독립·교차 심사 후 별도 R10b를 새 출력0개 시점에 등록했다. 간결 출력계약, 사후 관측 후 A 직접 공지 anchor 명료화, 세계/기능 round-robin, 독립 질문별 등록 futility, R9와 R10의 사용·예약 세계/입력 모두 배제. 원래 가설·terminal parser·256 cap·retry0·기초 gate·missing bounds 유지. 양쪽 모두 수리 Go이며 의미적 동등성이나 개선 원인의 인과 귀속을 주장하지 않는다. Claude는 공백 불일치 추정·A 선행 자격·추가2%/90% gate를 철회했다.

[별도 R10b 등록](../reports/ROUND10B_REPAIR_PREREG_KO.md), [원래 실패 집계](../logs/summary/round10_attempt1_abort.json). CPU17테스트와 새11040입력 준비는 실제12B 가설 결과가 아니다.
