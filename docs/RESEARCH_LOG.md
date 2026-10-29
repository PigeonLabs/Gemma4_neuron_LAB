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
