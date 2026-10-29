# P0 결과 재검토: 다음 연구는 수치적 공통 반응의 원인부터 묻는다

**최종 추천은 1위 수치 연산의 양방향 검증, 2위 에너지를 맞춘 뉴런 연합, 3위 미관측 개입 예측(조건부)이다.** GPT Pro와 Claude Opus 5.5 Extra에 동일한 실측 요약을 전달했고, 완료된 답변 및 Claude의 수정 답변을 읽었다. 두 검토자의 1순위가 달라 최종 순위는 사용자의 Novelty 우선 조건에 따른 Codex의 판단이다.

이번에는 새 GPU 추론을 하지 않았다. 아래 실험은 전부 **DRAFT / NOT_EVALUATED / 미등록**이다. 중단된 R3-A를 그대로 재개하거나 당시 gate를 낮추는 제안이 아니다. [R3-A 실측](ROUND3A_RESULTS_KO.md) · [구조화 검토](../logs/summary/round3a_peer_review.json) · [예산·설계 초안](../configs/round4_plan.yaml)

## 실제 검토와 현재 근거

| 검토자 | 완료 답변과 접근 범위 | 자체 우선순위 |
|---|---|---|
| GPT Pro | 웹 검색을 포함한 10분30초 답변. 보고서·등록문·paired JSON 일부 확인을 보고함 | 독립 P→Q 예측 → 수치 연산의 양방향 이전 → donor별 작은 규칙 경로 |
| Claude Opus 5.5 Extra | 공개 실측 보고서 확인. JSON·등록문 미열람을 명시. 초안 뒤 통계·구현·원전 보강을 요청하고 수정본까지 확인 | 수정본: 에너지 맞춘 연합 → placebo/readout 동반 감사. 단독 3위는 권하지 않음 |

새 장시간 Claude Research나 별도 GPT Deep Research 문서가 생성됐다고 표현하지 않는다. 두 답변은 연구 자문이며 독립적인 모델 재실행·재현이 아니다. 전송 원문·응답·화면은 로컬 `runs/round3a_peer_review/`에 보존하고 공개에는 요약과 해시만 기록한다.

실측은 576개 조건 행·699회 실제 forward, 8 source family의32변형이다. 개입 전달·훅·복원은 통과했지만 eligible 좌표는0개였다. 두 모델 모두 등록 중단을 지지하고, 기능 부재·효과0·FP32 참값·R2 소급 무효화는 결론에서 제외했다.

| 후보 | BF16 평균 절대 d | FP32 평균 절대 d | Spearman |
|---|---:|---:|---:|
| H001 | 0.044800 | 0.000199 | 0.321 |
| H002 | 0.041870 | 0.000133 | 0.096 |
| H003 | 0.039124 | 0.000144 | 0.189 |
| H004 | 0.036865 | 0.001019 | -0.378 |

효과 d의 단위는 nats이고 Spearman은 무차원이다. d는 고정 의미축의 ±개입 반차분이다. 32는8 family의 파생이고 독립 n=32가 아니다. 두 dtype의 nominal delta는 같았지만 clean trajectory와 실제 전달 벡터까지 고정하지 않았다. 따라서 최종 readout 반올림·중간 전파·prefix 상태·개입 반올림 중 어느 것이 원인인지 아직 모른다.

## 공통 선행 검사 G0 — 추천 순위와 별개인 준비 작업

새8 family×4변형에서 trunk(BF16/FP32)와 terminal readout(BF16/FP32)을 교차한다. 모든 clean/편집 조건의 **final RMSNorm 이전 hidden**을 새로 저장하고, 동일 저장 가중치의 functional 복사본으로 final RMSNorm·전체 LM head·tanh softcap을 재계산한다. 이미 반올림된 logits에 `.float()`를 적용해도 원래 값은 복구되지 않는다.

실제 Gemma4 소스에서 `lm_head.weight`는 input embedding과 tied다. 모듈을 직접 `.float()`하면 prefix 계산까지 바뀔 수 있어 복사본을 쓴다. 기존 편집 행/cache에는 필요한 최종 hidden이 없어 재실행 비용이 필요하다. Native readout을 먼저 exact replay하고, 동일 endpoint·경계 dtype·softcap 순서를 검증하지 못하면 다음 단계는 식별불가로 둔다. 제안 기본 576 full forward + 1,152 local readout, cap800/1,500회·30분이다. 현재 모두 미실행이다.

[MLP/head rounding 연구](https://arxiv.org/html/2610.01889v2)는 이미 site별 차이를 다룬다. [Float32 activation-patching 연구](https://www.mdpi.com/2079-3197/14/9/212)는 결정론적 반복 통과만으로 수치 validity가 확보되지 않음을 다룬다. 그래서 정밀도 비교나 head 위치 확인 자체를 논문 Novelty로 추천하지 않는다. 이 원전들의 수치 상수는 우리 모델에 전용하지 않는다.

## 1위 R4-A — 공통 입력 반응 패턴을 만드는 수치 연산의 양방향 검증

**질문:** 서로 무관한 개입들이 같은 입력에서 크게 반응하는 패턴을, 특정 연산 구간의 정밀도만 넣고 빼서 만들거나 없앨 수 있는가? 평균 효과가 줄어드는 현상보다 더 좁고 반증 가능한 원인 설명이다. GPT의 2위를 Novelty 관점에서 올렸다.

**최소 조작:** 독립 탐침 P 두 개와 미관측 Q 두 개를 사용한다. 공통 post-L26 시작 상태와 BF16에서도 정확히 표현 가능한 동일 endpoint 쌍을 고정한 뒤, 주변 연산 BF16/FP32 × 선택 구간 BF16/FP32를 교차한다. 이 공통 endpoint 연구는 native 좌표에서 유래한 잔차 변화의 재생을 평가하며, 두 dtype의 원래 native 개입이 완전히 같다는 주장이 아니다.

| 요인/대조 | 고정할 내용 |
|---|---|
| 수치 구간 | Discovery에서 L35 MLP branch·L47 block·terminal readout 중 하나 선택 |
| 반대 방향 조작 | FP32 주변에 BF16 구간 삽입 / BF16 주변의 같은 구간을 FP32로 교체 |
| 능동 대조 | 다른 구간 하나를 미리 고정하고 같은 교차 조작 |
| 음성 대조 | cast/copy identity, 과제 내 P 순서 교환, 길이·clean margin·dose baseline |
| 예측 순서 | P를 먼저 실행해 예측 봉인 → Q는 채점용으로 실행 |

**1차량:** calibration에서 각 수치 cell의 scale과 baseline/P→Q 회귀를 고정한다. Family별 표준화 Q 예측오차 감소를 계산해, 선택 구간과 다른 구간 사이의 **삽입 효과·제거 효과**를 검정한다(Holm2). 두 대비 모두 95% family CI 하한>0, 제안 최소효과0.05 표준화 loss 이상이어야 지지한다. 과제 내 교환 대조·calibration 고정 표준화로 단순 전체 크기 축소를 구분하며, clean margin/choice 변화도 함께 보고한다.

**분할/중단:** 새 discovery8·calibration24·confirmation64 family, 각 두 선택지 배정. Discovery는3구간 중 하나와 대조 하나만 결정하고 이후 바꾸지 않는다. Q의 변동이 수치 floor 아래이거나 replay/boundary 동등성이 실패하면 식별불가다. 평균 크기만 줄거나 다른 구간도 같은 패턴을 만들면 이 특정 원인 가설을 내린다. 상관이나 예측력만으로 의미적 입력 공통원인을 주장하지 않는다.

**제안 비용:** 기본 10,656 forward-equivalent, cap 12,500회·2시간. Suffix replay도 보수적으로1 forward-equivalent로 세고 local 모듈 작업량·시간을 별도 기록한다. 필요 개념은 부동소수점 반올림, 2×2 대비, 고정 예측오차다.

**기여 경계:** [Zhu의 수치 audit](https://arxiv.org/html/2608.11797v3), [configuration/random-control 연구](https://aclanthology.org/2026.blackboxnlp-1.73/), [LayerCast](https://arxiv.org/abs/2506.09501)와 겹치는 단순 민감성 재현을 넘어서야 한다. 새 family에서 구간별로 공통 반응 패턴을 양방향으로 옮기고, 다른 구간·균일 재조정 설명을 배제할 때만 중심 기여 후보가 된다. 정확한 신규성은 **UNVERIFIED**다.

## 2위 R4-B — 같은 쓰기 에너지에서 검출 뉴런 연합의 선택적 효과

**질문:** 모델 출력을 동일한 크기로 건드려도, 활성 선택도로 고른 좌표 묶음이 출력가중치/자연 변화량 순위나 무작위 묶음보다 특정 donor 답으로 더 잘 이동시키는가? Claude의 수정 1위다. 단일 좌표 실패를 더 큰 총 개입량으로 덮는 접근을 피한다.

**설계:** 제공 사전 translate/copy와 reverse/sort를 사용한다. Discovery의 clean 통계만으로 층 하나와 DET 순위를 고정한다. OUT는 |자연 donor 변화|×‖down-proj 열‖ 순위, RAND는 고정3개 seed 패널이다. 수신 좌표를 짝 donor의 자연값 쪽으로 α∈[0,1]만큼 이동하고 최대128좌표의 고정 순위 prefix를 사용한다. Primary는 **full FP32 upcast**로 측정하며, BF16+functional FP32 readout은 고정16 family의 보조 비교다.

세 에너지 수준은 calibration에서 고정한다. E는 layer scalar를 포함한 post-MLP 정규화 뒤 실제 잔차 변화의 제곱 L2 norm이다. 단순 k 사다리 대신 같은 E에서 DET/OUT/RAND를 비교하고 L2 오차±10%를 기록한다. 자연 좌표값을 섞은 hybrid 전체가 자연 상태라는 보장은 없다. 확증에서 좌표를 몰래 빼거나 clip하지 않고, local geometry만으로 suffix 출력 전에 정한 common support·매칭 실패율·전체표본 민감도를 공개한다. 매칭 실패가20%를 넘으면 해당 matched-energy 주장은 식별불가다.

**1차량:** donor 정답−receiver 정답의 logit margin 이동에 대해 family별 DET−RAND3평균, DET−OUT를 세 E에서 검정한다(Holm6). 이는 donor 방향의 반사실적 행동 전환이지 원래 receiver 지시의 정답률 개선이 아니다. 적어도 두 E에서 두 대조 모두에 대한 실용 효과와 양의 CI, donor를 향하는 선택 변화, AB/CD 재현이 있어야 강한 후속으로 확장한다. AB/CD는 모두 calibration하므로 미관측 답변 토큰 전이라고 부르지 않는다.

별도 비표적 연산에서는 최고 고정 E의 guardrail을 측정하고 90% CI가 사전 실용 범위 안에 들어야 동등성을 말한다. DET와 OUT의 차이가 유의하지 않다는 것만으로 DET≥OUT를 주장하지 않는다. 모든 E에서 DET 초과효과가 실용 범위 안으로 작으면 해당 detector-leverage 가설을 내리고, 구간이 넓으면 미결정으로 둔다. 에너지 crossing은 실제 관측 수준만 보고하고 비단조 곡선이나 범위 밖 미도달을 보간하지 않는다.

**제안 비용:** 새 discovery8·calibration16·confirmation48, 두 규칙×두 기호배정×두 코드=8변형. 보조 BF16·비표적 guardrail까지 기본 11,776 forward, cap 14,000회·2시간. 필요 개념은 활성 순위, 잔차 norm, paired donor 대비다. 모든 수치는 제안 예산이다.

**기여 경계:** [Language-Specific Neurons](https://arxiv.org/abs/2402.16438)와 [Sparse Neuron Circuits](https://arxiv.org/abs/2601.22594)가 이미 뉴런 집합의 인과 제어를 다룬다. 따라서 “묶으면 답이 바뀐다”는 결과는 부족하다. 같은 에너지·OUT 기하를 넘는 선택성, 비표적 guardrail, 실제 donor 방향 선택이 함께 있어야 새 기여를 검토할 수 있다. 그 결과도 좌표 간 synergy나 분산 회로 전체의 식별을 자동으로 증명하지 않는다. **Novelty UNVERIFIED**다.

## 조건부 3위 R4-C — 미관측 개입 반응의 독립 예측

GPT는 이를 1위로 유지했고 Claude는 수치 감사 뒤로 보류했다. 우리 순위에서는 **조건부3위**다. G0 뒤 Q 반응의 실제 변동이 충분히 측정 가능할 때만, 별도 P 패널로 새 과제 생성기·새 Q 좌표의 반응을 예측한다. Candidate-dependent norm 매칭을 금지하고 예측을 Q 실행 전에 hash/commit한다. Baseline 대비 held-out 표준화 loss 감소와 과제 내 P 교환 대조가 primary이며, 확증 refit을 하지 않는다.

BF16에서만 성공하면 연산 설정에 특이적인 반응 예측이다. 정밀도 일반화·의미 기능·호출 절감을 주장하지 않는다. 변동이 floor 아래이면 식별불가, 변동은 충분하지만 예측 이득의 CI가 실용 범위 아래로 좁으면 전이 가설 미지지다. 새8/24/64 family의 기본 3,456 forward, cap 4,500회·1시간을 제안한다. 정확한 전이 조합의 신규성은 미확정이며, [SPARK](https://arxiv.org/abs/2607.10296) 및 [Per-Instance Steering](https://arxiv.org/abs/2608.08829)과의 구별이 필요하다.

## 보류·수정한 제안

| 제안/해석 | 최종 반영 |
|---|---|
| 후보/placebo 비율 CI가1을 포함하면 동등 | 철회. 분모0 위험을 피하고 직접 family 차이와 사전 실용 구간을 사용 |
| 전체12B FP64 인증 | 제외. 대수적 등가와 국소 FP64 spot audit만 구분해 사용; 전체 출력 오차 보증 아님 |
| Placebo가 비슷한 크기를 만들면 모든 뉴런 효과가 artifact | 제외. 산술 계산 그래프 대조이며 BF16의 실제 효과를 무효화하지 않음 |
| Paired 상쇄 선행을 P0가 반증 | 제외. 서로 다른 estimand·모델·조건이며 보편 상쇄 정리가 아님 |
| k 증가에 따른 효과만으로 연합 기제 | 제외. 에너지·OUT·RAND·비표적·정밀도 대조 필요 |
| 단일 좌표 signed 내용/기호 분해 재개 | 보류. 수치 gate 실패와 직접 선행 겹침 |
| 작은 하류 donor 규칙 스위치 | 신뢰 가능한 선택적 자연 신호가 먼저 있어야 하는 고위험 후속 |

특히 [Cross-Encoding Steering](https://arxiv.org/abs/2608.22985), [Decide/Bind](https://aclanthology.org/2026.findings-acl.1144/), [What a Model Computes From](https://arxiv.org/html/2606.29522v2)가 내용·기호·내부 상태 사용을 이미 구별한다. 새 경로 조작 없이 같은 분해를 다른 모델에 적용하는 것은 독립 핵심 기여로 내세우지 않는다.

## 실행 의사결정

추천 순위는 논문 질문의 우선순위이며, 실행은 G0부터다. 그 뒤 수치적 공유 패턴이 측정되면1위, 새 자연 donor 연합에 안정적인 선택 효과가 있으면2위를 소규모 discovery로 검토한다. 어느 경우에도 현재 P0를 새 확증으로 재사용하거나 기존 null이 양성이 될 때까지 dose를 조정하지 않는다. 두 AI의 동의나 검색에서 미발견은 신규성의 증명이 아니다. 정확한 코드·manifest·threshold를 고정하기 전에는 사전등록이나 실행 완료로 표시하지 않는다.

출처 확인은 각 원전의 관련 부분에 한정한 조사이며 체계적 문헌조사 전체가 아니다. 상세 접근 범위는 [원전 audit 목록](../logs/summary/round3a_peer_review.json)에 기록했다. 수치·예산·초안 상태는 해당 JSON/YAML에서 검산 가능하다.
