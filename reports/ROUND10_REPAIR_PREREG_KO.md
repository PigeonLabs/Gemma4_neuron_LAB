# R10 A/B/C/E 계측 수리 — 과학 출력 전 등록

사용자가 R9의 A/B/C/E는 가설이 틀린 것이 아니라 평가하지 못한 것임을 지적하고, 협업을 통한 수리와 새 데이터 재실험을 명시했다. 네 질문을 유지하는 별도 R10 평가다. R9 원본·판정과 D는 변경하지 않는다. 생성 계산이 추가된 새 프로토콜이며, 관측하지 않은 과거 R9 최종답을 복원했다고 하지 않는다.

이 등록 전에 R10의 과학 calibration/P0 출력은0개다. 별도 기술 검증은58회 실제 forward로 완료했다. BF16/FP32 훅34회, cache/full-prefix16회, 생성 계수8회를 포함한다. 별도 기술검증의 두 고정 입력·네 위치에서 top1이 같았고 최대 logit 차이는4.25e−5 이하였지만 보편적인 수치 오차 바닥이 아니다.

[등록 JSON·입력/소스 해시](../logs/summary/round10_registration.json) · [예산과 판정](../configs/round10_repair.yaml) · [실제 독립·교차 검토](../logs/summary/round10_peer_review.json) · [CPU 검증](../logs/summary/round10_CPU_validation.json)

## 고친 부분과 보존한 질문

| 질문 | 수리 | main에서 남겨 둔 관측 대상 |
|---|---|---|
| A | 플레이어가 아닌 독립 감사자 역할, 같은 신호 규칙 재인 | 관측 정보와 양립하지 않는 전략의 선택적 과승인 |
| B | 합계 전용 anchor에서 경로 선택 명령 제거, 명시 숫자 비교 분리 | 총합 보존 재배분에 따른 실제 경로 선택 변화 |
| C | 정답 의미·문자·표시 위치 균형, 우도 정규화와 장치 규칙 분리 | 우도를 보존/변화시키는 수집 절차의 선택적 구별 |
| E | 개별 edge, distinct union, 재사용 규칙을 분리 | 국소 자격과 전역 배정 가능성의 선택적 불일치 |

주 질문의 정답이나 전체 일관성을 입장 조건으로 삼지 않는다. 기초 gate 실패 시 해당 질문의 main 전체를 열지 않으며, 세계별로 통과 사례만 남기지 않는다. calibration의 복잡한 질문에서는 최종답 판독률만 쓰고 main 정확도나 효과로 선정하지 않는다.

## 생성·파서·무효 응답

동일 저장 BF16 값을 FP32로 upcast한 계산 조건, 동결 모델, greedy, 단일 sequence, KV cache, 최대256 새 토큰, 재시도0회다. 전체 어휘에서 생성하며 A/B 강제·top-k/top-p·강제 EOS·기본 예약토큰 suppression을 사용하지 않는다. pinned tokenizer의 enable_thinking=False 입력은 빈 thought를 닫는다.

생성 token ID와 cleanup 없는 원문을 보존하고, 실제 종료 EOS 한 개만 마지막에서 분리한다. 내부 channel/control 토큰은 별도 기록 후 무효로 둔다. thought를 정규식으로 삭제하지 않는다. EOS가256번째에 자연스럽게 나온 경우는 절단이 아니다. EOS 없이 cap에 닿으면 무효다.

고유한 terminal 대문자 FINAL: 선언의 A/B 한 문자만 판독한다. colon 뒤 ASCII 공백0개 이상, CRLF와 바깥 공백은 허용한다. 중복 선언, 인용·열린 fence 안의 선언, 마지막 선언 뒤 본문, 다른 문자·Unicode colon은 무효다. 이전의 닫힌 계산 code block은 허용한다. 앞 설명과 최종 선언의 의미 일관성을 judge로 사후 변경하지 않는다. 과학적 관측은 등록한 최종 선언이다.

미판독의 semantic 값은 null이다. NO나 의미상 오답으로 대체하지 않는다. 전체 분모 N, 확인된 YES 수 y, 미판독 u일 때 YES율을 [y/N,(y+u)/N]로 두고, signed 대비의 최악 경계를 세계 안에서 먼저 계산한다. 세계 bootstrap의 outer95 구간은8–12개 탐색적 family에서 coverage를 보증하지 않는다. 조건부 유효 답만의 정확도는 진단값이다.

기초 anchor 확인 정답률85%(code95%)와 판독률95%를 요구한다. main 판독률95%는 전체와 개별 주요 요인 수준에 적용하며 작은 모든 교차 셀에100%를 강요하지 않는다. channel/cap 사유별 수치는 별도 보고한다. 95% 판독만으로 의미 효과가 식별되는 것은 아니며 worst-case 경계를 계속 적용한다. main 미실행은 NOT_EVALUATED, 실제 main 후 넓은 경계/구간은 측정한 Inconclusive다.

## 새 세계와 계산 예산

| 질문 | calibration 세계/응답 | 새 P0 세계/응답 | 미사용 확인 세계/응답 |
|---|---:|---:|---:|
| A | 4/192 | 8/1024 | 16/2048 |
| B | 4/160 | 12/864 | 16/1152 |
| C | 4/192 | 12/1152 | 16/1536 |
| E | 4/224 | 8/832 | 16/1664 |

사용·예약 R9 입력 11900개와 새 분할을 검사했다. 새 authoritative 입력 11040개는 전부 고유하며 기존 입력과 겹치지 않는다. 최대 413토큰이다. A/E는 정규화 동형 구조, B는 모든 재배분의 수치 orbit까지 기존 세계를 제외했다. C의 device 확률은 기존 R9 및 새 split끼리도 겹치지 않는다. 단순 rename을 새 family로 세지 않는다.

실행 상한은 calibration768＋P0 3872＝4640개 응답이다. EOS 포함 m개 생성 토큰은1 prefill＋(m−1)cached steps＝m actual forward이며 실제 모델 hook이 권위값이다. 최대 생성 1187840회＋공유 기술≤64회가125만 cap 아래다. 이전58회 기술 검증은 별도로 합산한다. 단일 GPU·8시간·run당4GB 상한도 적용하며 먼저 도달하면 BUDGET_STOP이다. 최악의 모든 응답을8시간 안에 마친다는 보증은 아니다.

동결 이후 P0 출력으로 prompt·parser·세계·gate를 바꾸지 않는다. 확인 분할은 NOT_EVALUATED이며 별도 등록 없이는 실행하지 않는다. 신선한 P0의 준비도가 실패해도 가설 Negative로 기록하지 않는다.

## 실제 협업과 Codex 판단

같은 공개 자료로 GPT Pro 웹 독립 검토6m39s와 Claude Opus5.5 Extra Research3m45s(UI219sources)를 받은 뒤, 사용자가 승인한 동일 교차 패킷을 양쪽에 보냈다. GPT 교차8m22s와 Claude 일반 Extra 교차 답변이 실제 완료됐다. GPT 별도 Deep Research 완료를 주장하지 않으며 Claude의 UI source 수를 본문 독서 수로 세지 않는다. Claude는 원래 소스에 접근하지 못했다.

Claude는 교차검토에서 A/B/E의 순환적 일관성 gate, C의 likelihood-ratio/4지선다 변경, 첫 토큰의 원인 단정, 채널 regex 삭제와 넓은 CI를 미실행으로 부르는 용어를 철회했다. E의 union 질문을 feasibility 질문으로 바꾸자는 후속 예시는 채택하지 않았다. 기존 binary union 대비를 유지한다.

표본수·cap에는 이견이 있다. GPT는256토큰·8개 탐색 세계부터, Claude는384 보정·12개 이상 세계를 권했다. Codex는 예산 행 수를 직접 검산해256토큰·A/E8개·B/C12개를 출력 전에 고정했다. 작은 family 수의 불확실성을 그대로 보고하고 통계적 일반화나 논문 Go로 확대하지 않는다. 수리 성공·가설 결과·논문 기여는 세 층으로 구별한다.

출력 전 최종 준비가 반복되는 B 합계 비교 anchor를 거부했다. 해당 세계의 실제 경로 문맥을 포함해 수정했고, 렌더링된 credits/verified totals/질문/코드 mapping에서 정답을 독립 재구성하는 CPU 검사를 추가했다. nonce로 중복을 숨기지 않았고, 과학 출력0인 단계에서 수정했다. CPU12개 검사가 통과했다.

외부 전송의 초기 자동 심사 차단과 공개 자료 대안, 사용자의 명시적 교차·후속 집계 전송 승인도 심사 JSON에 남겼다. 가중치·원시 전체 생성·인증정보는 전송하거나 공개하지 않는다.
