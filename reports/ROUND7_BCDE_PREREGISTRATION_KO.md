# R7 B–E 실행 프로토콜 고정

A의 실제 모델 출력과 분리해 각 후보의 입력·코드·분석·예산을 최초 B–E 출력 전에 해시 등록했다. 기존 R2–R6 파일 91개는 보호한다. 모델 revision과 BF16 원본값의 FP32 상향 계산은 A와 같고, 모델 재로딩마다 BF16/FP32 훅 기술 자격 34회를 별도로 실행한다. 여러 개입 seed와 응답 코드 행은 독립 세계 수를 늘리지 않는다.

| 후보 | 독립 P0 세계 | 별도 보정 세계 | 최대 science forward | 준비도 실패 시 |
|---|---:|---:|---:|---|
| B 봉인된 질의 전 token 경로 | 12 | 미사용 | 1228 | 경로 기술/lookup/code 실패에서 중단 |
| C 최종 상태 동치의 취소 이력 영향 | 16 | 미사용 | 1184 | final-only direct/compose/code 실패에서 중단 |
| D 답과 독립인 prefix 순서 | 16 | 미사용 | 672 | forward/inverse/code 실패에서 중단 |
| E 개입 계열의 전향적 반응 예측 | 20 | 6 | 1392 | 보정 과제/효과 envelope/readout parity 또는 P0 준비도 실패에서 중단 |

각 재로딩의 technical cap은 64회이며 실제 표준 기술 검사는 34회다. B의 봉인 그래프 추가 instrument 4회와 E의 보정 312회는 science budget 안에서 별도 phase로 기록한다. 전체 다섯 후보 P0 한도는 8000회다.

B의 허용 경로는 1개 중립 마침표 토큰이다. 답 suffix에서 그 토큰을 제외한 **모든 prefix 위치**로 가는 attention edge를 48층 모두에서 차단한다. 원문과 donor가 달라도 carrier까지 차단한 null graph의 logits가 같아야 한다. 두 key 중 하나만 바꾼 donor의 동일 key/다른 key 효과를 비교하고, 전제 이전·첫 행 경로 대조를 둔다. 0–46층 token 궤적 전체 교환이며 작은 단일 상태나 자연 경로의 필요성을 주장하지 않는다. 긴 간격은 padding 1000 token이다.

C는 권위 있는 최종 F/G 표를 byte 단위로 고정하고, 취소된 옛 값 두 종류와 무관 key의 옛 기록만 바꾼다. 관련−무관 이력 효과의 compose−direct 상호작용을 최종표 앞/뒤에서 모두 검사한다. 단순 수정 성공률은 RippleEdits와 중복되어 논문 명제로 삼지 않는다.

D는 target과 독립인 prefix의 오름/내림 순서를 바꾼다. 전체 P0의 prefix token-ID 다중집합이 정확히 같은지 CPU로 검사한다. forward/inverse 과제의 반대 부호 선택성이 각 코드북에서도 유지되어야 한다. 순서와 n-gram을 독립된 원인으로 분리했다고 주장하지 않는다.

E는 별도 6세계에서 고정 dose의 네 seed씩 native MLP/attention/잔차 반응을 측정한다. 과제·clean margin·길이·세 site norm·clean 최종 norm/head/softcap의 local isotropic readout 민감도를 baseline으로 쓴다. 보정 ridge 계수를 동결한 뒤 P0의 MLP source 반응으로 attention/잔차 target을 예측하고, **해당 target 실행 전에 예측 파일을 저장**한다. P0 refit이나 target 출력에 맞춘 dose 변경은 없다. 두 과제 모두의 개선과 세계 단위 CI, 과제·코드 내 source shuffle을 요구한다. 양성도 공통 인과적 병목의 증명은 아니다.

## A에서 확인한 프로토콜 결함

실제 A P0의 144행 중 R1 명시 anchor 48행에는 R0을 따르는 새 예시가 남았다. 두 지시가 모순되므로 R1 점수를 순수 규칙 실행 능력으로 해석할 수 없다. 제 설계 오류이며 원본을 수정하거나 결과를 숨기지 않는다. 모순이 없는 R0 near 17/24, far 8/24도 원래 85% 기준보다 낮지만, 사후 제외를 적용한 재판정이나 재실행은 하지 않는다. 본문 논리 footprint 1152행, P1/P2는 NOT_EVALUATED다.

[CPU 입력 감사](../logs/summary/round7_input_audit.json) · [A 원시집계](../logs/summary/round7_A_P0.json) · [실행 driver](../scripts/run_round7.py)

각 후보의 수치 기준·고정 seed·source hash·split manifest hash는 [B](../logs/summary/round7_B_P0_registration.json), [C](../logs/summary/round7_C_P0_registration.json), [D](../logs/summary/round7_D_P0_registration.json), [E](../logs/summary/round7_E_P0_registration.json)에 있다. 전체 입력과 캡처 활성은 로컬 runs/에만 둔다. 신규성은 모두 UNVERIFIED이며 현재 실험 승인과 논문화 Go는 다르다.
