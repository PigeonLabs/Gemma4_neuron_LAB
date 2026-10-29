# R6 과제 준비도 실패와 1회 한정 별도 과제 등록

원래 R6-A (`round6_A_discovery`)는 실제 12B 모델에서 기술 검증 34회와 과학적 forward 192회를 실행했다. BF16 훅 동등성과 동일 BF16 저장값의 FP32 상향 변환, RMSNorm 기준 분모 재생은 통과했다. 충돌 과제에서 직접 경로 26/32, 합성 경로 7/32, 단독 직접 경로 27/32, 단독 합성 경로 25/32, 코드 변환 63/64였다. 사전 준비도 기준을 충족하지 못해 `STOPPED_AT_TASK_READINESS`로 종료했다. 뉴런 탐색·calibration·confirmation은 `NOT_EVALUATED`다.

원래 결과·설정·등록은 보존한다. 실패 결과를 본 뒤 설계를 바꾸었으므로 후속 A를 별도 연구 `R6_A2_selector_bottleneck`로 명시한다. 표를 8행에서 4행으로 줄이고, 숫자 키 번역을 요구하던 방식에서 선택할 경로를 상단 지시문에 명시하는 과제로 변경했다. 새 seed와 nonce 세계를 사용하고, discovery/calibration/confirmation을 독립 분할했다. B/C에는 이 시점까지 실제 모델 출력이 없으며, 첫 측정 전에 동일한 4행 계열로 등록한다.

이번 변경은 최대 1회다. 준비도·실제 행동 손상·코드 보존·무작위 대조·calibration 기준을 낮추지 않는다. 기존 A의 226회를 전역 예산 및 A의 총 예산에 합산한다. 전역 52,000회/4시간 상한은 유지한다. 발견 단계가 통과해야 고정 후보의 calibration 및 confirmation을 별도 등록하고 실행한다. 실패 조건에서는 조건부 후속 단계를 `NOT_EVALUATED`로 남긴다.

명시적 경로 과제에서 경로 지시는 상단 지시문에 있고, 원래 `CUE_BEGIN` 영역에는 숫자 키가 남아 있다. 따라서 숫자 cue 교체는 그 필드의 참고 대조이며 전체 경로 지시 교체나 성공률 상한으로 해석하지 않는다. query 범위의 전달과 native 좌표 개입을 주 분석으로 유지한다.

입력·출력 행과 활성값은 로컬에 보존한다. 공개 파일에는 집계와 해시만 포함한다. 실제 수치는 `logs/summary/round6_A_discovery.json`, 별도 과제 사전 등록은 `logs/summary/round6x_A_discovery_registration.json`에서 확인할 수 있다.
