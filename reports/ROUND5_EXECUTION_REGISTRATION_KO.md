# R5 실행 등록: 기술 검증과 차단 자격 탐색

GPT Pro Deep Research·Claude Opus 5.5 Extra Research와 검토한 [설계](ROUND5_DESIGN_KO.md)를 실행한다. 연구 명제는 **고정된 차단 아래 오류 이력이 질의 시점의 다른 뉴런 집합에 대한 인과적 의존성을 바꾸는가**다. 이 문서는 모델 출력을 관측하기 전에 작성했다. 아직 명제를 지지하는 실측은 없다.

[실행 설정](../configs/round5_execution.yaml)과 [기계 판독 등록](../logs/summary/round5_registration.json)에 새 입력 manifest 해시·독립 에피소드 분할·seed·코드 해시를 남긴다. 기존 설계·R2/R3 실측은 보존한다. 예시 입력을 확증에 재사용하지 않는다. 모델은 같은 pinned revision의 저장된 BF16 값을 전체 FP32로 upcast하며, BF16/FP32 모두 실제 down_proj 입력 훅 검증을 통과해야 한다. 가중치를 학습하지 않는다.

먼저 technical 3 episode에서 dtype별 8조건씩 48회의 전체 forward로 read-only·identity·모든 토큰 zero·query-only zero·같은 모듈의 동시 차단·동일 forward 복원·훅 제거·최종 tied-head/softcap 재생을 확인한다. Scientific readout은 캐시 없이 매번 전체 문맥을 재계산한 A/B/C/D 제한 선택이다. 전체 어휘 top1을 별도로 저장한다. 최대 입력은 1000토큰이며 잘라내지 않는다.

Discovery는 8개의 새 episode에서 각각 고정 순서의 test 질의 3개를 사용한다. 직접 표·두 단계 표·양쪽 표·코드 변환만의 4 witness, 총 96조건이다. Healthy는 dual/코드 변환 정확도≥.90, 단일 경로 각각≥.85를 요구한다. 과제 준비가 실패하면 차단 결과를 해석하지 않고 STOPPED_AT_TASK_READINESS로 끝낸다.

건강한 단일 경로 witness에서 **query 구간 평균 절대 활성값**의 (composition−direct)/(composition+direct+1e-8)을 구한다. 층 15/22/26/35, 크기 32/128, 두 방향의 16개 집합을 순위대로 고정한다. 동점은 좌표 번호로 정한다. 각 집합을 모든 토큰에서 계속 zero로 차단한다. 전체 탐색은 healthy 포함 96×17=1632 scientific forward다. 기존 설계의 72조건에 코드 변환 guard를 추가한 사전 수정으로 408회가 늘며, 전체 조건부 예산 31744회는 원래 32000회 cap 안이다.

두 차단의 discovery 점추정 자격은 각각 dual 손상≥.20, 반대 단일 경로의 손상 차≥.15, 코드 변환 손상≤.05, 두 dual 손상 차≤.10, 서로 겹치지 않는 좌표다. 오류를 예측하는 가시적 표 위치의 세 후보 특징과 방향은 discovery AUC로만 고정하며 AUC≥.70을 요구한다. 두 차단의 반대 경로 특이성은 단일 경로 witness 손상의 반대 방향으로 정의한다. 오류 예측 특징의 부호 자체가 서로 반대여야 한다는 뜻은 아니다. 최소 경로 특이성→최소 dual 손상→최소 AUC 순으로 두 집합을 선택하고 마지막 동점은 ID 순이다. 이 단계의 CI는 탐색용 기술 통계다.

적격 쌍이 없으면 **STOPPED_AT_DISCOVERY_LESION_GATE / bounded search에서 NOT_IDENTIFIABLE**이다. 차단 크기·층·과제·기준을 바꾸어 같은 연구를 통과시키지 않는다. 이는 모델 전체의 적응 능력이 없다는 증명이 아니다. Calibration/confirmation, 피드백 회복, query-time 모집, norm replay는 NOT_EVALUATED로 남긴다.

적격 쌍이 있으면 discovery에서만 남은 R 집합과 무작위 대조 identity를 선택하고 독립 calibration에서 사전 자격을 검증한다. 후보 재선정은 금지한다. 이후 확증은 기존 계획의 별도 freeze를 마친 뒤에만 연다. 전체 시간 4시간·실제 forward 32000회·run 산출물 4GB 중 먼저 닿는 한도에서 중단한다. 첫 50회 완료 시 실측 비용을 기록한다. GPU 계산 프로세스는 하나만 사용한다.
