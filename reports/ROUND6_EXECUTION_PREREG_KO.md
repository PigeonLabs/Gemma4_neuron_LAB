# R6-A/B/C 실행 전 등록

사용자가 세 실험의 실행을 요청했다. 기존 [공동 검토·추천](ROUND6_RECOMMENDATIONS_KO.md)과 [초안](../configs/round6_plan.yaml)은 당시 기록으로 보존한다. 이 [실행 설정](../configs/round6_execution.yaml)은 모델 출력을 보기 전에 세 과제의 oracle, 후보 탐색, 손상·전이·복원 및 조건부 gate를 구체화한다. R5의 차단·gate·중단 결과는 변경하지 않는다.

모든 세계와 연관 donor는 새 seed에서 생성한다. R5의 미사용 calibration/confirmation도 재사용하지 않는다. 각 연구는 discovery 12 / calibration 16 / confirmation 64 독립 episode를 가진다. 같은 세계의 질의·함수·donor·코드 교환은 같은 분할에 묶는다. 후보와 token scope는 발견 자료로만 고정한다. 원시 입력·활성값·모델 출력 행은 로컬 runs에만 둔다.

## 단계와 예산

공통 주 모델은 pinned `google/gemma-4-12B-it` revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`다. 저장된 BF16 값을 전체 FP32로 upcast하고 가중치를 고정한다. Native MLP 좌표는 down_proj 입력, attention 대조는 o_proj 입력의 head 범위로 따로 이름 붙인다. 실제 BF16/FP32 훅·복원·제거·tuple output·tied-head/softcap·RMSNorm replay를 먼저 검사한다. 주 실행은 use_cache=False, 1회 답마다 전체 prefix forward 1회이며 donor/독립 복원/재생의 호출도 모두 센다.

초기 실행 범위는 단계별 source·manifest hash를 등록한 **discovery feasibility**다. 발견 gate가 통과하면 해당 후보와 대조 identity를 그대로 고정하고, 별도로 실행 코드를 등록한 뒤 calibration/confirmation을 진행한다. 발견 통과만으로 실험 전체를 끝내지 않는다. 발견 gate가 실패하면 그 연구의 조건부 단계는 NOT_EVALUATED로 남긴다. 확인 표본을 연 뒤 후보·가설·임계값을 바꾸지 않는다.

| 연구 | 발견 cap | 전체 조건부 cap | 주 estimand |
|---|---:|---:|---|
| A 선택 병목 | 2500 | 18800 | 받는 문제의 대체 답에 대한 반대-selector−같은-selector donor 효과에서 무작위군의 같은 차이를 뺌 |
| B 정답 교정/출처 추종 | 2500 | 15000 | persistent L×R 차단×대체표 일치/충돌의 상호작용 및 특정 대체값 전달 |
| C 중간값 재사용 | 2000 | 17500 | 받는 g(y_donor) 대 donor 최종 답의 효과가 새로운 g에서도 남는지 |

전체 cap은 기술 검증 포함 **52000 actual forwards 또는 4시간**, run당 디스크 4GB, 실제 GPU reserved 80GB다. 각 연구의 조기 중단 뒤 다음 독립 연구를 계속한다. 모델/새 훅을 재로딩하면 기술 검증을 반복하고 그 비용도 기록한다. 실제 최초 50회 완료 비용과 장치·dtype·seed·환경·코드 해시·GPU/CPU 자원을 각 run에 남긴다.

## A/C 발견에서 고정할 선택

반대/같은-selector donor는 같은 donor 세계·코드표에서 cue만 다르다. 받는 문제의 두 답과 donor의 선택·미선택 답 및 응답 문자는 충돌 회피 제약 아래 구분한다. C는 직접 shortcut을 없애고 donor y′에 대한 받는 g0/g1의 결과와 donor 최종 답을 다르게 만든다.

6개 8층 구간×attention/MLP×사실/cue/query 위치의 자연 상태 교환을 탐색한다(C는 사실/query). Query 구간의 후보만 작은 병목의 주 대상이다. 가장 높은 query 대비 구간에서 8개 단일 층을 실제로 비교하고, donor 차이×실제 projection 열 norm의 절대/일관된 신호로 정한 128/512 MLP 좌표 또는 1/2 attention-head 집합 네 개를 실제 유한 개입으로 검증한다. 전수 역전파는 사용하지 않는다. 이 순위는 warm start이며 각 집합의 실제 전이·선택 변화·query-zero 손상으로 자격을 판단한다.

발견의 margin 기준은 .10 nat, 행동 대비 .10이다. Task readiness는 .85, code-only는 .95를 요구한다. Code-only 정확도 손상≤.05 및 평균 절대 margin 변화≤.20 nat를 확인한다. 기술 noise 바닥과 이 실용 목표는 구별한다. 그룹은 하나만 선택하며 발견 안의 후속 세계와 무작위 3군·연속 code guard에서 다시 검사한다. 실패 뒤 다른 후보로 몰래 교체하지 않는다. 큰 residual에서만 움직이면 작은 native 병목을 확인했다고 하지 않는다.

독립 복원은 다른 입력에서 가져온 같은-selector/올바른 중간값의 상태를 query G에 넣는 별도 개입이다. 원래 값을 같은 forward에서 되돌리는 항등 복원은 기술 검사다. G를 donor 값으로 다시 여는 arm이 계속 zero라고 주장하지 않는다. Calibration에서만 무작위군의 전역 attenuation을 실제 local postnorm 쓰기 크기에 맞추며, 확인 출력에 맞춰 대조를 재선택하지 않는다.

## B의 별도 자격

R5나 A의 집합이 곧바로 적격 주경로라는 가정을 하지 않는다. B의 새 발견 자료에서 primary를 찾고, 적격할 때만 remaining backup을 찾는다. Direct-only 손상≥.20, composition-only 손상≤.10, code guard, agreement dual 보존을 요구한다. B의 primary는 모든 token에서 계속 zero이며 remaining R만 질의 구간에서 다룬다. 정답 oracle은 authority인 h에 고정한다. 일치/두 다른 오답 조건을 구분해 generic damage와 특정 대체값 추종을 나눈다. Primary 또는 backup 자격이 없으면 각각의 gate에서 중단한다.

모든 확증 대비는 episode 안에서 질의·donor·코드를 먼저 평균한다. 세 주 연구의 multiplicity는 Holm, 열지 않은 연구의 p는 1로 둔다. 64 episode는 확정된 power 보장이 아니다. 효과 상한이 사전 실용 목표 미만인 방향 미지지, 넓은 구간의 미결정, 자격·도구 실패의 식별불가를 구분한다. Norm replay는 등록한 첫 4 confirmation episode와 codebook 0의 별도 modified-graph 민감도 분석이며 모든 대안을 제거하지 않는다.

현재 이 문서의 수치는 사전 설계 값이다. 실제 모델 결과는 별도 run 요약과 보고서로 기록한다.
