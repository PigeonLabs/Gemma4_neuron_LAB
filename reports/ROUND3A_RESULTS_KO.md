# R3-A 실측: BF16/FP32 수치 검증에서 중단

**추천 1번을 실제 모델에서 시작했으나, 네 후보 모두 사전등록한 수치 안정성 기준을 통과하지 못했다.** 계획대로 P0에서 중단했으며 discovery·calibration·confirmation은 실행하지 않았다.

새8 source family의32변형에서 **576개 조건 행·699회 실제 모델 forward**를 측정했다. 훅·복원 검증과 실제 nonzero delta 전달은 모두 통과했다. 중단 원인은 기술 오류나 누락이 아니라, 현재 dose·위치·과제에서 효과가 dtype을 넘어 재현되지 않은 것이다.

| 후보 | BF16 평균 절댓값 (nats) | FP32 평균 절댓값 (nats) | Signed-effect Spearman | FP32 절댓값≥.02 입력 | 판정 |
|---|---:|---:|---:|---:|---|
| H001 L15N9637 | 0.044800 | 0.000199 | 0.321 | 0/32 | 미통과 |
| H002 L22N7859 | 0.041870 | 0.000133 | 0.096 | 0/32 | 미통과 |
| H003 L26N719 | 0.039124 | 0.000144 | 0.189 | 0/32 | 미통과 |
| H004 L35N990 | 0.036865 | 0.001019 | -0.378 | 0/32 | 미통과 |

d는 내용축 u−v에서 (m_plus−m_minus)/2로 정의한 부호 효과다. 32는8개 family의 규칙/기호 변형 수이며 독립 n=32가 아니다. 위 수치는 이 고정 표본의 기술 통계이며 모집단 신뢰구간이나 의미 기능 검정이 아니다.

![R3-A P0 수치 비교](../figures/round3a_p0_precision.svg)

왼쪽은 동일 nominal dose에서의 평균 효과 크기(가로축 로그), 오른쪽은32 paired 효과의 순위 상관이다. 점과 사각형으로 dtype을 구분했다. .70 선은 실행 전에 정한 수치 gate이며 유의성 기준선이 아니다.

## 어떤 기준이 실패했나

각 후보에서 FP32 부호 효과의 절댓값이0.02 nats 이상인 입력이 최소8개 필요했지만 모두0개였다. 따라서 해당 입력의 부호 일치율은 **NOT_IDENTIFIABLE / 해당 없음**이며, 일치율0%로 바꾸어 보고하지 않는다. Signed Spearman도 모든 후보가 등록한0.70에 미달했다. 각 후보의 평균 절대 오차는 허용 상한0.05 안에 들었지만, 모든 조건을 함께 충족해야 하므로 다음 단계로 진행할 후보는0개다.

BF16과 FP32 모두 좌표 개입의 actual delta는100% nonzero였다. 원 BF16 smoke28검사, 정렬 hook34검사, 두 dtype의 read-only/동일-forward 복원/훅 제거 exact 검사 및 finite 검사를 통과했다. 즉 개입이 전달되지 않아서 작은 FP32 효과가 나온 것은 아니다.

## 해석할 수 있는 범위

FP32는 같은 BF16 저장 가중치와 부동 버퍼 값을 올려 계산한 조건이다. 별도 native FP32 가중치의 결과나 실수 연산의 참값은 아니다. 전체 모델의 연산 dtype을 바꾸었으므로 차이를 마지막 logit의 반올림, 중간 계층의 비선형 전파, 개입 자체의 반올림 중 하나로 국소화하지 못했다. 관찰된 BF16 변화는 BF16 모델의 실제 인과적 변화지만, 이를 정밀도에 안정적인 의미 기능으로 해석할 근거가 부족하다.

BF16에서만 보이는 변화 전부가 무작위 잡음이라는 결론, 네 좌표의 기능 부재, 효과의 정확한0, 다른 위치·dose·과제에서의 무효는 주장하지 않는다. 적은 새 표본으로 R2의 원시 수치나 등록 검정을 소급 무효화하지도 않는다. R2-E의 예측 개선은 등록한 BF16 범위의 결과로 보존하고, 정밀도에 일반적인 공통 원인·의미 기제 해석은 보류한다.

## 행동과 미실행 단계

| 조건 | Clean 정답 | 개입 후 다음 토큰 변경 |
|---|---:|---:|
| bfloat16 | 28/32 (강제 선택·raw top-1 동일) | 0/256 후보 개입 행 |
| float32 | 28/32 (강제 선택·raw top-1 동일) | 0/256 후보 개입 행 |

각 dtype의 clean 오답 변형은2개 source family에 포함됐다. 이는 P0 표본의 기술 통계이며, 후속 confidence 식별 gate나 확증 결과로 사용하지 않는다. 본 과제의 자유 생성 행동은 측정하지 않았다.

| 단계 | 상태 |
|---|---|
| P0 실제 12B 비교 | 실행 PASS · 수치 gate 미통과 |
| 내용/기호/규칙 discovery | NOT_EVALUATED |
| β·dose·대조 calibration | NOT_EVALUATED |
| 독립64 family confirmation | NOT_EVALUATED |

P0 실패 뒤 dose를 올리거나 판정선을 낮추어 같은 연구를 계속하지 않았다. 추천2번도 자동 실행하지 않았다. 이 방향을 재개하려면 정밀도를 명시한 별도의 새 연구로 설계해야 한다. 우선 검토할 질문은 “좌표 개입의 의미 효과가 연산 정밀도에 얼마나 의존하며, 어느 연산 구간에서 커지는가”다. 이는 이번 결과에서 나온 후속 가설이며 아직 검증하지 않았다.

## 자원·재현과 검증

Run `20261029T190213_round3a_p0`. 본 run 54.24초, 모델 로딩6.51초는 별도다. 최대 allocated 47.997GB, reserved 49.543GB, 관측 최소 GPU 여유 49.718GB였다. 총699 forward는 기본576회에 smoke·generation/훅·복원123회를 더한 값이며 cap1,000회·1,800초 이내다. 단일 GPU 프로세스만 사용했다.

모델 `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, seed20261029, RTX PRO 6000 Blackwell Max-Q, eager/batch1, 최종 prefill chat-template 위치다. 실제 환경·코드/설정 SHA·모델 dtype·입력 위치·자원은 run metadata와 요약에 보존했다.

- [실행 전 등록](../docs/ROUND3A_PREREGISTRATION.md) · [수치 기준과 예산](../configs/round3a.yaml)
- [실측 요약](../logs/summary/round3a_p0.json) · [CPU 원시 재계산 검사](../logs/summary/round3a_p0_analysis.json)
- [실행 코드](../scripts/run_round3a_p0.py) · [정확한 대비/수치 gate](../src/neuron_lab/round3a.py)
- 원시 JSONL·baseline activation/logit cache·전체 prompt manifest는 로컬 `runs/`에만 보존한다.
- CPU 재계산: `.venv/bin/python scripts/analyze_round3a_p0.py --check`.
- 공개 paired 숫자만으로 gate 재계산: `.venv/bin/python scripts/replay_round3a_p0.py`. 원시 inference·전달량·hook을 독립 재검증하는 것은 아니다.
- 보고서·SVG 최신성: `.venv/bin/python scripts/build_round3a_report.py --check`.

CPU toy 시험은 대비식·분할·gate의 동작 검사이며 실제12B 결과와 구별한다. 실제 P0의 원시 SHA·등록 hash·과거 결과 보호·행렬 완결성과 예산 등 31개 CPU replay 검사는 통과했다. 새 연구의 Novelty나 출판 가능성은 이 비교만으로 확정하지 않는다.
