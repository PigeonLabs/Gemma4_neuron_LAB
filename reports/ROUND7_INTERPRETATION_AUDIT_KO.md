# R7: 등록 판정과 해석의 경계

원시 요약과 등록 gate는 바꾸지 않는다. 다음은 결과 이후 수행한 CPU 진단이며 독립 확증이나 새 성공 판정이 아니다.

- A: R1 anchor 48행의 규칙 지시와 예시가 모순된다. main footprint는 미측정이다. R1 점수를 순수 실행 능력이나 규칙 일반화의 증거로 쓰지 않는다.
- B: 등록된 Negative는 이 특정 봉인·궤적 교환 프로토콜의 .1 nat 방향 기준에 대한 판정이다. 봉인 후 full-vocab 코드 출력은 late-short 0/24, late-long 2/24로 무너졌다. 단일 중립 토큰의 일반 저장 용량이나 자연 모델의 기억 부재를 입증하지 않는다. 코드표와 응답 지시는 suffix에 있어 차단되지 않았음을 실제 249개 고유 encoding의 토큰 위치로 확인했다. 앞쪽 task cue는 차단되며, 봉인 안에서 명시적 답을 주는 format 양성 대조는 미측정이다. 따라서 실패 원인을 코드표 차단으로 단정하는 설명도 틀리다.
- C: 취소된 값 old1 대 old2의 logit은 움직였다. 이 둘은 정답이 아닌 값이다. 정답과 full-vocab top1도 1024행 모두 기록했다. 같은 history 범위·위치에서 old1/old2만 바꾼 512쌍 중 top1 변경은 7쌍, correctness 변경은 6쌍이고 현재 언급된 취소 값을 실제 선택한 행은 0개다. 합성 특이적 증폭의 before CI는 넓고, after CI 상한은 .1 아래다. 앞/뒤 모두를 요구한 등록 명제는 지지되지 않았지만 효과 0이나 동등성을 뜻하지 않는다. 무관 이력 값이 최종 무관 값과 우연히 일치할 때 생기는 편향의 방향은 데이터나 추가 가정 없이 확정할 수 없다.
- D: 32개 추상 질의 중 14개는 forward와 inverse가 같은 답이다. 이는 oracle 모순이 아니지만 연산자 사용을 구분하지 못하는 진단상의 약점이다. 서로 다른 답을 내는 inverse 36개 코드 행의 정답은 7, forward 답 추종은 24였다. 본 priming은 미측정이다. 단순 역조회 실패를 새로운 발견으로 제시하지 않는다.
- E: 여섯 보정 세계의 선형 ridge를 고정해 새 20세계·80행에서 예측한 알고리즘의 등록 ΔR²는 Negative다. 여섯 보정 세계를 24개의 독립 세계로 세지 않는다. 이 결과는 모든 가능한 예측기나 공통 취약성의 부재를 말하지 않는다. 과제별 raw Spearman과 동결 baseline을 뺀 target에 대한 사후 연관 진단도 계산했다. target만 residualize한 진단은 partial Spearman이 아니며, 양수이면 보정 실패·0 근처이면 상관 부재라는 이분법은 성립하지 않는다. 현재 P0에서 재적합하지 않았다.

## 실제 렌더링된 입력 분할

서로 다른 latent seed/world가 서로 다른 모델 입력을 보장하지 않았다. B의 등록 12개 세계는 실제 main view로는 11개다. 같은 view를 합쳐 11개에 같은 가중치를 준 사후 bootstrap에서 late-short CI는 [-.17892,.08613], late-long은 [-.16301,.09948] nat다. 후자는 .1 경계에 가깝다. 등록된 12-world 결과를 그대로 보존하고 두 결과를 함께 제시한다.

B의 미사용 P2에는 P0와 같은 전체 view 1개가 있고, D의 미사용 P2에는 P0와 같은 prompt text가 112개 있다. E는 P0 80개의 조건 입력이 78개 고유 text이며 실제 사용한 calibration과 P0 사이의 prompt 겹침은 0이다. 미사용 P2에는 P0와 같은 text 6개가 있다. P1/P2는 실행하지 않았으므로 이 자료를 독립 확증으로 주장한 적이 없다. 향후 연구는 최초 모델 출력 전에 latent world와 **렌더링된 prompt** 양쪽으로 분할 중복을 막고, 이 라운드의 예약 분할을 새 확증으로 그대로 사용하지 않는다. 기존 manifest를 사후에 조용히 교체하지 않는다.

[검토 의견의 사실 확인](../logs/summary/round7_peer_factual_checks.json) · [렌더링 분할·고유 view 민감도](../logs/summary/round7_rendered_split_audit.json) · [등록 원시 결과](../logs/summary/round7_execution_results.json)
