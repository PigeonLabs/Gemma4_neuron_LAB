# R9 재개와 현재 증거

목표는 신규성이 있고 검증 가능한 논문 주제의 발견이다. 이전 R8의 [실패 종료](AUTONOMOUS_DISCOVERY_OUTCOME_KO.md)와 R1–R7 실측·기각 판단은 보존한다. 재개를 성공으로 해석하지 않는다.

GPT 독립 Pro 웹 검토31분44초, Claude Opus5.5 Extra Research19분21초(UI241 sources), 이어진 두 실제 적대적 교차답변을 모두 회수했다. 전체 실제 원문·실패한 요청·완료 화면은 로컬 runs/round9_peer_review/에 보존한다. source 수는 실제 원전 독서 범위를 보장하지 않는다.

GPT의 Deep Research 요청은 오류로 끝났고 완결 답변은 일반 Pro의 원전 검토다. Claude는 전체 공통 독립 패킷, GPT는 오류 복구 후 동일 핵심 실측·기각 범위의 짧은 재진술과 기존 채팅·공개 보고서를 받았다. 독립 단계의 전송 바이트가 같았다고 하지 않는다. 두 독립 답변 완료 후 보낸 교차 패킷은 실제 동일하다. Claude는 교차 단계에서 새 원전을 읽지 않았다고 명시했다.

두 심사 모두 논문 Go는0개다. GPT는 정보 분할을1순위, Claude는 생성 과정 선택성을1순위로 뒀다. Claude는 중립 예시를 수정해 남겼고 GPT는 신규성 부족으로 지지하지 않았다. Codex는 기존 ICL 풀 중복 때문에 그 후보를 제외했다. 잘못된 선행 범위·호출 수·weak sampling 정의, BF16–FP32 floor와 제한 log-prob의 해석을 수정했다.

Codex는 정보 분할·보상 재배분·수집 과정 순으로 우선 실행한다. 고차 확률·자원 경쟁은 외부 Hold를 보존하면서 정확 CPU oracle와 비동형 입력 공간이 확보된 뒤 작은 조건부 P0만 배정한다. 외부 모델의 동의를 과학적 증거로 세지 않는다.

[다섯 질문의9항목](ROUND9_P0_PREREG_KO.md), [동결 프로토콜](../configs/round9_P0.yaml), [사전등록](../logs/summary/round9_P0_registration.json)을 따른다. 현재 수치의 원본은 [진행 JSON](../logs/summary/round9_progress.json)이다. [문헌 범위](../logs/summary/round9_literature_audit.json), [심사 기록](../logs/summary/round9_peer_review.json)에 접근 한계·이견을 남긴다.

기본 anchor 실패 시 main은 NOT_EVALUATED다. main 오답 자체를 준비도 실패로 제거하지 않는다. 전체 어휘 top1·유효 질량을 보존하고 제한 점수로 형식 실패를 구제하지 않는다. P0 양성도 논문화 검증이 아니며 P1/P2는 별도 등록·실측 전까지 NOT_EVALUATED다.
