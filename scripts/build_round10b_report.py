#!/usr/bin/env python3
"""Regenerate the R10b registration/results narrative only from saved summaries."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT/'logs/summary'/name).read_text())


def main():
    reg = load('round10b_registration.json'); cfg = reg['scientific_protocol']
    old = load('round10_attempt1_abort.json'); peer = load('round10b_peer_review.json')
    rows = ['# R10b: A/B/C/E 새 계측 수리 사전등록', '',
        f"등록 시각 UTC: {reg['recorded_at_UTC']}. 새 R10b 모델 출력 관측 수: **{reg['registered_scientific_outputs_seen']}**.", '',
        'R9 A/B/C/E는 평가 자격 또는 판독 실패로 주요 가설을 평가하지 못했습니다. 이 재실험은 가설이 틀렸다는 해석을 바로잡고 네 질문을 새 데이터로 평가하기 위한 계측 수리입니다. R9의 원래 결과와 D의 좁은 등록 판정은 보존합니다.', '',
        '## 앞선 R10 실제 실패를 보존', '',
        f"R10 A calibration 완결 응답 {old['partial_calibration_responses']}개: VALID {old['parse_status']['VALID']}, TOKEN_CAP {old['parse_status']['TOKEN_CAP']}. 완결 응답 forward {old['recorded_completed_response_forwards']}회 외에 중단된 미완결 응답의 진입 {old['in_flight_forward_entries_without_completed_response']}회를 별도 비용으로 셌습니다. calibration 진입 {old['calibration_forward_entries']}회와 기술108회를 합쳐 기존 비용은 {old['all_R10_forward_entries_so_far']}회입니다.", '',
        '필수 그룹의 남은 답이 모두 유효해도 최종95% 판독을 달성할 수 없어 자기 프로세스만 중단했습니다. 이 중단은 원래 등록되지 않은 추가 공학적 결정이었습니다. 실제 FAIL·원시 행·해시를 보존하며, 어떤 P0도 열리지 않았습니다. 의미 정답 실패나 과학적 Negative가 아닙니다.', '',
        '## 실제 두 채팅의 독립·교차 심사', '',
        f"검토 상태: **{peer['design_review_status']}**. GPT Pro 웹과 Claude Opus5.5 Extra의 실제 독립 완결답을 서로 공유하기 전에 보존하고, 같은 교차검토문을 양쪽에 보냈습니다. 이번 Claude는 앞선 Research에 이은 일반 Extra 후속 검토이며 새 Research 작업이라고 부르지 않습니다. 양쪽 모두 로컬 원시 생성문이나 GPU를 독립 재현하지 않았습니다.", '',
        '양쪽은 간결한 최종답 지시와256토큰 한도를 권했습니다. Claude는 실제 파서에 대한 공백 불일치 추정, A를 통과해야 B/C/E를 실행한다는 제안, 추가2%·90% gate를 철회했습니다. 판독 가능한 terminal 선언과 한 줄 형식 준수는 별도 지표입니다.', '',
        '## 동결된 변경과 추정 대상', '',
        '출력 지시는 선택지 뒤에 한 번만 둡니다. 어휘·코드 logit 제한, 강제 답 접두사, 실패 재시도는 없습니다.', '',
        '```text',
        'Answer this question using its option code. Output one plain ASCII line in the form FINAL:<code>, replacing <code> with A or B. Output no other text.',
        '```', '',
        'A의 기초 순서 질문은 상태를 “안다”에서 행동 전에 상태를 “직접 공지받는가”로 좁힙니다. 신호 추론 가능성과 직접 공지를 혼동하지 않도록 사후 관측 후 anchor를 명료화한 것입니다. 원래 장문·절단의 원인으로 확정하지 않습니다. Main은 그대로 신호가 구분하는 정보와 전략의 구현 가능성을 평가합니다.', '',
        '각 문항은 대화 이력이 없는 독립 user prompt이며 KV cache도 호출마다 새로 만듭니다. Anchor의 명료화 문구가 다음 main의 문맥으로 넘어가지 않습니다.', '',
        f"추정 대상: **{cfg['measurement_scope']}**. R9/R10과 결과를 합치거나 동등성을 주장하지 않습니다. 새 세계와 출력 문구·A anchor가 함께 바뀌므로 이전 대비 개선의 원인을 특정 문구에 인과적으로 귀속할 수 없습니다. B/C의 오류는 산술·규칙 이해·판단 등의 대안 설명과 구별해야 합니다.", '',
        '기존 terminal parser를 그대로 사용합니다. 고유한 uppercase FINAL: 뒤 ASCII 공백0개 이상과 A/B 한 글자, 정상 EOS를 요구합니다. 설명 뒤 유효한 terminal 선언은 읽되 한 줄 준수로 분류하지 않습니다. 중복·인용·열린 fence·내부 제어토큰·실제 절단은 미판독 null이며 NO로 바꾸지 않습니다. 자연 EOS가 정확히256번째에 오면 정상 종료입니다.', '',
        '## 새 분할·고정 순서·예산', '',
        '| 질문 | 새 보정 세계 | 새 P0 세계 | 봉인 확인 세계 |', '|---|---:|---:|---:|']
    for s, detail in cfg['studies'].items():
        n = detail['splits']; rows += [f"| {s} | {n['calibration']} | {n['P0']} | {n['confirmation']} |"]
    rows += ['', f"R9 입력 {reg['prior_R9_input_rows_checked']}행과 R10 입력 {reg['prior_R10_input_rows_checked']}행을 대조해 새 token 입력 {sum(v[s]['presentations'] for v in reg['contracts'].values() for s in ['calibration','P0','confirmation'])}개 전체의 비중복을 확인했습니다. 최대 입력 {reg['maximum_input_tokens']}토큰. 사용·예약된 R9와 R10의 A/E 행렬, B 보상변환 orbit, C 확률 값과 분할도 배제했습니다. 문자 매핑2×선택지 순서2를 모든 의미 질문마다 교차합니다.", '',
        '실행 순서는 B → C → E → A입니다. 보정 문항은 세계와 기초 기능 종류를 고정 round-robin으로 섞습니다. 한 질문이 자격 실패해도 다른 독립 질문의 보정은 계속합니다.', '',
        '기초 정답률85%, 코드 정답률95%, 필수 그룹 판독률95%와 고정 복합 sentinel 판독률95%를 사용합니다. Sentinel의 과제 정답·과학적 효과는 진입 gate가 아닙니다. 개별 세계를 정답에 따라 버리지 않습니다. Main의 오류는 관측 결과로 남깁니다.', '',
        '보정의 각 응답 뒤, 예정된 최종 분모에서 남은 응답이 모두 완벽해도 기준을 달성할 수 없으면 그 질문의 보정을 중단합니다. 단순 미판독 futility는 V+(N−n)<ceil(.95N)입니다. 확인 정답률도 같은 최종분모 상한으로 기초 자격만 검사합니다. 복합 sentinel은 판독률 상한만 검사합니다. 조기 성공은 선언하지 않습니다.', '',
        'R10b에서 자격 실패한 질문은 추가 문구 수정·384토큰 연장·재시도·R10c 없이 종료합니다. 공통 기술 실패나 전체 예산 소진은 별도 전체 중단입니다. 시간 때문에 실행하지 못한 질문은 BUDGET_STOP/NOT_EVALUATED로 기록합니다.', '',
        f"최대 새 응답 {reg['maximum_generated_responses']}개, 최대 생성 forward {reg['maximum_generation_forwards']}회와 기술≤{cfg['technical_call_cap']}회. 실제 새 전체 상한 {cfg['max_actual_forward_calls']}회, 기존{cfg['prior_actual_forward_entries']}회 포함 {cfg['max_actual_forward_calls']+cfg['prior_actual_forward_entries']}≤{cfg['cumulative_forward_cap']}회입니다. 기존 실측 GPU {cfg['prior_measured_GPU_seconds']:.6f}초+새{cfg['max_seconds']}초≤{cfg['cumulative_GPU_seconds_cap']}초. 모델 로드·기술·생성 단계를 새 전체 시간에 포함합니다.", '',
        f"모델 revision `{cfg['model_revision']}`, 같은 BF16 저장 가중치의 FP32 upcast, RTX PRO6000 Blackwell96GB, seed{cfg['seed']}, corpus seed{cfg['corpus_seed']}. 동결·검증한 이전 파일 {len(reg['protected_prior_sha256'])}개. 실제 BF16/FP32 훅 검증이 통과하기 전에 과학 출력은 실행하지 않습니다. 뉴런은 MLP.down_proj 입력 좌표이며 이번 단계는 내부 회로 인과 발견이 아닌 행동·입력 실험입니다.", '',
        '모든 효과는 세계 단위로 짝지어 전체 분모의 미판독 최악 경계와 탐색적 bootstrap outer95 구간을 보고합니다. 8–12세계의 작은 표본은 모집단 동등성·정확한95% coverage를 보장하지 않습니다. 수리 성공·가설 결과·논문 가치는 별도 판정입니다. P1/P2와 봉인 확인은 미평가입니다.', '',
        'CPU 합성·oracle 테스트17개 통과는 실제12B 행동 결과가 아닙니다. 원시 전체 생성·token 입력·가중치는 로컬에 보관하고 집계·코드·해시만 공개합니다.', '',
        '[등록 JSON](../logs/summary/round10b_registration.json) · [예산](../configs/round10b_repair.yaml) · [실제 공동심사](../logs/summary/round10b_peer_review.json) · [원래 실패](../logs/summary/round10_attempt1_abort.json)', '']
    (ROOT/'reports/ROUND10B_REPAIR_PREREG_KO.md').write_text('\n'.join(rows))


if __name__ == '__main__':
    main()
