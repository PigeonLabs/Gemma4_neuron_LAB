"""Render the adjudicated R7 review and cumulative ranking from public JSON.

This does not generate peer opinions or modify any registered scientific result.
Raw peer responses and screenshots remain private; their hashes are receipts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    ('question', '연구질문·가설'), ('novelty', '신규성'),
    ('prior_evidence', '이전 실측과 연결'), ('design', '구체 설계'),
    ('controls_metrics', '대조·통제·지표'),
    ('outcome_interpretation', '결과별 해석'), ('confounds', '실패·교란'),
    ('cost', '예산·난이도'), ('paper_contribution', '논문 기여'),
]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def render():
    peer = read('logs/summary/round7_post_results_peer_review.json')
    pool = read('logs/summary/round7_next_candidate_screening.json')
    ranking = read('logs/summary/paper_candidate_ranking.json')
    measured = read('logs/summary/round7_execution_results.json')
    lit = {x['id']: x for x in read('logs/summary/round7_literature_audit.json')['entries']}
    assert peer['status'] == 'ACTUAL_INDEPENDENT_AND_ADVERSARIAL_CROSS_REVIEWS_COMPLETED'
    assert len(pool['items']) == 5 and pool['Go_count'] == 0
    assert measured['total_actual_forwards'] == 4278
    assert measured['P1'] == measured['P2'] == 'NOT_EVALUATED'
    lines = [
        '# R7 사후 공동 심사와 다음 연구 후보', '',
        '다섯 P0를 실제 실행하고 같은 핵심 증거를 GPT Pro와 Claude Opus 5.5 Extra에 전달해 독립 심사 및 적대적 교차검토를 완료했다. **즉시 P1/P2 승격과 논문 Go는 모두 0개다.** 실행 성공, 등록된 과학 판정, 논문 투자 판단을 구분한다.', '',
        '실제 모델 forward는 **4,278회**다(science 4,108 + technical 170). A/D는 준비도 중단으로 핵심 명제가 미측정이고, B/C/E는 각 고정 프로토콜의 등록 Negative다. P1/P2와 R8은 NOT_EVALUATED다. [원시 결과·그림·환경](ROUND7_EXECUTION_RESULTS_KO.md) · [해석·분할 감사](ROUND7_INTERPRETATION_AUDIT_KO.md)', '',
        '## 실제 협업과 접근 범위', '',
        '| 검토자 | 독립 검토 | 교차검토·한계 |', '|---|---|---|',
        f"| [GPT · {peer['GPT']['chat_title']}]({peer['GPT']['url']}) | 실제 완료, UI {peer['GPT']['independent_duration_ui']} | 실제 완료, {peer['GPT']['cross_duration_ui']}. Pro 웹 검토이며 별도 Deep Research 보고서 아님. 원시 로그 독립 재계산·GPU 재현 아님. |",
        f"| [Claude · {peer['Claude']['chat_title']}]({peer['Claude']['url']}) | 실제 완료, Opus 5.5 Extra | 실제 완료. 이번 응답에서 원전을 다시 읽지 않았다고 명시. 의견 속 인용은 Codex가 별도 원전 확인. |", '',
        '다른 심사자의 의견을 처음부터 공유하지 않았다. 두 독립 답변 완료 후 양측 평가와 Codex의 사실 검산을 같은 교차검토 패킷으로 보냈다. GPT의 첫 긴 첨부 요청은 서버 오류가 났고, 전체 핵심 증거를 입력란에 펼쳐 재전송한 실제 완료 답변만 심사로 집계했다. 원문·전송문·완료 화면은 로컬 runs에 보존하며 공개 JSON에는 SHA256과 접근 범위를 기록한다.', '',
        '## 수용한 비판과 바로잡은 사실', '',
        '- **A:** 모순된 R1 anchor 48개는 우리의 설계 오류다. 규칙 일반화나 모델 능력 부재의 증거로 쓰지 않는다. 원본·gate를 바꾸거나 수정 반복으로 구제하지 않았다.',
        '- **B:** full-vocabulary 형식 붕괴와 sealed explicit-answer 양성 대조의 부재는 실제 한계다. 반면 Claude의 “코드표도 차단됐다”는 설명은 249개 고유 encoding의 토큰 위치와 맞지 않아 철회됐다. 47층 궤적 교환과 반복 원문 접근은 단일 compact state의 기억을 검정하지 않는다.',
        '- **C:** “정확도 미측정”은 틀려 철회됐다. 1,024행의 정답과 전체 어휘 top1을 기록했다. 512개 old-value 교환 쌍 중 top1은 7쌍, 정답 여부는 6쌍에서 바뀌었고 현재 언급된 obsolete 답 선택은 0행이다. 따라서 “행동 변화가 전혀 없다”도 지나친 표현이다. 무관 이력 값과 FINAL 값의 우연 일치가 편향을 반드시 0 쪽으로 만든다는 주장 역시 근거가 없어 철회됐다.',
        '- **D:** forward와 inverse 답이 같은 14/32 질의는 연산자 사용을 구별하지 못했다. 원 oracle 모순은 아니지만 준비도 판독을 약하게 만든다. 본 순서 효과는 미측정이다.',
        '- **E:** 6개 보정 세계로 고정한 특정 알고리즘의 prospective Negative는 유효하다. 보정 실패가 원인이라고 확정하거나 모든 예측 가능성이 없다고 말하지 않는다. target만 잔차화한 상관은 partial Spearman이 아니며 상관 부호로 두 설명을 이분하지 않는다.', '',
        'B는 등록 12회 latent draw에 11개 고유 main view가 있었다. 복원추출의 중복 자체가 곧 통계 무효는 아니다. 원래 draw 가중 추정과 11개 view 동일가중 사후 민감도는 다른 추정 대상이므로 둘 다 보존한다. 미사용 P2의 B/D/E 입력 중복도 발견했지만 P2 출력은 보지 않았고 독립 확증으로 주장하지 않았다. 실제 E calibration–P0 prompt 중복은 0이다.', '',
        '## 의견 차이와 연구 책임자의 판단', '',
        'GPT는 C의 좁은 margin 관찰과 E의 고정 알고리즘 음성을 논문 가능성 Hold로 남겼다. Claude는 새 이론이나 독립 행동 명제가 없으므로 No-Go를 주장했다. Codex는 **측정 증거를 보존하되 현재 주장의 독립 논문·추가 실행은 No-Go**로 판단한다. 이는 효과가 0이라는 결론이 아니다. 다른 이론에서 새 예측이 생기면 별도 연구로 재검토할 수 있다.', '',
        'GPT는 최초 후속 1순위의 조건부 Go를 LLMCAPLENS·CoRe를 확인한 후 Hold로 낮췄다. Claude는 내부 유사도 예측·철회 문구·직렬화 역조회 세 제안을 모두 철회했다. Codex는 명시 trace 승인 과제가 자동 채점 가능하다는 점은 인정하나, 같은 답만으로 실제 내부 사용 출처를 알아냈다고 주장하지 않는다. 기존 binding 평가 이상의 예측이 없는 namespace는 현재 No-Go다. 모델 간 합의는 신규성의 증거로 쓰지 않았다.', '',
        '## 다음 다섯 제안의 신규성 심사 순위', '',
        '**이 표는 다섯 실행 가능 연구의 선정표가 아니다.** Go 0, Hold 2, No-Go 3이다. 모두 미등록·미실행이며 아래 호출 수는 심사 중 제안된 예산 초안이다. 이미 기각한 연구의 층·dose·prompt 탐색을 새 질문으로 세지 않는다.', '',
        '| 순위 | 제안 | 판단 |', '|---:|---|---|',
    ]
    for x in pool['items']:
        lines.append(f"| {x['rank']} | {x['title']} | {x['decision']} |")
    for x in pool['items']:
        lines += ['', f"### {x['rank']}. {x['title']} — {x['decision']}", '']
        for k, label in FIELDS:
            lines.append(f'- **{label}:** {x[k]}')
        lines += ['', '재검토에 필요한 근거: ' + x['missing_decisive_evidence'], '']
        lines.append('원전: ' + ' · '.join(f"[{lit[k]['title']}]({lit[k]['url']})" for k in x['sources']))
    lines += [
        '', '## 현재 가장 강한 명제와 다음 실행 조건', '',
        '이 고정 모델·동일 BF16 값의 FP32 upcast·표본·프로토콜에서 B/C/E의 등록 양성 기준은 충족되지 않았다. C는 동일한 FINAL 표에서도 취소 이력에 따라 old-value logit 대비가 달라졌지만 등록한 합성 특이 증폭은 지지되지 않았다. E는 독립 보정으로 고정한 예측기의 최소 이득 .02를 새 P0에서 달성하지 못했다. A/D의 핵심 가설은 검정되지 않았다. 자연 기억 부재, 모든 공통 취약도 부재, 의미 뉴런의 발견이나 다른 모델 일반화는 주장하지 않는다.', '',
        '기존 R7은 닫고 새 실행 예산을 배정하지 않는다. S1/S2는 정확한 신규성·메커니즘 예측을 확인하는 단계다. 다섯 실행 가능한 새 연구는 아직 선정하지 못했다. 이는 가능한 모든 연구질문이 없다는 결론도, 자율 연구 목표 달성도 아니다. 다음 독립 후보 탐색은 열어 두며, 같은 실패 계열의 반복 실행은 하지 않는다.', '',
        '새 연구는 첫 모델 출력 전에 deterministic anchor의 일관성, 실제 보이는 세계 signature와 authoritative token 입력의 분할 중복을 검사해야 한다. 추가된 입력 계약은 향후 연구용 CPU 인프라다. 실제 R7을 소급 수정하거나 새 실험으로 계산하지 않는다. 과학 실행 당시 CPU 시험 153개와 이후 추가한 4개를 분리했으며 현재 합계 157개가 통과했다. 독립 GPU 반복은 0회다.', '',
        '[실제 심사 원문 해시·정정·반론](../logs/summary/round7_post_results_peer_review.json) · [27편 원전 접근·본문 확인 범위](../logs/summary/round7_literature_audit.json) · [후보 9항목 JSON](../logs/summary/round7_next_candidate_screening.json) · [누적 논문 순위](PAPER_CANDIDATE_RANKING_KO.md)', '',
    ]
    ranking_lines = [
        '# 누적 논문 후보 순위', '',
        '기록 기준 2026-10-30 KST. **논문 Go 0개.** Hold는 신규성·식별력이 미해결인 옵션이며 실행 선정이 아니다. No-Go는 현재 주장·설계의 투자 판단이며 현상 부재의 증명이 아니다. 과거 실측과 중단 판정을 보존한다.', '',
        '| 순위 | 질문 | 판단 | 확보한 증거 |', '|---:|---|---|---|',
    ]
    for x in ranking['entries']:
        assert digest(x['source']) == x['source_sha256'], f"Stale ranking source: {x['id']}"
        ranking_lines.append(f"| {x['rank']} | [{x['id']} · {x['question']}](../{x['source']}) | {x['decision']} | {x['evidence']} |")
    for x in ranking['entries']:
        ranking_lines += ['', f"## {x['rank']}. {x['id']} · {x['question']}", '',
                          f"{x['decision']} — {x['novelty']}", '',
                          '부족한 결정적 근거: ' + x['missing_decisive_evidence'], '']
    ranking_lines += ['[실측 이후 실제 공동 심사](ROUND7_REVIEW_AND_NEXT_KO.md) · [계산 가능한 순위·근거 해시](../logs/summary/paper_candidate_ranking.json)', '']
    return {'reports/ROUND7_REVIEW_AND_NEXT_KO.md': '\n'.join(lines),
            'reports/PAPER_CANDIDATE_RANKING_KO.md': '\n'.join(ranking_lines)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, content in render().items():
        target = ROOT / path
        if args.check:
            assert target.exists() and target.read_text() == content, f'Stale report: {path}'
        else:
            target.write_text(content)
    print(json.dumps({'status': 'PASS', 'mode': 'check' if args.check else 'render',
                      'peer_review_complete': True, 'new_GPU_calls': 0}))
