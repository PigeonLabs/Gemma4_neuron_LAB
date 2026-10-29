"""Render the permitted unsuccessful-search outcome from inspected evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [('question', '명제'), ('novelty', '선행과 신규성'),
          ('prior_evidence', '이전 실측의 동기'), ('design', '제안된 설계'),
          ('controls_metrics', '대조·지표'), ('outcome_interpretation', '해석'),
          ('confounds', '교란·실패'), ('cost', '예산·난이도'),
          ('paper_contribution', '논문 기여')]


def read(path):
    return json.loads((ROOT / path).read_text())


def render():
    s = read('logs/summary/round8_selection_results.json')
    p = read(s['peer_review_path'])
    assert hashlib.sha256((ROOT / s['peer_review_path']).read_bytes()).hexdigest() == s['peer_review_sha256']
    assert p['status'] == 'ACTUAL_INDEPENDENT_CROSS_AND_FINAL_FACTUAL_REVIEWS_COMPLETED'
    assert s['paper_Go_count'] == s['new_Gemma_model_forwards'] == 0
    assert len(s['items']) == 5 and all(x['decision'] == 'No-Go' for x in s['items'])
    measured = read('logs/summary/round7_execution_results.json')
    r6 = read('logs/summary/round6_execution_results.json')
    oracle = read('logs/summary/round8_s2_oracle_audit.json')
    reuse = read('logs/summary/round8_reuse_identifiability_audit.json')
    assert measured['total_actual_forwards'] == s['R7_forwards_preserved']
    assert r6['total_actual_forwards'] == s['R6_forwards_preserved']
    assert oracle['status'] == reuse['status'] == 'PASS'
    assert oracle['new_model_forwards'] == reuse['new_model_forwards'] == 0
    lines = [
        '# 자율 논문 주제 탐색 결과와 종료 근거', '',
        '**논문화할 가치가 검증된 새 주제를 찾지 못했다.** 실제 실험·반례·원전 방법과 두 채팅의 독립 심사·교차검토를 근거로, 현재 자율 탐색을 사용자가 허용한 실패 종료 조건에서 마친다. 이는 과학적 발견 성공이나 가능한 모든 연구질문의 소진을 뜻하지 않는다.', '',
        '사용자 목표 §4E의 종료 규칙: “' + s['stop_rule_quote'] + '” 현재 사유는 자원 부족이나 정해진 라운드 수가 아니라 **반복된 식별 문제와, 후속 후보에서 신규성·구별 가능한 인과 대비를 함께 확보하지 못한 점**이다.', '',
        '## 실제 수행과 남은 한계', '',
        f"요청한 R6 세 질문과 별도 A2는 실제 **{r6['total_actual_forwards']:,}회** forward로 준비도·선행 자격까지 평가했다. 이어 R7은 독립 질문 다섯 개에 **{measured['total_actual_forwards']:,}회**(science {measured['scientific_forwards']:,} + technical {measured['technical_forwards']:,})를 수행했다. 이번 후속 선정 감사에서는 Gemma 모델을 추가 실행하지 않았다. R8은 실제 모델 실험 라운드가 아니다.", '',
        '| R7 질문 | 등록 P0 분류 | 해석 경계 |', '|---|---|---|',
        '| A · 논리 영향 범위 | Inconclusive | 모순된 R1 anchor 48개는 우리 설계 오류. main 미측정. |',
        '| B · 질의 전 carrier | Negative | 특정 봉인·47층 궤적 교환의 음성. 형식 붕괴·원문 재접근 때문에 자연 기억 부재를 주장할 수 없음. |',
        '| C · 취소 이력 | Negative | 합성 특이 증폭 미지지. 옛값 logit 흔적과 실제 답 변화는 구분. |',
        '| D · 선행 순서 | Inconclusive | inverse 준비도 중단. 본 priming 미측정. |',
        '| E · 교란 계열 예측 | Negative | 독립 보정으로 동결한 특정 예측기의 제한적 음성. 모든 예측 가능성의 부재가 아님. |', '',
        'R7 원시 4,108행·등록 코드·기존 91개 보호 파일을 CPU로 재검산했다. B의 query-blind cache 48쌍과 E의 target 이전 예측 80개도 확인했다. 실제 모델은 한 backbone이며 독립 GPU 반복은 0회다. P1/P2는 NOT_EVALUATED다. [실측·환경·그림](ROUND7_EXECUTION_RESULTS_KO.md) · [설계·분할 한계](ROUND7_INTERPRETATION_AUDIT_KO.md)', '',
        'R2-E의 제한된 조건부 양성 예측 관계도 지우지 않았다. 사후 dose 매칭·string 과제에 집중된 이득과 선행 중복은 남으며, 새 R7-E 음성이 R2를 소급 무효화하지 않는다. R3 네 후보의 수치 자격 실패, R5 적격 차단쌍 0개, R6 준비도 중단 역시 그대로 보존한다.', '',
        '## 실제 공동 연구에서 바뀐 판단', '',
        f"[GPT · {p['GPT']['chat_title']}]({p['GPT']['url']})의 독립 검토 {p['GPT']['independent_duration_ui']}, 교차검토 {p['GPT']['cross_duration_ui']}, 마지막 사실 검산 {p['GPT']['final_factual_duration_ui']} 답변을 실제 수집했다. Pro 웹 원전 검토이며 별도 Deep Research 보고서는 아니다. [Claude · {p['Claude']['chat_title']}]({p['Claude']['url']})의 독립·교차·마지막 수정 답변도 실제 수집했다. Claude는 여러 원전을 초록 수준으로만 확인했거나 미독이라고 명시했고, Codex가 본문과 대조했다. 어느 심사자도 Gemma를 독립 재현하지 않았다.", '',
        '양측은 먼저 같은 핵심 증거를 독립 검토했고, 두 답변 완료 후 서로의 주장과 Codex 반례를 같은 패킷으로 교차심사했다. S1/S2는 양측 모두 현재 형태로 No-Go였다. GPT가 새로 남긴 계산 공유 Go는 교차검토에서 철회됐다. Claude는 자신의 취소 이력 경로 분해를 철회했지만, 입력을 고정하고 함수표만 바꾸는 계산 공유 수정안을 잠시 남겼다. 이 의견 차이는 다음 CPU 반례로 검토했다.', '',
        '정확한 NeuroTaint 대조가 S1과 동일하다는 주장, ATT가 base LLM의 정책 판정을 검증해 차별점을 선점했다는 주장은 원문 근거가 부족해 철회됐다. 이것이 곧 S1의 신규성을 확보하지는 않는다. 동일 값의 행동 효과가 있어도 정책 오해·값 비교 휴리스틱·태그 결합 실패를 가르는 예측이 없다는 것이 현재 No-Go의 근거다.', '',
        '## CPU 검산 두 가지 — 실제 모델 결과와 별개', '',
        '### 검증자 선택의 정답 기준', '',
        '32개 항목의 이진 truth T, 기존 판정 A, 추가 판정 V에 대해 두 분포의 모든 쌍별 결합표(T,A)/(T,V)/(A,V), 각 정확도 .75와 공동오류율 .125가 같도록 만들었다. 전체 결합 패턴은 다르다.', '',
        '| (T,A,V) | 분포 1 개수 | 분포 2 개수 |', '|---|---:|---:|',
    ]
    a, b = oracle['rows'][0], oracle['rows'][2]
    for left, right in zip(a['counts'], b['counts']):
        assert left['values'] == right['values']
        lines.append('| ' + ''.join(map(str, left['values'])) + f" | {left['n']} | {right['n']} |")
    lines += [
        '', f"추가 판정을 최적으로 사용한 정답률은 각각 **{a['both_verdicts_Bayes_accuracy']:.1%}와 {b['both_verdicts_Bayes_accuracy']:.1%}**다. 따라서 쌍별 오류 중첩만으로 S2의 oracle을 정할 수 없다. 이는 기초적인 고차 확률 반례이며 새 정리·실제 LLM 현상이나 CAVEAT/SPLIT의 선언된 가정에 대한 반증이 아니다. [정확한 CPU 열거](../logs/summary/round8_s2_oracle_audit.json)", '',
        '### 계산 재사용의 관측 동등성', '',
        f"결과 f(x′)를 저장해 읽는 모형과 원입력 x′를 저장하고 질의마다 f를 재계산하는 모형을 구성했다. reverse/not/parity/first-pair의 **{reuse['total_observables_compared']:,}개 합성 관측**에서 답값이 모두 같았다. 둘 다 명시 공유 witness와 duplicate−coincidental 전이 패턴을 만들 수 있다. 전단사 reverse/not의 같은 결과·다른 입력 donor는 존재하지 않는다.", '',
        f"Claude의 같은 입력·다른 표 수정안도 검사했다. 결과 D[x]를 저장하는 모형과 표 D를 저장해 매 질의 D(x)를 계산하는 모형은 **{reuse['same_input_different_table_repair']['observables_compared']:,}개 관측**에서 같았다. 입력 probe x도 그대로여서 입력 보존만으로 표·연산자 복사를 배제할 수 없다. 다른 입력 f(x₂)를 묻는 추가 관측은 일부 모형을 구분할 수 있지만 원래 열거 범위 밖이며, x 특이 행 정보와 계산 결과 재사용까지 일반적으로 분리한 것은 아니다.", '',
        'GPT는 마지막 검산에서 답값 비교를 logit 비교로 과장하지 말라고 지적했다. 이를 수용해 원래 답값-only 감사본을 로컬에 보존하고, 두 모형에 동일한 값→4코드 logit/softcap readout을 명시한 가능 구성을 추가했다. 그 구성에서는 logit 벡터도 같고 positive margin 전이까지 양쪽에서 발생한다. **이 head는 합성 구성이지 실제 Gemma head를 측정·복제한 것이 아니다.** 이 반례는 제안의 양성 패턴이 계산 결과 재사용을 유일하게 식별하지 못함을 보인다. 모든 다른 관측으로도 영원히 식별할 수 없다는 주장은 아니다.', '',
        'Claude는 이 범위를 확인한 뒤 남은 조건부 수정안을 철회했다. 같은 값 donor는 identity donor와 다르며, 같은 값 donor에서 변화가 생겼다고 바로 기술 실패라 할 수 없다. 특정 위치 접근을 막아도 정답이 유지된다고 재계산을 증명한 것도 아니다. 다른 중복 상태가 남을 수 있다. [두 기제·수정안·readout의 CPU 감사](../logs/summary/round8_reuse_identifiability_audit.json)', '',
        '## 마지막 다섯 제안의 판정', '',
        '**실행 가능한 다섯 새 연구를 선정했다고 하지 않는다.** 다음은 검토한 다섯 제안과 거절 사유다. 모두 미등록·미실행이며 Negative 효과 판정을 받은 새 모델 실험이 아니다.', '',
        '| 검토 순서 | 제안 | 현재 투자 판단 |', '|---:|---|---|',
    ]
    for item in s['items']:
        lines.append(f"| {item['rank']} | {item['title']} | {item['decision']} |")
    for item in s['items']:
        lines += ['', f"### {item['id']}. {item['title']}", '']
        for key, label in FIELDS:
            lines.append(f'- **{label}:** {item[key]}')
    lines += [
        '', '## 종료의 의미와 재개 조건', '',
        '현재 결과는 검증 가능한 의미 뉴런·자연 계산 병목·독창적인 기제 논문의 발견으로 이어지지 않았다. 단일 backbone에서 나온 제한적 양성·음성·미식별 결과와 설계 실패는 재현 가능한 연구 기록으로 남긴다. 이 기록을 논문급 기여로 자동 승격하지 않는다.', '',
        '새로운 결과가 나올 때까지 기존 층·dose·표현을 반복 탐색하거나, 후보 수를 맞추기 위해 약한 질문을 실행하지 않는다. 현재 데이터와 선행에서 정당화한 후보가 모두 탈락한 지점에서 사용자의 명시적 종료 규칙을 적용했다. 모든 연구 분야나 미래 질문이 소진됐다는 주장이 아니며 GPU/RAM/디스크 부족을 사유로 삼지 않았다.', '',
        '재개에는 새로 동기화된 명제, 경쟁 기제들이 서로 다른 결과를 예측하는 사전 고정 대비, 선행 방법과 구별되는 기여, 해석 가능한 양성 도구 대조가 필요하다. 이후 새 연구에서는 실제 보이는 세계와 token 입력의 분할 중복을 첫 출력 전에 막아야 한다. 현재 예약 분할을 조용히 교체해 확증으로 사용하지 않는다.', '',
        '[누적 후보 순위](PAPER_CANDIDATE_RANKING_KO.md) · [요구사항·종료 조건 감사](../logs/summary/autonomous_discovery_completion_audit.json) · [실제 심사 원문 해시·의견 차이](../logs/summary/round8_peer_review.json) · [후속 원전 확인 범위](../logs/summary/round8_literature_audit.json) · [계산 가능한 종료 요약](../logs/summary/round8_selection_results.json)', '',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'reports/AUTONOMOUS_DISCOVERY_OUTCOME_KO.md'
    content = render()
    if args.check:
        assert path.exists() and path.read_text() == content, 'Stale discovery outcome report'
    else:
        path.write_text(content)
    print(json.dumps({'status': 'PASS', 'new_model_forwards': 0,
                      'scientific_topic_found': False, 'permitted_stop_branch': True}))
