#!/usr/bin/env python3
"""Render the curated post-result review and current ranking from public JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    peer = read('logs/summary/round9_result_peer_review.json')
    decision = read('logs/summary/round9_final_decision.json')
    measured = read('logs/summary/round9_execution_results.json')
    ranking = read('logs/summary/paper_candidate_ranking.json')
    assert peer['status'] == 'BOTH_INDEPENDENT_AND_ADVERSARIAL_CROSS_COMPLETE'
    assert decision['scientific_discovery_achieved'] is False
    assert measured['total_actual_forwards'] == decision['actual_R9_forwards'] == 1790
    assert measured['scientific_classification_counts'] == {'Inconclusive': 4, 'Negative': 1}
    assert decision['paper_Go_count'] == decision['qualified_new_GPU_studies'] == 0

    lines = [
        '# R9 실측 후 공동 심사와 현재 탐색의 종료', '',
        '**논문 가치가 검증된 새 주제를 찾지 못했다.** 새 다섯 질문의 실제 P0와 사후 독립·교차 검토를 마쳤지만, 현재 추가 GPU 연구를 정당화할 독립적인 예측과 기여를 확보하지 못했다. 사용자 목표 §4E·§8의 실패 탐색 종료 분기를 적용한다. 발견 성공이나 가능한 모든 연구질문의 소진을 뜻하지 않는다.', '',
        '이 판단은 R8의 과거 종료를 재사용한 것이 아니다. 재개 후 새로 선정·사전등록한 다섯 P0에서 실제1790회(과학1756＋공유 기술34) forward를 수행했다. 남은 예산을 모두 쓴 결과도 아니다. 네 질문은 이 계측으로 평가하지 못했고, D만 등록한 좁은 공동 패턴이 나타나지 않았다. P1/P2는 모두 **NOT_EVALUATED**다.', '',
        '[실측·그림·실행 한계](ROUND9_EXECUTION_RESULTS_KO.md) · [최종 판단 JSON](../logs/summary/round9_final_decision.json) · [실제 협업·해시·접근 한계](../logs/summary/round9_result_peer_review.json) · [누적 후보 순위](PAPER_CANDIDATE_RANKING_KO.md)', '',
        '## 실제로 수행한 범위', '',
        '| 질문 | 과학 forward | 결정적인 관측 | 등록 판정 | 현재 P1 |',
        '|---|---:|---|---|---|',
        '| A 정보 분할 | 896 | anchor128/128, main 코드99/768 | Inconclusive / NOT_IDENTIFIABLE | No-Go |',
        '| B 보상 재배분 | 96 | path-sum anchor47/96, main 미측정 | Inconclusive | No-Go |',
        '| C 수집 절차 | 96 | anchor22/96, 유효 코드48/96, main 미측정 | Inconclusive | No-Go |',
        '| D 고차 분포 | 540 | table anchor60/64, 공동 패턴0/8 family | Negative, 유한한 등록 패턴만 | No-Go |',
        '| E 자원 경쟁 | 128 | edge63/64, union의 유효11/64는 모두 정답, main 미측정 | Inconclusive | No-Go |', '',
        '1790회는 서로 독립적인1790표본이 아니다. code·query·prefix·reference는 세계 안 반복이다. D의540행은421개 고유 입력이며, 주 rule tree32개·고유 full-table tree16개·고유 uniform reference tree3개와 anchor64개로 구성된다. 독립 단위는8 q-family다. 같은 입력의 동일 readout은 결정성 점검이다.', '',
        '## 두 독립 심사와 적대적 교차 검토', '',
        '같은 핵심 실측 패킷을 두 기존 채팅에 실제 전달했고, 양쪽 독립 답변을 모두 보존한 뒤 새 의견을 처음 교환했다. GPT 비판을 반영한 guard도 Claude 독립 완료 전에는 공개하지 않았다.', '',
        '| 단계 | GPT 채팅 | Claude 채팅 |', '|---|---|---|',
        '| 독립 검토 | 실제 Pro 웹 검토10m02s | 실제 Opus5.5 Extra Research5m29s, UI276 sources |',
        '| 교차 검토 | 실제 Pro 완료6m22s | 실제 Opus5.5 Extra 일반 교차 답변 완료 |',
        '| 확인 범위 | 공개 보고서·실행기·생성기·분석기·등록YAML, 로컬 raw 재현 아님 | 결과 보고서, 등록·YAML·코드 직접 접근 실패; 외부 원전 대부분 발췌 |', '',
        'GPT의 별도 Deep Research는 앞 요청에서 오류였으므로 완료 Research 보고서로 세지 않는다. Claude의 UI source 수는 본문을 읽은 원전 수가 아니다. 원문·화면·첨부는 로컬 runs에 보존하고 공개에는 해시와 요약만 둔다. [GPT 채팅](https://chatgpt.com/c/6ae2d6e2-af1c-83ee-b796-7145db91d8b2) · [Claude 채팅](https://claude.ai/chat/aa908934-d215-41b7-834b-53f83f60a5e7).', '',
        '가장 큰 이견은 D였다. Claude는 처음에 저차 정상 조건이0개이므로 D 전체를 준비도 실패로 바꾸자고 했다. Codex는 실행 전 YAML의 P0_negative가 저차 실패를 명시적으로 포함한다는 점과, 실제 분석기가 저차 정상·고차 오류·유효 질량의 conjunction을 검사한다는 점을 제시했다. 실제 readiness인 table anchor와 전체 어휘 형식은 통과했다. 교차 답변에서 Claude는 재분류 의견을 철회했고 GPT는 좁은 등록 판정 보존에 동의했다.', '',
        '유지할 두 문장은 다음과 같다. **등록된 공동 패턴은0/8로 Negative다. 저차가 정상인 조건에서의 고차 오류율은 eligible0이므로 추론 불가다.** 둘을 합쳐 보편적인 고차 결함의 부재나 조건부 가설의 반증이라고 하지 않는다. 심사자 동의는 새 모델 관측이 아니다.', '',
        'Codex가 전부 수용하지 않은 의견도 있다. C의22/48은 유효 형식에 조건을 건 기술통계로 계산할 수 있다. Claude가 이를 단순한 미확인 추정으로 철회한 것은 과하다. 다만 그 선택된 부분집합으로 전체 능력을 추론할 수 없다는 제한은 맞다. C 수리 순위를 높였던 의견도 위치 shortcut 발견 뒤 약해졌으며, 수리 우선순위와 논문 우선순위는 별개다.', '',
        '## 실행 전 검산이 놓친 문제와 반영한 조치', '',
        '독립 CPU oracle8개는 수학적 정답·분할·카운터를 검증했지만 지시의 의미 일관성과 정답 선택지 위치를 충분히 감사하지 못했다. B의 route-choice cue와 YES/NO anchor 질문은 충돌할 수 있고, C의96개 anchor는 정답 분수가 항상 첫 번째였다. 이 결함은 실측 후 발견했다. C는 첫 선택지 지름길로 통과할 수 있으므로 통과했더라도 확률 계산의 충분한 자격 검사가 아니었다. 어느 결함도 이번 실패의 실제 인과적 원인으로 입증하지 않았다.', '',
        'A의 To는 첫 토큰 관측이다. 설명·거부·최종 오답의 기제로 해석하지 않는다. 실제 template와11900개 저장 입력 감사는 enable_thinking=False가 적용되고 빈 thought를 닫았음을 확인했다. 이 사실도 To의 원인을 밝히지 않는다. B47/96을 무작위 추측과 동등하다고 주장하지 않으며, E11/64를 union 능력17%로 부르지 않는다.', '',
        '원래 소스·입력·gate·요약을 바꾸지 않았다. 별도 [해석 guard](../logs/summary/round9_interpretation_guard.json)에서 A/B/C/E의 주 추정값과 CI를 null로 두고, 무효 응답을 NO로 코딩한 숫자를 행동 증거로 재사용하지 못하게 했다. 원본 R9 측정·실행 전 소스20개와 기존139개 보호 해시를 검증한다. CPU replay는 기록 검산이며 전체 어휘 logits를 두 번째 모델 실행으로 재현한 것은 아니다.', '',
        '첫 토큰 점수와 생성 후 판정의 차이는 기존 연구도 직접 검사했다. 이 문헌은 계측 한계를 뒷받침하지만 이번 Gemma의 실패 원인을 설명하지는 않는다. [The First Token Is Not the Verdict, §1–3](https://arxiv.org/html/2610.00054v1).', '',
        '## 후속 제안 선별 순서와 현재 투자 판단', '',
        '다음 다섯 개는 실측 후 검토한 대안이다. **실행할 만한 새 연구 다섯 개를 선정한 목록이 아니다.** 모두 현재 GPU No-Go이며 새 모델 호출0회, 새 P0 미등록이다. 순서는 검토 순서일 뿐 논문 추천 순위가 아니다. 각 안의 질문·신규성·연결·설계·통제/지표·결과별 해석·교란·비용·기여의9항목은 최종 판단 JSON에 보존했다.', '',
        '| 검토 순서 | 제안 | 현재 부족한 근거 |', '|---|---|---|',
        '| 1 | 사본 수와 독립 출처 수의 구별 | 근접 문헌이 이 대비와 결정 임계까지 검사했다. 추가 구별 예측 없음 |',
        '| 2 | 분포 기술과 실제 표집의 불일치 | D의 재명명과 다른 독립 설명·readout 자격·정확한 선행 차이 없음 |',
        '| 3 | explaining-away의 인과적 구별 | 교과서 posterior 계산을 넘는 기제 예측 없음. 인용된 noisy-OR 본문 미확인 |',
        '| 4 | 필수 전제 누락에서의 보류 | 일반 answerability/abstention 점수 외 새 예측 없음. 관련 상세 방법 미확인 |',
        '| 5 | martingale 갱신 일관성 | 기존 진단 회귀를 넘는 독립적 원인 예측 없음 |', '',
        '사본/출처의 단순 대비는 controlled log·pointer·padding·독립 센서 대조와 함께 이미 검사됐다. [Copies or Sources?, §2–4](https://arxiv.org/html/2610.06192v1). 일반 martingale drift 회귀도 기존 방법이며, zero slope만으로 완전한 Bayes 추론이 증명되지는 않는다. [Martingale Score, §4–6](https://arxiv.org/html/2512.02914).', '',
        '나머지 인용의 본문 미확인은 UNVERIFIED로 남긴다. 초록·검색 결과로 정확한 중복을 확정하지 않는다. SE-RRM의 학습된 대칭 강제 구조도 frozen Gemma의 모든 치환 실험을 기각할 근거가 아니다. Working Memory Readout은 Codex가 §2–5 본문을 읽었고, miss와 lure false alarm의 attention 패턴이 다르므로 합쳐 일반화하지 않았다. [원전별 독서 범위와 한계](../logs/summary/round9_literature_audit.json).', '',
        '## 종료와 재개 조건', '',
        '종료 이유는 현재 확보한 질문·계측에서 반복된 식별 실패와, 사후 대안에서 새 구별 예측·학술 기여를 갖춘 실행안을 확보하지 못한 점이다. 원래 목표 §4E는 유효 질문 부재 또는 반복 식별 불가에서 사유를 기록하고 중단하도록 허용한다. “다섯 가설이 거짓”이나 “다른 모델·생성 후 파싱·다른 설계로도 영원히 불가능”이라는 주장은 하지 않는다.', '',
        'C의 선택지 위치, B의 지시 일관성, A의 역할/질문 층위, E의 답변 형식은 향후 측정 수리 과제다. 기존 결과를 수리해 다시 확증한 것으로 만들지 않는다. 재개에는 경쟁 설명을 가르는 새로운 고정 예측, 원전 방법·결과와의 차이, 의미/위치 감사, 사용하지 않은 분할과 새 예산·gates의 사전등록이 필요하다. 현재 승인된 추가 P1/P2나 새 GPU 실험은 없다.', '',
        '[종료 조건·보호 해시·실행 범위 감사](../logs/summary/round9_closure_audit.json). 공개 readback 영수증은 별도로 기록한다.', '',
    ]
    (ROOT / 'reports/ROUND9_REVIEW_AND_NEXT_KO.md').write_text('\n'.join(lines))

    for entry in ranking['entries']:
        if entry['id'].startswith('R9-'):
            entry['source_sha256'] = sha('reports/ROUND9_EXECUTION_RESULTS_KO.md')
    (ROOT / 'logs/summary/paper_candidate_ranking.json').write_text(json.dumps(ranking, ensure_ascii=False, indent=2) + '\n')
    rank_lines = [
        '# 누적 논문 후보와 현재 투자 판단', '',
        f"현재 논문 Go **{ranking['Go_count']}개**, 실행 가능한 새 Hold **{ranking['Hold_count']}개**, 다음 모델 연구 **{ranking['active_next_experiment_count']}개**다. 새 논문 주제 발견에는 실패했고 현재 탐색을 종료한다.", '',
        'R9의1–5는 실행 전의 우선순위를 보존한다. 현재 모두 No-Go이며 재실행 추천이 아니다. 이전 항목의 순위는 역사적 비교용이다. No-Go는 현재 주장·설계에 대한 투자 판단이며 현상의 보편적 부재를 뜻하지 않는다.', '',
        '[실측 후 심사·종료 근거](ROUND9_REVIEW_AND_NEXT_KO.md) · [원본 후보 JSON](../logs/summary/paper_candidate_ranking.json) · [새 대안 선별9항목](../logs/summary/round9_final_decision.json)', '',
        '| 순번 | 후보 | 현재 판단 | 확보한 증거 | 남은 핵심 요건 |',
        '|---:|---|---|---|---|',
    ]
    for entry in ranking['entries']:
        def cell(value):
            return str(value).replace('|', '\\|').replace('\n', ' ')
        rank_lines.append('| ' + ' | '.join(cell(entry.get(key, '')) for key in
            ['rank', 'question', 'decision', 'evidence', 'missing_decisive_evidence']) + ' |')
    rank_lines += ['', '각 후보의 신규성·출처·출처 해시는 JSON에 보존한다. 사후에 과거 양성·음성·미측정 판정을 변경하지 않는다.', '']
    (ROOT / 'reports/PAPER_CANDIDATE_RANKING_KO.md').write_text('\n'.join(rank_lines))
    (ROOT / 'reports/ROUND9_REOPENING_KO.md').write_text('\n'.join([
        '# R9 연구 재개와 현재 종료 상태', '',
        'R8 실패 종료 이후 사용자의 활성 목표로 새 방향을 조사하고 다섯 새 P0를 실제 수행했다. R8 기록과 과거 실측은 보존한다.', '',
        f"현재 상태: **{decision['status']}**. 실제 Gemma forward1790회, 과학1756＋공유 기술34회. 4 Inconclusive / 1 제한된 등록 Negative. 새 논문 발견은 실패, P1/P2 NOT_EVALUATED다.", '',
        '실행 전·실측 후 각각 실제 GPT Pro와 Claude Opus5.5 Extra Research 독립 비판 및 양쪽 교차 답변을 수집했다. GPT의 별도 Deep Research 오류와 Claude의 등록/코드 접근 실패 등 한계를 보존한다. UI source 수를 본문 독서 수로 세지 않는다.', '',
        '현재는 목표 §4E·§8에 따라 이 탐색풀을 종료한다. 자원 고갈이나 모든 가능한 연구질문의 소진을 주장하지 않는다. 사후 형식/anchor 수리를 새 학술 명제로 세지 않는다.', '',
        '[다섯 질문 사전등록](ROUND9_P0_PREREG_KO.md) · [실제 결과](ROUND9_EXECUTION_RESULTS_KO.md) · [실측 후 심사와 최종 판단](ROUND9_REVIEW_AND_NEXT_KO.md) · [현재 상태 JSON](../logs/summary/round9_progress.json)', '',
    ]))
    print(json.dumps({'status': 'RENDERED_FROM_CURATED_JSON', 'reports': 3,
                      'scientific_discovery_achieved': False, 'model_forwards_added': 0}))


if __name__ == '__main__':
    main()
