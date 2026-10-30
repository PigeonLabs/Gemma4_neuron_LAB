#!/usr/bin/env python3
"""Build R10/R10b results and README inputs from actual replayed summaries."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS={'A':'관측 정보와 전략의 구현 가능성','B':'총합 보존 보상 재분배와 경로 선택',
        'C':'자료 수집 절차와 posterior 판단','E':'국소 자원 집합과 전역 배정 가능성'}


def read(n):
    return json.loads((ROOT/'logs/summary'/n).read_text())


def write(n, d):
    (ROOT/'logs/summary'/n).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def percent(v):
    return f'{100*v:.2f}%'


def main():
    agg = read('round10b_aggregate.json'); old = read('round10_attempt1_abort.json')
    result_rows = []
    report = ['# R10/R10b 실제 계측 수리 결과', '',
        'A/B/C/E의 원래 미평가 상태를 과학적 반증으로 해석하지 않고, 실제 두 채팅과 계측을 수리해 새 데이터로 실행했습니다. 원래 R10 실패와 별도 R10b 결과를 분리합니다.', '',
        f"모델 revision `{agg['model_revision']}`. {agg['primary_dtype']}. 이번 수치는 실제12B 모델에서 생성한 응답의 집계이며, CPU 합성 테스트17개나 토큰 replay는 추가 모델 표본으로 세지 않습니다.", '',
        '## 원래 시도와 별도 수리', '',
        f"R10 A 보정은 완결{old['partial_calibration_responses']}응답에서 VALID{old['parse_status']['VALID']}/TOKEN_CAP{old['parse_status']['TOKEN_CAP']}였습니다. 원래 등록에 없던 추가 공학적 중단을 사용했고, 당시 모든 P0는 미실행입니다. 미완결1응답의{old['in_flight_forward_entries_without_completed_response']}forward 진입은 INTERRUPTED 비용이며 TOKEN_CAP 응답으로 세지 않았습니다. 원래 실패와 소스는 보존했습니다.", '',
        'R10b에는 간결 출력계약과 A의 직접 공지 anchor 명료화를 등록했습니다. R9와 R10의 사용·예약 세계/입력을 모두 배제하고,256토큰·retry0·기존 terminal parser·문자/선택지 교차를 유지했습니다. 질문별 futility는 새 출력 전에 등록했습니다. 후속 추가 문구·cap 수리는 하지 않았습니다.', '',
        f"추정 대상: {agg['measurement_scope']} 새 세계와 출력 계약이 함께 바뀌어 R10 대비 개선의 원인을 분리할 수 없습니다. 기초 정답 자격 실패, 최종답 판독 실패, 실제 main의 넓은 구간은 서로 다른 상태입니다.", '',
        '## 실행 범위와 결과', '',
        '| 질문 | 보정 완결 응답 | 보정 terminal 유효 | 보정 한 줄 준수 | 새 P0 응답(기초+main) | P0 실제 forward | 최종 상태 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for s in ['A','B','C','E']:
        d = agg['studies'][s]; cal = d['calibration']; p0 = d['P0']
        measured = p0.get('status') != 'NOT_EVALUATED'
        out = p0.get('outcome', {})
        nmain = p0.get('main_or_sentinel_responses', 0)
        classification = out.get('classification', 'NOT_EVALUATED') if nmain else 'NOT_EVALUATED'
        gate = out.get('gate', cal['outcome']['gate'])
        result_rows.append({'study':s,'P0_responses':p0.get('responses',0),'P0_main_responses':nmain,
            'P0_forwards':p0.get('actual_forwards',0),'gate':gate,'classification':classification})
        report += [f"| {s} | {cal['responses']} | {cal['parse_status'].get('VALID',0)}/{cal['responses']} | {cal['single_line_contract_compliant']}/{cal['responses']} | {p0.get('responses',0)} ({p0.get('anchor_responses',0)}+{nmain}) | {p0.get('actual_forwards',0)} | {classification}; {gate} |"]
    report += ['', '보정 중단은 최종 예정 분모에서 기준 달성이 불가능하다는 등록 규칙에 따릅니다. 이 상한은 고정 문항 묶음의 산술적 판정이며 모집단 판독률이95% 미만이라는 통계 검정이 아닙니다. 중단 시까지의 관측 분모와 최종 예정 분모를 혼동하지 않습니다. Futility로 생성하지 않은 나머지는 NOT_EVALUATED이며 TOKEN_CAP 응답으로 세지 않습니다. Sentinel은 판독 자격만 검사하며 과제 정답으로 main 진입을 고르지 않습니다.', '', '판독된 기본 답에서 알려진 오답0은 종료된 응답에 조건을 건 기술통계입니다. 절단된 응답까지 정답이라고 가정하거나 전체 기초 능력이 확립됐다고 해석하지 않습니다. 한 줄 준수와 terminal 판독도 분리합니다. 이 실험은 간결 출력 지시 아래의 최종 선택을 측정하며, 실제 응답이 모두 설명 없는 직접답이었다고 주장하지 않습니다.', '']
    for s in ['A','B','C','E']:
        d=agg['studies'][s]; cal=d['calibration']; co=cal['outcome']; p0=d['P0']
        report += [f'### {s}: {LABELS[s]}', '',
            f"보정 gate: **{co['gate']}**. 기초 {cal['anchor_responses']}/{co['anchor_rows_planned']}행, 복합 sentinel {cal['main_or_sentinel_responses']}/{co['sentinel_rows_planned']}행을 실제 생성했습니다.", '',
            '| 기초 기능 | 관측 수 | 확인 정답 | 알려진 오답 | 미판독 | 가능한 최종 판독률 상한 | 가능한 최종 확인 정답률 상한 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        fut = co.get('elementary_anchor_futility') or {'groups':{}}
        for kind, q in co['readiness']['groups'].items():
            u=fut['groups'].get(kind,{})
            report += [f"| {kind} | {q['n']} | {q['correct']} | {q['known_wrong']} | {q['unparsed']} | {percent(u['best_possible_parse_fraction']) if u else '—'} | {percent(u['best_possible_confirmed_correct_fraction']) if u else '—'} |"]
        if fut.get('stop'):
            report += ['', '등록 futility 기능: '+', '.join(fut['failed_groups'])+'. 해당 main 가설은 NOT_EVALUATED이며 Negative가 아닙니다.']
        if co.get('sentinel_parse_futility'):
            sf=co['sentinel_parse_futility']
            report += ['', f"복합 sentinel 판독의 최종 가능 상한 {percent(sf['best_possible_final_parse_fraction'])}로 중단했습니다. 과제 정답/효과로 중단하지 않았습니다."]
        if 'outcome' in p0:
            out=p0['outcome']
            report += ['', f"P0 gate **{out['gate']}**, Main 생성 {p0['main_or_sentinel_responses']}행."]
            if p0['main_or_sentinel_responses']:
                report += [f"실제 main의 가설 분류는 **{out['classification']}**입니다."]
            else:
                report += ['기초 자격에서 멈췄으므로 main 가설 결과는 **NOT_EVALUATED**입니다. 계측 자격의 Inconclusive와 과학적 반증을 혼동하지 않습니다.']
            if out.get('main_quality'):
                q=out['main_quality']; report += [f"Main terminal 판독 {percent(q['parsed_fraction'])}; 확인 정답 {q['correct']}/{q['n']}, 알려진 오답 {q['known_wrong']}, 미판독 {q['unparsed']}. 정확도 최악 경계 [{percent(q['accuracy_bounds'][0])}, {percent(q['accuracy_bounds'][1])}]."]
            if out.get('estimand_valid'):
                report += ['', '| 등록 대비 | 식별 경계 | 탐색적 outer95 구간 | 독립 세계 |', '|---|---|---|---:|']
                for k,v in out['all_contrasts_diagnostic_if_invalid'].items():
                    lo,hi=v['identified_interval']; a,b=v['ci95_outer']
                    report += [f'| {k} | [{lo:.4f}, {hi:.4f}] | [{a:.4f}, {b:.4f}] | {v["families"]} |']
            elif p0['main_or_sentinel_responses']:
                report += ['Main은 실제 실행했으나 판독 자격이 부족해 primary는 null입니다. 원시 분모의 미판독 경계·진단은 JSON에 보존합니다. 이것은 main 미실행과 다릅니다.']
        report += ['']
    report += ['## 비용·검증·해석 한계', '',
        f"R10b 기술{agg['R10b_technical_forwards']} + 보정{agg['R10b_calibration_forwards']} + P0{agg['R10b_P0_forwards']} = {agg['R10b_total_forwards']} actual forward. 원래 R10의{agg['earlier_R10_forwards']}를 포함해 총{agg['cumulative_R10_R10b_forwards']}회입니다. 한 응답은 여러 forward를 사용합니다.", '',
        f"R10b 완결 응답{agg['R10b_responses']}개, terminal 상태 {json.dumps(agg['parse_status'],ensure_ascii=False)}, 한 줄 준수{agg['single_line_contract_compliant']}개. 누적 실측 GPU시간 {agg['cumulative_measured_GPU_seconds']:.3f}초. 최대 할당 메모리 {agg['peak_allocated_bytes']/1e9:.3f}GB.", '',
        '실제 BF16/FP32 훅 검증과 제한된 KV-cache/full-prefix spot check를 통과했습니다. CPU replay는 원시 token ID·최종답·입력 해시·실제 카운터·최초 futility 시점·동결된 통계를 재계산합니다. 모델 logit을 다시 실행한 독립 반복은 아닙니다.', '',
        '정상 판독은 의미 정답이나 가설 양성이 아닙니다. 원래 가설의 특정 효과가 Negative여도 해당 유한 생성기·모델·출력 계약에 한정하며, 전역적 능력이나 내부 인과 회로의 부재를 뜻하지 않습니다. 기초 실패는 해당 측정 자격 실패이며 가설 반증이 아닙니다. 8–12세계 bootstrap은 탐색적이며 여러 대비의 구간은 다중검정 보정이나 동시95% 구간이 아닙니다. P1/P2와 봉인 확인은 미평가입니다.', '',
        '[집계 JSON](../logs/summary/round10b_aggregate.json) · [원시 token CPU replay](../logs/summary/round10b_replay.json) · [사전등록](ROUND10B_REPAIR_PREREG_KO.md) · [원래 R10 실패](../logs/summary/round10_attempt1_abort.json)', '']
    sensitivity=read('round10b_uncertainty_sensitivity.json')
    report += ['## B의 퇴화 bootstrap에 대한 사후 민감도 점검', '',
        f"등록된 B Negative와 원래 primary 구간은 변경하지 않았습니다. 다만 {sensitivity['observed_families']}세계에서 효과가 모두0이라 bootstrap[0,0]이 된 것을 모집단 동등성 증명으로 해석하면 안 됩니다. 예시 IID 모집단에서10%세계만 대비+1이고 나머지는0이면 모집단 평균은10pp이지만,12개 모두0을 관측할 확률은 {sensitivity['counterexample_population']['probability_observe_n_zero_contrasts']:.4f}입니다.", '',
        f"추가 참고 모형에서 세계별 어떤 위반 사건도0/12라면 그 사건 확률의 단측95% 상한은 {sensitivity['reference_one_sided95_upper_event_probability']:.4f}입니다. 이것은 실제 유한 corpus의 IID를 입증한 CI가 아닌 가정을 드러낸 사후 민감도입니다. 따라서 **관측12세계의 등록 규칙상 Negative와 모집단10pp 가설의 미확립을 구분**합니다. 새 모델 호출이나 후보 변경은 없습니다.", '',
        '[사후 민감도 JSON](../logs/summary/round10b_uncertainty_sensitivity.json)', '']
    review_path=ROOT/'logs/summary/round10b_results_peer_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        report += ['## 실제 결과 공동 재검토와 다음 실험', '',f"재검토 상태: **{review['status']}**. 전체 원시 생성문 없이 집계·효과·불확실성·한계를 기존 두 채팅에 전달했습니다.", '']
        for item in review['adjudication']: report += ['- '+item]
        if review['recommendations']:
            report += ['', '| 추천순 | 다음 실험 | 목적·전제 |', '|---:|---|---|']
            for i,item in enumerate(review['recommendations'],1):report += [f"| {i} | {item['experiment']} | {item['purpose']} |"]
            proof=read('round10b_next_cpu_design.json')
            report += ['', '### 다음 명제를 위한 CPU 구성 예시: 모델 결과 아님', '',
                '명제 후보는 **국소 정답과 관련 사실을 고정한 상태에서, 묻지 않은 전역 제약이 국소 답 전달을 선택적으로 지연시키는가**입니다. 기존 A/C/E의 의미적 main을 평가했다고 주장하지 않으며, 형식·cap 재수리를 이름만 바꿔 재개하지 않습니다.', '',
                f"새8×8 조합 예시 한 구조군에서 exhaustive 배정 검사 {proof['permutation_checks']}회와 부분집합 검사를 교차해 구성 가능성을 확인했습니다. 추가 모델 호출0, CPU {proof['CPU_seconds']:.4f}초입니다.", '',
                '| 예시 | 독점 배정 가능 | 무제한 재사용 가능 | 중복 행 | 최소 실패 업무 집합 크기 |',
                '|---|---|---|---:|---:|']
            for name,v in proof['results'].items():
                witness=v['Hall_witness']
                report += [f"| {name} | {v['exclusive_feasible']} | {v['unlimited_reuse_feasible']} | {v['duplicate_rows']} | {witness['minimum_witness_size'] if witness else '없음'} |"]
            report += ['', '기본·실제전환·sham은 행/열 degree, 고정된 질의 행·union, degree로 층화한 행/열 두 집합씩의 교집합 histogram이 같습니다. 실제전환과 sham은 각각2행·2열의4edge 항목을 바꾸며, 변경 endpoint의 degree도2/2로 같습니다. 기본·sham은 1개·2개 업무 조건은 모두 만족하지만3개 업무가 자원2개에만 연결되어 전체 배정이 불가능합니다. 중복 행 없이도 이런 장애가 생깁니다. 보조 예시는 중복된 다중 자원 행이 있어도 전체 배정이 가능함을 확인합니다.', '',
                '**아직 한 구조군뿐입니다.** 이름만 바꾼 예시는 독립 표본이 아닙니다. 연결 성분 크기 등 더 높은 구조는 아직 맞추지 않았고, 실제 문장/token 길이의 동등성도 검증하지 않았습니다. 보조 중복 예시도 모든 primary 구조 통계를 맞춘 것은 아닙니다. 따라서 전역 가능성만의 인과 효과를 분리했다고 주장하지 않습니다.', '',
                '다음 CPU 단계는 비동형 여러 구조·연결 성분·표면 단서·정확한 token 길이를 맞춘 corpus와 sham을 확보하고 원문 선행과의 차별성을 검증하는 것입니다. 확보 후에도 새 사전등록이 필요하며, 예측은 **전환−sham의 답 전달 지연이 독점 조건에서만 커지고 재사용 조건에서는 사라지는가**입니다. 정답 final·오답 final·미전달을 분리하며, 완결답만의 평균 길이를 비교하지 않습니다. Sham/재사용에서도 같은 효과이거나 국소 판독이 안 되면 종료하고 회로 patching으로 넘어가지 않습니다.', '',
                '표현 길이·진행도 조절 자체는 이미 [Overclocking](https://arxiv.org/html/2506.07240v1), 모순으로 계산 비용을 늘리는 것은 [CODE](https://arxiv.org/html/2601.13112v1)와 인접합니다. 해가 없는 배정 문제와 서로 모순되는 사실은 다르지만, 그 차이만으로 신규성이 확립되지는 않습니다. 현재 GPU Go·논문 Go는0입니다.', '',
                '[CPU 검증 JSON](../logs/summary/round10b_next_cpu_design.json) · [예산](../configs/round10b_next_cpu_design.yaml) · [구성·독립 oracle 코드](../scripts/check_round10b_next_cpu_design.py)', '']
        else:
            report += ['', '현재 증거로 추가 GPU 실행을 추천하는 새 실험은 0개입니다. 계측 수리 자체나 아직 식별되지 않은 내부 기제를 새 논문 기여로 제시하지 않습니다. 이는 다른 설계의 가능성 부정이 아닙니다.']
        report += ['']
    (ROOT/'reports/ROUND10_REPAIR_RESULTS_KO.md').write_text('\n'.join(report).rstrip()+'\n')
    write('round10_execution_results.json',{'schema_version':1,'status':'ACTUAL_R10b_REPLAY_VERIFIED','studies':result_rows,'aggregate':'logs/summary/round10b_aggregate.json'})
    progress=read('round10_progress.json')
    progress.update(status='ACTUAL_R10b_REPAIR_EXECUTION_COMPLETE_REPLAY_VERIFIED',new_technical_forwards=108+agg['R10b_technical_forwards'],new_calibration_forwards=7684+agg['R10b_calibration_forwards'],new_P0_forwards=agg['R10b_P0_forwards'],calibration=agg['calibration_admitted'],P0={r['study']:r['classification'] for r in result_rows},next_action='Registered R10b complete; no further prompt/cap tuning. Inspect actual result review and ranked recommendations.')
    write('round10_progress.json',progress)
    print(json.dumps({'status':'BUILT_FROM_REPLAYED_SUMMARIES','studies':result_rows}))


if __name__=='__main__':main()
