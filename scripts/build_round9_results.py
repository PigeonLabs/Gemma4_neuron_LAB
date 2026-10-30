#!/usr/bin/env python3
"""Render R9 actual summaries and scientific figures from reconciled JSON only."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import datetime
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/gemma4-r9-matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from neuron_lab.runtime import save_json


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def shown(x): return f'{x:.4f}'
def estimate(x): return f"{x['point']:.4f} [95% CI {x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"


def rows_for(summary):
    return [json.loads(x) for x in (ROOT / 'runs' / summary['run_id'] / 'rows.jsonl').read_text().splitlines()]


def save_svg(fig, path, **kwargs):
    fig.savefig(path, metadata={'Date': None}, **kwargs)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def figures(summaries, rows, aggregate):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'gemma4-r9-v1'})
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.6), layout='constrained')
    keys = list('ABCDE'); x = np.arange(5)
    anchor = [sum(r['role'] == 'anchor' for r in rows[k]) for k in keys]
    main = [len(rows[k])-n for k, n in zip(keys, anchor)]
    ax[0].bar(x, anchor, color='#436b95', label='Basic anchors')
    ax[0].bar(x, main, bottom=anchor, color='#82b7a0', label='Main / shared reference')
    for i, k in enumerate(keys):
        ax[0].text(i, anchor[i]+main[i]+12, str(anchor[i]+main[i]), ha='center')
        if not main[i]: ax[0].text(i, anchor[i]+60, 'Main NE', ha='center', color='#777777', fontsize=9)
    ax[0].set(xticks=x, xticklabels=['A\nInformation', 'B\nReward', 'C\nCollection', 'D\nHigher order', 'E\nResources'],
              ylabel='Actual full forwards', ylim=(0, max(np.array(anchor)+np.array(main))*1.24), title='Bounded P0 execution')
    ax[0].legend(frameon=False, loc='upper right')
    minimum = [min(v['accuracy'] for v in summaries[k]['outcome']['readiness']['groups'].values()) for k in keys]
    ax[1].bar(x, minimum, color='#436b95')
    main_formats = [summaries[k]['outcome'].get('full_top1_format_rate') for k in keys]
    for i, v in enumerate(main_formats):
        if v is not None:
            ax[1].plot(i, v, 'D', color='#a6523d', markersize=6, label='Main full-vocab valid format' if i == 0 else None)
            ax[1].text(i, v+.06, f'main {v:.1%}', color='#a6523d', ha='center', fontsize=8)
        else:
            ax[1].text(i, .035, 'Main NE', color='#ffffff', ha='center', fontsize=8)
    ax[1].axhline(.85, color='#9e5844', linestyle='--', linewidth=1, label='Basic anchor threshold')
    for i, v in enumerate(minimum): ax[1].text(i, v+.025, f'{v:.1%}', ha='center')
    ax[1].set(xticks=x, xticklabels=keys, ylim=(0, 1.15), ylabel='Full-vocabulary correct / anchor rows',
              title='Worst basic-anchor group in each study')
    ax[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.14), fontsize=9)
    fig.suptitle(f"R9 actual Gemma 4 12B P0: {aggregate['total_actual_forwards']} forwards including {aggregate['shared_technical_forwards']} shared technical")
    dest=ROOT/'figures/round9_P0_results.svg'; save_svg(fig,dest,bbox_inches='tight'); fig.savefig(ROOT/'runs/round9_P0_results_preview.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    save_json(ROOT/'figures/round9_P0_results_metadata.json', {'status':'PASS','source':'logs/summary/round9_execution_results.json',
        'source_sha256':sha(ROOT/'logs/summary/round9_execution_results.json'),'svg_sha256':sha(dest),
        'anchor_calls':dict(zip(keys,anchor)),'main_or_reference_calls':dict(zip(keys,main)),
        'minimum_anchor_accuracy':dict(zip(keys,minimum)),'shared_technical_counted_once':aggregate['shared_technical_forwards'],
        'main_full_vocab_valid_format':dict(zip(keys,main_formats)),
        'missing_main_is_not_zero_effect':True})
    effects=[]
    for k, label, key in [('A','A false approval excess','false_approval_excess'),('E','E global - local approval','false_global_minus_local_approval')]:
        o=summaries[k]['outcome']
        if key in o and o['gate'] == 'P0_MEASURED': effects.append((label,o[key]))
    o=summaries['B']['outcome']
    for reverse in [0,1]:
        if 'directional_U_choice_effect' in o and o['gate'] == 'P0_MEASURED':effects.append((f'B reward: listing {reverse}',o['directional_U_choice_effect'][f'totals0_reverse{reverse}']))
    o=summaries['C']['outcome']
    if 'selective_prior_choice_contrast' in o and o['gate'] == 'P0_MEASURED':
        effects.append(('C collection: narrative',o['selective_prior_choice_contrast']['0']))
        effects.append(('C collection: likelihood given',o['selective_prior_choice_contrast']['1']))
    if effects:
        fig,ax=plt.subplots(figsize=(9.8,4.8),layout='constrained')
        for i,(label,e) in enumerate(effects):
            ax.errorbar(e['point'],i,xerr=[[e['point']-e['ci95'][0]],[e['ci95'][1]-e['point']]],fmt='o',color='#335f87',capsize=4)
            ax.annotate(f" {e['point']:+.3f}",(e['point'],i),xytext=(6,6),textcoords='offset points',fontsize=9)
        ax.axvline(0,color='#888888',linewidth=.8);ax.axvline(.1,color='#9e5844',linestyle='--',linewidth=1)
        ax.set(yticks=range(len(effects)),yticklabels=[x[0] for x in effects],xlim=(-1.05,1.05),
               xlabel='Registered behavioral contrast (fraction of decisions); family bootstrap 95% CI',
               title='Different estimands: exploratory screens, not a common mechanism score')
        ax.invert_yaxis();dest=ROOT/'figures/round9_P0_effects.svg';save_svg(fig,dest);plt.close(fig)
        save_json(ROOT/'figures/round9_P0_effects_metadata.json',{'source':'logs/summary/round9_execution_results.json',
            'source_sha256':sha(ROOT/'logs/summary/round9_execution_results.json'),'svg_sha256':sha(dest),
            'effects':[{'label':label,**e} for label,e in effects], 'zero_events_do_not_establish_population_equivalence':True})
    else:
        # An unpublished draft plotted an invalid-format arithmetic zero. Remove
        # that demonstrated reporting error; no delivered figure is replaced.
        for rel in ['figures/round9_P0_effects.svg','figures/round9_P0_effects_metadata.json']:
            path=ROOT/rel
            if path.exists():path.unlink()
    o=summaries['D']['outcome']
    if 'conditions' in o:
        fig,ax=plt.subplots(figsize=(7.6,5.3),layout='constrained')
        for rep,marker,col in [('rule','o','#335f87'),('table','x','#ad6741')]:
            v=[x for x in o['conditions'] if x['representation']==rep]
            ax.scatter([x['low_order_max_error'] for x in v],[x['parity_mass_error'] for x in v],marker=marker,color=col,s=28,alpha=.8,label=rep)
        ax.axvline(.05,color='#777777',linestyle='--');ax.axhline(.15,color='#777777',linestyle='--')
        ax.set(xlabel='Maximum first/second-order marginal error',ylabel='Parity-mass absolute error',
               title='D: evaluate each q / code / description before averaging',xlim=(-.01,.51),ylim=(-.01,1.01))
        ax.legend(frameon=False);dest=ROOT/'figures/round9_parity_diagnostics.svg';save_svg(fig,dest);fig.savefig(ROOT/'runs/round9_parity_diagnostics_preview.png',dpi=140);plt.close(fig)
        save_json(ROOT/'figures/round9_parity_diagnostics_metadata.json',{'source':'logs/summary/round9_P0_D.json',
            'source_sha256':sha(ROOT/'logs/summary/round9_P0_D.json'),'svg_sha256':sha(dest),
            'conditions':len(o['conditions']),'low_order_max_error_cutoff':.05,'parity_error_min':.15,
            'scope':'restricted binary-grammar decoder; valid full-vocab mass and top1 remain separate original records'})


def main():
    replay=read(ROOT/'logs/summary/round9_P0_replay.json');assert replay['status']=='PASS'
    summaries={k:read(ROOT/f'logs/summary/round9_P0_{k}.json') for k in 'ABCDE'}
    rows={k:rows_for(v) for k,v in summaries.items()}; tech=read(ROOT/'logs/summary/round9_P0_technical.json')
    selection=read(ROOT/'logs/summary/round9_candidate_selection.json'); titles={c['id']:c['title'] for c in selection['candidates']}
    peer_path=ROOT/'logs/summary/round9_result_peer_review.json'
    peer=read(peer_path) if peer_path.exists() else {}
    guard_path=ROOT/'logs/summary/round9_interpretation_guard.json'
    guard=read(guard_path) if guard_path.exists() else None
    total=sum(v['actual_forward_calls'] for v in summaries.values())+tech['actual_forward_calls']
    assert total==replay['original_actual_model_forwards_verified']
    result={'schema_version':1,'date_kst':'2026-10-30','status':'ACTUAL_REGISTERED_P0_OUTPUTS_COMPLETE',
        'model_id':summaries['A']['model_id'],'model_revision':summaries['A']['model_revision'],
        'primary_dtype':summaries['A']['primary_dtype'],'shared_technical_forwards':tech['actual_forward_calls'],
        'science_forwards':sum(v['actual_forward_calls'] for v in summaries.values()),'total_actual_forwards':total,
        'scientific_classification_counts':dict(Counter(v['outcome']['classification'] for v in summaries.values())),
        'paper_Go_count':0,'P1':'NOT_EVALUATED','P2':'NOT_EVALUATED','rows':[],
        'post_result_review_status':peer.get('status','NOT_STARTED')}
    for k,v in summaries.items():
        o=v['outcome'];r=rows[k];ar=[x for x in r if x['role']=='anchor'];mr=[x for x in r if x['role']!='anchor']
        result['rows'].append({'id':k,'question':titles[k],'run_id':v['run_id'],'science_forwards':v['actual_forward_calls'],
            'anchor_rows':len(ar),'main_or_reference_rows':len(mr),'main_full_correct':sum(x['full_correct'] for x in mr) if k!='D' and mr else None,
            'main_accuracy':sum(x['full_correct'] for x in mr)/len(mr) if k!='D' and mr else None,
            'main_format_rate':sum(x['format_valid'] for x in mr)/len(mr) if mr else None,
            'gate':o['gate'],'classification':o['classification'],'paper_verdict':o['paper_verdict'],
            'raw_rows_sha256':v['raw_rows_sha256'],'summary_sha256':sha(ROOT/f'logs/summary/round9_P0_{k}.json'),
            'elapsed_seconds':v['elapsed_seconds'],'peak_allocated_bytes':v['peak_allocated_bytes'],'peak_reserved_bytes':v['peak_reserved_bytes']})
        if guard:
            g=guard['studies'][k];assert g['original_summary_sha256']==sha(ROOT/f'logs/summary/round9_P0_{k}.json')
            result['rows'][-1].update(estimand_valid=g['estimand_valid'],primary_estimate=g['primary_estimate'],
                primary_interpretation_status=g['interpretation_status'])
    save_json(ROOT/'logs/summary/round9_execution_results.json',result)
    figures(summaries,rows,result)
    decision_path=ROOT/'logs/summary/round9_final_decision.json'
    decision=read(decision_path) if decision_path.exists() else {}
    progress=read(ROOT/'logs/summary/round9_progress.json');progress.update(
        status=decision.get('status',peer.get('status',result['status'])),phase='R9 five actual P0s measured; raw replay complete; result-review status recorded separately',
        new_Gemma_model_forwards=total,new_P0='FIVE_EXECUTED_WITH_REGISTERED_GATES',
        new_P1='NOT_EVALUATED',new_P2='NOT_EVALUATED',
        recorded_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_json(ROOT/'logs/summary/round9_progress.json',progress)
    lines=['# R9 실제 P0 결과와 한계','',
        f"새 다섯 연구의 실제 모델 호출은 **{total}회**다. 과학 {result['science_forwards']}회와 공유 기술 {tech['actual_forward_calls']}회를 구별한다. 논문 Go는 아직0개며 P1/P2는 NOT_EVALUATED다.",'',
        '실행 전 [사전등록](ROUND9_P0_PREREG_KO.md)을 공개했고, 모든 과학 소스·입력·gates는 동결했다. 최초 공개 커밋은0a1d8ab3a4992435e6b6b572a2c9531061d66b93이다. [정확 결과 JSON](../logs/summary/round9_execution_results.json), [원시 행 재검산](../logs/summary/round9_P0_replay.json)을 참조한다.','',
        '원본 수치와 분류를 보존하며, 별도의 [해석 유효성 guard](../logs/summary/round9_interpretation_guard.json)에서 A/B/C/E의 주 추정값과 CI를 null로 둔다. 무효 형식을 NO로 코딩한 진단 수치를 후속 인과 증거로 재사용하지 않는다. [R9 측정 동결](../logs/summary/round9_measurement_freeze.json)과 [실제 template 감사](../logs/summary/round9_chat_mode_audit.json)도 분리했다.','',
        '![실제 호출·anchor](../figures/round9_P0_results.svg)','',
        '| 질문 | 과학 호출 | anchor/main | gate | 등록 분류 |','|---|---:|---:|---|---|']
    for r in result['rows']:
        lines.append(f"| {r['id']} {r['question']} | {r['science_forwards']} | {r['anchor_rows']}/{r['main_or_reference_rows']} | {r['gate']} | {r['classification']} |")
    lines+=['','## 등록한 대비','',
        '행동 대비의 추정 대상은 서로 다르다. CI는 세계 단위 탐색적 bootstrap이며 모집단 동등성·다중 후보의 확증 검정이 아니다. 형식이 무너진 A의 산술상0을 무효과 추정으로 그리지 않는다.','']
    if any(summaries[k]['outcome']['gate']=='P0_MEASURED' for k in 'ABCE'):
        lines += ['![등록 행동 대비](../figures/round9_P0_effects.svg)','']
    for k,v in summaries.items():
        o=v['outcome'];lines += [f'### {k}. {titles[k]}','',f"등록 분류 **{o['classification']}**, gate `{o['gate']}`.",'']
        if o['gate']=='STOPPED_AT_BASIC_ANCHORS':
            for name,g in o['readiness']['groups'].items():lines.append(f"- {name}: {g['correct']}/{g['n']} = {g['accuracy']:.2%}.")
            if k=='B':lines += ['', 'cue의 경로 선택 지시와 anchor의 YES/NO 질문이 충돌할 가능성이 있다. 이 결과로 합산 능력 부재를 단정하지 않는다.']
            if k=='C':lines += ['', '형식을 지킨 첫 토큰은48/96이다. 기본 계산 gate에는 형식 실패와 유효 코드의 오답이 함께 들어 있으므로 확률 추론 능력을 분리해 평가하지 못했다. 추가 사후 입력 감사에서96개 anchor의 정답 분수가 항상 첫 번째 선택지임을 확인했다. A/B 코드 교환은 이 위치 편향을 제거하지 않으므로, 통과했더라도 충분한 확률 계산 준비도 증거가 되지 못한다.']
            if k=='E':lines += ['', 'union-count의 유효 형식11/64개는 모두 정답이었다. 나머지53개는 형식 실패이므로 이 수치를 union 계산 능력17%로 해석하지 않는다.']
            lines+=['','main 미측정이다. 실패한 준비도를 회복하기 위한 prompt나 조건 수정은 하지 않았다.',''];continue
        if o['gate']=='STOPPED_AT_FULL_VOCABULARY_FORMAT':
            mr=[r for r in rows[k] if r['role']!='anchor'];freq=Counter(r['full_top1_text'] for r in mr)
            lines += [f"전체 어휘에서 지정 코드를 고른 main은 {sum(r['format_valid'] for r in mr)}/{len(mr)} ({o['full_top1_format_rate']:.2%})다. 첫 토큰 빈도: {dict(freq.most_common(5))}.",
                      '무효 형식 행을 YES=0으로 코딩한 보조 산술값은 과학적 무효과 추정이 아니다. 주 대비는 NOT_IDENTIFIABLE이며 제한 점수나 형식 조건부 선별로 구제하지 않는다.',''];continue
        if k=='A':
            lines += [f"무효 부분관측의 선택적 허위승인 초과: {estimate(o['false_approval_excess'])}. YES margin 초과: {estimate(o['YES_margin_excess_nats'])} nat.",
                      f"유효−무효관측 승인 차이: {estimate(o['compatible_minus_incompatible_approval'])}; 공통승리/불가능 control 정확도 {o['control_accuracy']:.2%}.",'']
        elif k=='B':
            lines += [f"총액을 안 준 조건의 U 선택 변화: 정순 {estimate(o['directional_U_choice_effect']['totals0_reverse0'])}, 역순 {estimate(o['directional_U_choice_effect']['totals0_reverse1'])}.",
                      f"배분 대비의 오류율: 합계 없음 {o['error_rate_without_totals']:.2%}, 합계 제공 {o['error_rate_with_totals']:.2%}.",'']
        elif k=='C':
            lines += [f"값 선별−우도보존 절차의 prior 선택 대비: narrative {estimate(o['selective_prior_choice_contrast']['0'])}, likelihood 제공 {estimate(o['selective_prior_choice_contrast']['1'])}.",
                      f"narrative 불변성 screen: {o['invariance_P0_screen_passed']}. {o['equivalence_status']}.",'']
        elif k=='D':
            lines += [f"모든 코드·설명을 통과한 선택적 고차 오류 family: {o['selective_q_families']}/{o['q_families']}.",
                      '![저차·고차를 분리한 진단](../figures/round9_parity_diagnostics.svg)','',
                      '각 q·코드·설명에서 먼저 계산한 분포다. 이진 문법 decoder의 정확한 분포이며 자유 생성 분포나 내부 난수 능력을 측정한 것이 아니다.','']
            lines += ['고유 prefix-tree 기준으로는 rule32개·full-table16개이며 모두 저차 기준을 실패했다. 등록된64조건은 반복을 포함하고 독립 단위는8 q-family다. anchors까지 합친421개 고유 token 입력을 독립 n으로 쓰지 않는다.',
                      '등록 Negative는 예상한 공동 패턴이 이 finite P0에 나타나지 않았다는 판정이다. 저차가 정상인 조건의 부분집합은 비어 있으므로, 그 부분집합 안의 고차 오류율이나 보편적 조건부 명제는 평가하지 못했다.','']
            for rep in ['rule','table']:
                v=[r for r in o['conditions'] if r['representation']==rep]
                lines += [f"{rep}: 저차 최대오차 범위 {min(r['low_order_max_error'] for r in v):.4f}–{max(r['low_order_max_error'] for r in v):.4f}; 저차 기준 통과 {sum(r['low_order_max_error']<=.05 for r in v)}/{len(v)}조건. weighted valid mass 최솟값 {min(r['grammar_weighted_valid_mass'] for r in v):.6f}.",'']
        elif k=='E':
            lines += [f"불가능 독점 세계의 전체−국소 허위승인 대비: {estimate(o['false_global_minus_local_approval'])}.",
                      f"marked-union 정확도 {o['marked_union_accuracy']:.2%}; 가능/재사용 control {o['feasible_or_reusable_control_accuracy']:.2%}.",'']
        lines += [o.get('claim_limit',''),f"전체 어휘에서 유효 답변을 고른 비율: {o['full_top1_format_rate']:.2%}. 제한 logit로 형식 실패를 구제하지 않는다.",'']
    lines+=['## 재현·해석 범위','',
        f"모델 {result['model_id']}, revision `{result['model_revision']}`. FP32는 같은 저장 BF16값의 upcast 계산 조건이며 참값 모델이 아니다. 공유 기술34회에서 두 dtype의 identity·복원·softcap·계측 검사가 통과했다. R1–R8 보호 파일139개와 등록 소스·입력이 일치했다.",'',
        '각 run metadata에 환경·seed·dtype·장치·시간·메모리와 source hash를 보존했다. 원시 입력/행은 로컬 runs/에 있고 공개 JSON에 해시를 둔다. CPU replay는 기록된 top1/전체 어휘 질량의 사용·분류를 검산한다. 전체262144개 logits를 저장하지 않았으므로 이 숫자를 두 번째 GPU 평가로 재현했다고 하지는 않는다.','',
        '사후 설계 감사: B의 cue는 경로 선택을 지시하면서 anchor query는 합계의 YES/NO를 물었다. 이것은 명령 충돌 가능성이며 합산 능력 부재로 단정할 수 없다. C와 E의 anchor 실패에는 첫 답변 형식 실패가 포함된다. E의 union-count는 형식 유효11/64이며 그11개는 모두 정답이었다. 따라서 union-count 이해가17%라고 해석하지 않는다.','',
        'D의 전체 확률표에서 (q,code flip)과 (1−q,반대 flip)은 동일한 입력이 될 수 있다. 동일 family의 반복이고 새 표본이나 독립 codebook 증거가 아니다. shared uniform reference도 반복 입력을 포함한다. 등록 시 세계 안 반복을 허용한 input contract와 일치하며 실제 호출 수에서 제거하지 않는다.','',
        f"실측 후 협업의 현재 기록: `{peer.get('status','NOT_STARTED')}`. 실제 원문·접근 한계·이견은 [결과 심사 기록](../logs/summary/round9_result_peer_review.json)에 있다. P0 분류와 논문 투자 판단은 다르며 새 가설/P1에는 사용하지 않은 분할과 고정 예측·기여 판단이 필요하다.",
        '', '[실측 후 공동 심사·후속 선별·현재 탐색 종료](ROUND9_REVIEW_AND_NEXT_KO.md). 논문 가치가 검증된 새 주제는 찾지 못했다.']
    (ROOT/'reports/ROUND9_EXECUTION_RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'BUILT_FROM_MEASURED_JSON','total_actual_forwards':total,'classifications':result['scientific_classification_counts']}),flush=True)


if __name__=='__main__':main()
