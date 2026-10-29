"""Build a source-backed R7 execution report and scientific figure from all five runs."""
from __future__ import annotations
from pathlib import Path
import json,hashlib
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='round7-fixed';matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def pct(s):return f"{s['correct']}/{s['rows']} ({s['accuracy']:.1%})"
def stat(s):return f"{s['point']:.5g} [{s['ci95'][0]:.5g}, {s['ci95'][1]:.5g}]"
def B_behavior(s):
    rr=[json.loads(x) for x in (ROOT/'runs'/s['run_id']/'rows.jsonl').read_text().splitlines()];groups=defaultdict(dict)
    for r in rr:
        if r['phase']!='P0':continue
        f=r['factors'];groups[(r['episode_id'],f['timing'],f['distance'],f['query_index'])][r['arm']]=r
    rows=[]
    for timing in ('late','early'):
        for dist in ('short','long'):
            cells=[v for (w,t,d,q),v in groups.items() if t==timing and d==dist]
            if not cells:continue
            n=len(cells);row={'timing':timing,'distance':dist,'paired_queries':n,'independent_worlds':len({w for w,t,d,q in groups if t==timing and d==dist})}
            for arm in ('clean','seal','null','identity','main_same','main_other','pre_same','row1_same'):
                row[arm+'_base_correct']=sum(bool(v[arm]['correct']) for v in cells)
                row[arm+'_full_top1_code']=sum(v[arm]['full_top1_code'] is not None for v in cells)
            row['same_key_donor_success_restricted']=sum(v['main_same']['prediction']==v['donor_same']['expected'] for v in cells)
            row['same_key_donor_success_full_vocab']=sum(v['main_same']['full_top1_code']==v['donor_same']['expected'] for v in cells)
            row['other_key_base_preserved_full_vocab']=sum(v['main_other']['full_top1_code']==v['clean']['expected'] for v in cells)
            rows.append(row)
    return {'scope':'posthoc descriptive behavior audit, not additional registered success gate','cells':rows}

def forest(ax,o,keys,title,threshold=.1,units='Logit contrast (nat)'):
    ax.set_title(title,loc='left',fontweight='bold')
    if 'statistics' not in o:
        ax.text(.05,.55,o['gate'].replace('STOPPED_AT_','Stopped: ').replace('_',' '),transform=ax.transAxes,wrap=True);ax.set_axis_off();return
    st=o['statistics'];keep=[k for k in keys if k in st]
    for y,k in enumerate(keep):
        v=st[k];a,b=v['ci95'];ax.plot([a,b],[y,y],color='#304b6c',lw=2);ax.plot(v['point'],y,'o',color='#136d75',ms=5)
    ax.axvline(0,color='#a9afb5',lw=.8);ax.axvline(threshold,color='#b04a3a',ls='--',lw=1)
    ax.set_yticks(range(len(keep)),[k.replace('_',' ') for k in keep]);ax.invert_yaxis();ax.set_xlabel(units+'; 95% world CI');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)

def main():
    studies=[]
    for letter in 'ABCDE':
        p=ROOT/f'logs/summary/round7_{letter}_P0.json';assert p.exists(),f'{letter} unmeasured; refuse complete-round report'
        s=read(p);assert s['status']=='PASS';o=s['outcome'];meta=read(ROOT/'runs'/s['run_id']/'metadata.json');phases=Counter(json.loads(x)['phase'] for x in (ROOT/'runs'/s['run_id']/'rows.jsonl').read_text().splitlines())
        studies.append({'letter':letter,'run_id':s['run_id'],'classification':o['classification'],'gate':o['gate'],'science_calls':s['actual_forward_calls'],'technical_calls':s['technical_forward_calls_separate'],'phase_calls':dict(phases),'elapsed_seconds':s['elapsed_seconds'],'peak_GPU_allocated_bytes':s['peak_allocated_bytes'],'peak_CPU_RSS_bytes':s['peak_CPU_RSS_bytes'],'summary_path':str(p.relative_to(ROOT)),'summary_sha256':sha(p),'config_sha256':sha(ROOT/f'configs/round7_{letter}.yaml'),'raw_rows_sha256':s['raw_rows_sha256'],'model_revision':s['model_revision'],'dtype':s['primary_dtype'],'measurement_seed':s['measurement_seed'],'corpus_seed':s['corpus_seed'],'environment':meta['environment'],'outcome':o})
    replay=read(ROOT/'logs/summary/round7_replay.json');assert replay['status']=='PASS' and len(replay['studies'])==5
    science=sum(x['science_calls'] for x in studies);tech=sum(x['technical_calls'] for x in studies);assert science+tech==replay['total_actual_forwards']
    behavior=B_behavior(read(ROOT/'logs/summary/round7_B_P0.json'))
    out={'schema_version':1,'round':'R7','date_kst':'2026-10-30','execution_status':'ALL_FIVE_BOUNDED_P0_EXECUTED','scientific_forwards':science,'technical_forwards':tech,'total_actual_forwards':science+tech,'independent_GPU_replications':0,'backbones':1,'studies':studies,'B_descriptive_behavior':behavior,'A_protocol_defect':read(ROOT/'logs/summary/round7_input_audit.json'),'P1':'NOT_EVALUATED','P2':'NOT_EVALUATED','novelty':'UNVERIFIED; paper ranking requires actual post-result peer review','prior_studies':'unchanged','verification':'independent CPU raw-row replay, not independent model rerun','readout_scope':'registered predictions/quality use argmax among A/B/C/D; full-vocabulary top1 is retained separately; no open generation evaluation'}
    save(ROOT/'logs/summary/round7_execution_results.json',out)
    fig,axes=plt.subplots(2,3,figsize=(15,8.2));fig.subplots_adjust(left=.11,right=.98,bottom=.1,top=.87,wspace=.65,hspace=.6)
    a=studies[0]['outcome']['readiness'];ks=list(a);colors=['#487e89' if k.startswith('code') else '#a7b4c4' if k.startswith('rule_1') else '#b65a4b' for k in ks];ax=axes[0,0]
    ax.barh(range(len(ks)),[a[k]['accuracy'] for k in ks],color=colors);ax.set_yticks(range(len(ks)),[k.replace('_',' ') for k in ks]);ax.set_xlim(0,1.05);ax.scatter([.95 if k=='code_anchor' else .85 for k in ks],range(len(ks)),marker='|',s=160,color='#743a34',label='Registered minimum');ax.invert_yaxis();ax.set_title('A: anchor STOP; R1 inputs conflicted',loc='left',fontweight='bold');ax.set_xlabel('Restricted-code accuracy; rule min .85, code min .95')
    forest(axes[0,1],studies[1]['outcome'],['late_short','late_long','control_excess'],'B: sealed-graph donor selectivity')
    forest(axes[0,2],studies[2]['outcome'],['compose_before','interaction_before','compose_after','interaction_after'],'C: obsolete-history contrast')
    forest(axes[1,0],studies[3]['outcome'],['forward_before','inverse_before'],'D: task-selective ordering')
    e=studies[4]['outcome']
    if 'statistics' in e:forest(axes[1,1],e,['primary','lookup','position'],'E: prospective prediction gain',.02,'ΔR²')
    elif 'calibration_mean_absolute_effect' in e:
        ax=axes[1,1];v=e['calibration_mean_absolute_effect'];ax.barh(list(v),list(v.values()),color='#487e89');ax.axvline(.02,color='#b04a3a',ls='--');ax.set_title('E: stopped at calibration',loc='left',fontweight='bold');ax.set_xlabel('Mean absolute margin effect (nat)')
    else:forest(axes[1,1],e,[],'E: qualification STOP')
    ax=axes[1,2];ax.bar([x['letter'] for x in studies],[x['science_calls'] for x in studies],color='#304b6c',label='Science budget');ax.bar([x['letter'] for x in studies],[x['technical_calls'] for x in studies],bottom=[x['science_calls'] for x in studies],color='#adc8c3',label='Technical');ax.legend(frameon=False,fontsize=9,loc='upper left');ax.set_title('Actual counted model forwards',loc='left',fontweight='bold');ax.set_ylabel('Calls');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('R7: five distinct questions, bounded P0 outcomes',x=.07,ha='left',fontsize=18,fontweight='bold');fig.text(.07,.91,'P0 is exploratory. Stops and protocol defects are retained. No P1/P2 claim.',fontsize=11,color='#59626e')
    fp=ROOT/'figures/round7_P0_results.svg';fig.savefig(fp,metadata={'Date':None});fp.write_text('\n'.join(x.rstrip() for x in fp.read_text().splitlines())+'\n');fig.savefig(ROOT/'runs/round7_P0_results.png',dpi=140);plt.close(fig)
    save(ROOT/'figures/round7_P0_results_metadata.json',{'source':'logs/summary/round7_execution_results.json','source_sha256':sha(ROOT/'logs/summary/round7_execution_results.json'),'figure_sha256':sha(fp),'generator':'scripts/build_round7_report.py','generator_sha256':sha(Path(__file__)),'uncertainty':'95% paired world bootstrap; quality bars are proportions without CI; no condition-row independence','A_defect':'R1 conflicted anchors invalid as pure competence evidence','scope':'actual 12B measurements, not synthetic unit tests'})
    lines=['# R7 실제 P0 결과와 해석','',f'독립 질문 다섯 개에 실제 **{science+tech}회** forward를 수행했다. Science budget {science}회에는 보정·과제 준비도·B 봉인 instrument가 포함되며, 별도 BF16/FP32 기술 검사 {tech}회다. P1/P2는 NOT_EVALUATED다.','', '원본 BF16 파라미터 값을 FP32로 상향한 동일값 비교이며 FP32를 정답 모델로 간주하지 않는다. 모델 revision은 `'+studies[0]['model_revision']+'`이다. 한 번에 GPU 프로세스 하나만 실행했다.','', '| 후보 | 실제 science / technical | 분류 | 등록 gate |','|---|---:|---|---|']
    for x in studies:lines.append(f"| {x['letter']} | {x['science_calls']} / {x['technical_calls']} | {x['classification']} | {x['gate']} |")
    lines+=['','등록 준비도와 주효과는 A/B/C/D 중 argmax와 그 logit margin을 사용한다. 전체 vocabulary top-1은 별도로 보존했으며 자유 생성 평가를 수행하지 않았다. 이 구분 없이 모델의 전체 정확도나 의미 동작으로 일반화하지 않는다.','', 'A의 48개 R1 anchor는 R0 새 예시와 충돌하는 설계 오류다. 해당 점수를 순수 규칙 실행 능력이나 논리 전파의 부재로 해석하지 않는다. 원본 입력과 중단 판정을 보존했다. 모순 없는 R0 near 17/24와 far 8/24도 기준 미만이지만 사후 제외·재실행을 하지 않았다.','', '![실제 P0 요약](../figures/round7_P0_results.svg)','']
    for x in studies:
        o=x['outcome'];lines += [f"## {x['letter']} — {x['classification']}",'']
        if 'readiness' in o:lines+=['준비도: '+', '.join(f'{k} {pct(v)}' for k,v in o['readiness'].items())+'.','']
        if 'statistics' in o:
            lines+=['| 등록 효과 | 평균 [95% 세계 bootstrap CI] |','|---|---|']+[f'| {k} | {stat(v)} |' for k,v in o['statistics'].items()]+['']
        if 'calibration_mean_absolute_effect' in o:lines+=['보정의 평균 절대 margin 변화(nat): '+', '.join(f'{k}={v:.6g}' for k,v in o['calibration_mean_absolute_effect'].items())+'. 고정 dose를 변경하지 않았다.','']
        lines += [f"Run `{x['run_id']}`. science 경과 {x['elapsed_seconds']:.2f}초, peak GPU allocated {x['peak_GPU_allocated_bytes']/1e9:.2f} GB. 정확한 seed·환경·분할·해시·시간은 [집계 JSON](../logs/summary/round7_execution_results.json)에 있다.",'']
    lines += ['## B의 실제 답 변화: 사후 기술 통계','', '등록된 logit 성공 기준과 행동 전달을 구별하기 위한 기술 집계다. 아래 쌍들은 독립 세계 안의 반복 질의이며 추가 성공 gate로 사용하지 않았다.','', '| 질의 공개 / 간격 | query 행 | seal 원답 정확 | same-key donor 답(full vocab) | other-key 원답 보존(full vocab) |','|---|---:|---:|---:|---:|']
    for b in behavior['cells']:lines.append(f"| {b['timing']} / {b['distance']} | {b['paired_queries']} | {b['seal_base_correct']} | {b['same_key_donor_success_full_vocab']} | {b['other_key_base_preserved_full_vocab']} |")
    lines+=['', 'B는 47개 층의 token 궤적을 반복 교환하며 carrier의 원문 접근 자체는 계속 허용한다. 따라서 단일 상태의 저장 용량, 원문 재열람 없이 유지되는 기억, 자연 모델에서의 필요성을 입증한 것이 아니다. null graph equality는 우회 봉인의 기술 증거이고 자연 저장 메커니즘의 증거가 아니다.','', 'C의 효과는 일치하는 최종 상태에서의 이력 treatment다. 단순 갱신 실패는 [RippleEdits](https://arxiv.org/html/2307.12976v2), 업데이트 범위와 순서의 넓은 문제는 [DeReLab](https://aclanthology.org/2026.emnlp-main.1625/)와 겹친다. D의 순서 효과는 n-gram과 독립된 구조 원인이 아니며 [Invisible Reasoning](https://arxiv.org/html/2607.22925v1)의 일반 filler 효과를 새 발견으로 세지 않는다. E의 예측 전이는 [SPARK](https://arxiv.org/html/2607.10296v1)의 susceptibility와 비교해야 하고, 공통 head/readout 대안을 통과해도 공통 인과 기제를 바로 뜻하지 않는다.','', '## 검증 범위','', '원시 행의 oracle·응답 코드·logits/softcap·margin·토큰 직렬화·개입 scope·forward 수·등록 해시·기존 91개 파일과 세계 단위 통계를 CPU에서 독립 재계산했다. GPU 독립 반복은 수행하지 않았다. synthetic CPU 단위 테스트는 실제 12B 측정과 별개다.','', '[원시 집계 재검산](../logs/summary/round7_replay.json) · [입력 결함·동치 감사](../logs/summary/round7_input_audit.json) · [사전등록 공개](../logs/summary/round7_BCDE_preregistration_publication.json) · [후보와 선행 대비](ROUND7_CANDIDATES_KO.md)','', '실제 결과를 두 연구 채팅에 동일한 핵심 증거로 전달한 뒤 독립 검토와 적대적 교차검토를 수집해야 한다. 완료된 이후의 판단은 별도 공동 검토 기록으로 남기며, 현재 문서 자체는 외부 모델의 사후 동의를 뜻하지 않는다.']
    (ROOT/'reports/ROUND7_EXECUTION_RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'PASS','actual_forwards':science+tech,'science':science,'technical':tech,'classifications':{x['letter']:x['classification'] for x in studies}},ensure_ascii=False))
if __name__=='__main__':main()
