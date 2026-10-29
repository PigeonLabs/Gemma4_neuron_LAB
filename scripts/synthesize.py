"""Record the bounded cycle's measured conclusions and explicit future costs."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def main():
    stages={name:json.loads((ROOT/f'logs/summary/{name}.json').read_text()) for name in
            ['model_load','smoke','pilot','survey','steering','validation','expanded','utility','analysis','explorer_qa','reproduction']}
    if any(stage.get('status')!='PASS' for stage in stages.values()):
        raise ValueError('Complete measured stages and verification are required before synthesis')
    a,u,s=stages['analysis'],stages['utility'],stages['survey']
    if not a['all_stages_available'] or len(a['candidates'])!=4:
        raise ValueError('Incomplete independent evidence')
    records=a['candidates']
    if any(r['causal_signature'][split]['positive_lower_bound'] for r in records for split in ['validation','test']):
        raise ValueError('Observed result changed: revise the written scientific conclusion')
    if any(r['correlation']['test']['auc']!=1 for r in records):
        raise ValueError('Observed test AUC changed: revise scientific conclusion')
    utility_rows=[r for r in u['aggregates'] if r['group']=='all']
    if len(utility_rows)!=20 or any(r['correct']!=16 or r['outputs_changed_count']!=0 for r in utility_rows):
        raise ValueError('Observed utility result changed: revise scientific conclusion')
    rate=stages['expanded']['elapsed_seconds']/stages['expanded']['intervention_count']
    relation=next(r['pearson_r'] for r in a['category_profile_relationships']['relations'] if r['candidate_ids']==['H001','H003'])
    result={
      'status':'PASS','scope':'First bounded, Codex-guided text-only research cycle; execution complete, semantic neuron mechanisms unresolved',
      'created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
      'findings':[
        {'text':f"BF16 실제 모델의 {stages['model_load']['neuron_count']:,}개 MLP 좌표를 확인하고 smoke 28/28을 통과했다. Discovery {s['prompts']}개 입력의 사용자 본문 {s['tokens_per_layer']:,}토큰/계층을 집계했다.", 'evidence':['logs/summary/model_load.json','logs/summary/smoke.json','logs/summary/survey.json']},
        {'text':'동결한 후보 4개의 독립 test 활성 AUC는 모두 1.0이었다. 각 후보의 양성은 3개, 음성은 42개인 자체 작성 편의 표본이다. 의미 기능을 확정하는 결과는 아니다.', 'evidence':'logs/summary/analysis.json'},
        {'text':'네 후보 모두 validation/test에서 norm-matched 방향 대조군 대비 범주별 개입 효과 차이의 95% CI가 0을 포함했다. 등록한 선택적 분포 효과 기준은 지지되지 않았으며 카탈로그 상태는 모두 ambiguous다.', 'evidence':['logs/summary/analysis.json','catalog/public/neurons.json']},
        {'text':f"별도 동결 utility 16문제(일반 8·번역 8), 실제 {u['generation_count']}회 생성에서 모든 ±1SD·zero·대조군 조건은 16/16 정답이고 출력 변화는 0이었다. 이 작은 쉬운 표본의 ceiling 결과로 일반 성능 무손상이나 뉴런의 불필요성을 주장하지 않는다.", 'evidence':'logs/summary/utility.json'},
        {'text':f"번역 후보 H001/H003의 discovery 15개 범주 평균 프로필은 Pearson r={relation:.6f}였다. 후보 선택·범주 구성의 영향을 받는 기술적 상관이며 연결이나 상호작용의 증거는 아니다.", 'evidence':'logs/summary/analysis.json'}],
      'hypothesis_verdicts':[
        {'id':r['id'],'status':'ambiguous','category_association':'observed on both fixed authored held-out splits',
         'registered_distribution_selectivity':'NOT_SUPPORTED_UNDER_TESTED_CONDITIONS',
         'semantic_function':'NOT_ESTABLISHED','falsified':False} for r in records],
      'negative_results':[
        'High category-activation separation did not yield the registered category-selective distribution advantage at the last chat-wrapper token.',
        'No utility accuracy or generation change was observed; the 16-item ceiling check cannot establish absence of harm.',
        'No semantic function hypothesis was conclusively falsified or causally established. Unsupported effects are distinguished from proof of absence.'
      ],
      'next_experiments':[
        {'priority':1,'title':'관측·개입 토큰 위치 일치','text':'본문 평균 관측과 마지막 wrapper 개입의 불일치를 해소한다. 동일 prompt의 본문/마지막 사용자 token/wrapper/decode 범위에서 +1SD·zero·matched 방향을 비교하고 새 독립 평가를 등록한다.',
         'planned_condition_rows':5760,'estimated_gpu_seconds':math.ceil(rate*5760*1.5),'novelty':'NOT_EVALUATED'},
        {'priority':2,'title':'어휘·길이 대조와 반례 확대','text':'번역 요청·언어명 언급·비번역 문장을 길이와 문자 체계별로 맞추고, 프로그래밍 요청에서 Python/Rust/R 이름과 코드 의도를 교차한다. 4대4 family 검증의 작은 표본을 늘린다.',
         'planned_condition_rows':3840,'estimated_gpu_seconds':math.ceil(rate*3840*1.5),'novelty':'NOT_EVALUATED'},
        {'priority':3,'title':'후기 정규화와 dose 비단조성 분리','text':'down-projection 뒤 RMSNorm의 실제 residual perturbation을 측정하고 norm을 맞춘다. 현재 작은 dose sweep의 비단조 평균을 normalization·BF16 rounding·prompt 변동 중 하나로 단정하지 않는다.',
         'planned_condition_rows':2880,'estimated_gpu_seconds':math.ceil(rate*2880*1.5),'novelty':'NOT_EVALUATED'}],
      'cost_estimate_basis':{'observed_test_gpu_wall_seconds_per_condition_row':rate,'headroom_multiplier':1.5,
          'qualification':'Planning estimates from this model/run, not measured future runtimes. Generation mix and instrumentation may change cost.'},
      'limitations':[
        'Discovery는 범주당4개 prompt, held-out 양성은 범주당3개다. 토큰이나 random seed를 독립 prompt 수로 계산하지 않았다.',
        '관측은 사용자 본문 평균, 개입은 마지막 chat-template token이다. 자연 활성 범위와 위치가 다르며 음의 dose가 관측 범위를 벗어나는 경우도 있다.',
        '유한 64-token reservoir quantile은 근사치이며 tail·rare activation의 모집단 추정이 아니다.',
        '16개 utility는 별도 post-validation 편의표본이며 정답률이 ceiling이다. 장문·다중모달·대표 benchmark·다른 모델 seed·외부 backend는 미평가다.',
        '범주별 상관·프로필 유사도·KL 변화는 의미 기능이나 뉴런의 인과적 필요성·충분성을 입증하지 않는다.',
        '후속 후보의 신규성·논문 가능성은 NOT_EVALUATED다. 단일 뉴런 개입·random norm 대조 자체는 선행연구에 존재한다.'
      ],
      'source_sha256':{f'logs/summary/{name}.json':hashlib.sha256((ROOT/f'logs/summary/{name}.json').read_bytes()).hexdigest() for name in stages},
      'total_condition_rows':sum(stages[x]['intervention_count'] for x in ['steering','validation','expanded']),
      'utility_generation_count':u['generation_count'],
      'research_loop':'Observed discovery -> Codex competing hypotheses -> new falsification prompts -> validation -> untouched test -> utility -> synthesis. New studies remain future work.'
    }
    (ROOT/'logs/summary/synthesis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','condition_rows':result['total_condition_rows'],'utility_generations':u['generation_count'],'semantic_status':'ambiguous'}))


if __name__=='__main__':
    main()
