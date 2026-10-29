#!/usr/bin/env python3
"""Generate P0 report/figure from measured summaries, never model inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'logs/summary/round3a_p0.json'
ANALYSIS=ROOT/'logs/summary/round3a_p0_analysis.json'
REPORT=ROOT/'reports/ROUND3A_RESULTS_KO.md'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def render(s,a):
    q=s['qualification'];stopped=q['status']=='STOPPED_AT_NUMERICAL_QUALIFICATION'
    load=json.loads((ROOT/'logs/summary/round3a_p0_model_load.json').read_text())
    rows=['# R3-A 실측: BF16/FP32 수치 검증에서 중단', '',
        '**추천 1번을 실제 모델에서 시작했으나, 네 후보 모두 사전등록한 수치 안정성 기준을 통과하지 못했다.** '
        '계획대로 P0에서 중단했으며 discovery·calibration·confirmation은 실행하지 않았다.' if stopped else
        '**P0 수치 검증을 완료했다.** Discovery는 별도 등록 후 진행한다.', '',
        f"새8 source family의32변형에서 **{s['condition_rows']:,}개 조건 행·{s['forward_calls']:,}회 실제 모델 forward**를 측정했다. "
        '훅·복원 검증과 실제 nonzero delta 전달은 모두 통과했다. 중단 원인은 기술 오류나 누락이 아니라, 현재 dose·위치·과제에서 효과가 dtype을 넘어 재현되지 않은 것이다.', '',
        '| 후보 | 평균 |d| BF16 (nats) | 평균 |d| FP32 (nats) | Signed-effect Spearman | FP32 |d|≥.02 입력 | 판정 |',
        '|---|---:|---:|---:|---:|---|']
    # Escape literal pipes inside math-like column headings for Markdown tables.
    rows[-2]='| 후보 | BF16 평균 절댓값 (nats) | FP32 평균 절댓값 (nats) | Signed-effect Spearman | FP32 절댓값≥.02 입력 | 판정 |'
    for r in q['candidates']:
        rows.append(f"| {r['id']} L{r['layer']}N{r['neuron']} | {r['mean_abs_signed_effect_BF16']:.6f} | {r['mean_abs_signed_effect_FP32']:.6f} | {r['spearman_signed_effects']:.3f} | {r['n_nontrivial_fp32']}/32 | {'통과' if r['eligible'] else '미통과'} |")
    rows += ['', 'd는 내용축 u−v에서 (m_plus−m_minus)/2로 정의한 부호 효과다. 32는8개 family의 규칙/기호 변형 수이며 독립 n=32가 아니다. '
        '위 수치는 이 고정 표본의 기술 통계이며 모집단 신뢰구간이나 의미 기능 검정이 아니다.', '',
        '![R3-A P0 수치 비교](../figures/round3a_p0_precision.svg)', '',
        '왼쪽은 동일 nominal dose에서의 평균 효과 크기(가로축 로그), 오른쪽은32 paired 효과의 순위 상관이다. '
        '점과 사각형으로 dtype을 구분했다. .70 선은 실행 전에 정한 수치 gate이며 유의성 기준선이 아니다.', '',
        '## 어떤 기준이 실패했나', '',
        '각 후보에서 FP32 부호 효과의 절댓값이0.02 nats 이상인 입력이 최소8개 필요했지만 모두0개였다. '
        '따라서 해당 입력의 부호 일치율은 **NOT_IDENTIFIABLE / 해당 없음**이며, 일치율0%로 바꾸어 보고하지 않는다. '
        'Signed Spearman도 모든 후보가 등록한0.70에 미달했다. 각 후보의 평균 절대 오차는 허용 상한0.05 안에 들었지만, 모든 조건을 함께 충족해야 하므로 다음 단계로 진행할 후보는0개다.', '',
        'BF16과 FP32 모두 좌표 개입의 actual delta는100% nonzero였다. 원 BF16 smoke28검사, 정렬 hook34검사, '
        '두 dtype의 read-only/동일-forward 복원/훅 제거 exact 검사 및 finite 검사를 통과했다. '
        '즉 개입이 전달되지 않아서 작은 FP32 효과가 나온 것은 아니다.', '',
        '## 해석할 수 있는 범위', '',
        'FP32는 같은 BF16 저장 가중치와 부동 버퍼 값을 올려 계산한 조건이다. 별도 native FP32 가중치의 결과나 실수 연산의 참값은 아니다. '
        '전체 모델의 연산 dtype을 바꾸었으므로 차이를 마지막 logit의 반올림, 중간 계층의 비선형 전파, 개입 자체의 반올림 중 하나로 국소화하지 못했다. '
        '관찰된 BF16 변화는 BF16 모델의 실제 인과적 변화지만, 이를 정밀도에 안정적인 의미 기능으로 해석할 근거가 부족하다.', '',
        'BF16에서만 보이는 변화 전부가 무작위 잡음이라는 결론, 네 좌표의 기능 부재, 효과의 정확한0, 다른 위치·dose·과제에서의 무효는 주장하지 않는다. '
        '적은 새 표본으로 R2의 원시 수치나 등록 검정을 소급 무효화하지도 않는다. R2-E의 예측 개선은 등록한 BF16 범위의 결과로 보존하고, '
        '정밀도에 일반적인 공통 원인·의미 기제 해석은 보류한다.', '',
        '## 행동과 미실행 단계', '',
        '| 조건 | Clean 정답 | 개입 후 다음 토큰 변경 |', '|---|---:|---:|']
    for dtype in ('bfloat16','float32'):
        b=a['baseline'][dtype]
        rows.append(f"| {dtype} | {b['forced_choice_correct']}/{b['n_cases']} (강제 선택·raw top-1 동일) | {b['candidate_top1_change_rows']}/256 후보 개입 행 |")
    rows += ['', '각 dtype의 clean 오답 변형은2개 source family에 포함됐다. 이는 P0 표본의 기술 통계이며, '
        '후속 confidence 식별 gate나 확증 결과로 사용하지 않는다. 본 과제의 자유 생성 행동은 측정하지 않았다.', '',
        '| 단계 | 상태 |', '|---|---|',
        '| P0 실제 12B 비교 | 실행 PASS · 수치 gate 미통과 |' if stopped else '| P0 실제 12B 비교 | 실행 PASS · gate 통과 |',
        '| 내용/기호/규칙 discovery | NOT_EVALUATED |',
        '| β·dose·대조 calibration | NOT_EVALUATED |',
        '| 독립64 family confirmation | NOT_EVALUATED |', '',
        'P0 실패 뒤 dose를 올리거나 판정선을 낮추어 같은 연구를 계속하지 않았다. 추천2번도 자동 실행하지 않았다. '
        '이 방향을 재개하려면 정밀도를 명시한 별도의 새 연구로 설계해야 한다. 우선 검토할 질문은 '
        '“좌표 개입의 의미 효과가 연산 정밀도에 얼마나 의존하며, 어느 연산 구간에서 커지는가”다. 이는 이번 결과에서 나온 후속 가설이며 아직 검증하지 않았다.', '',
        '## 자원·재현과 검증', '',
        f"Run `{s['run_id']}`. 본 run {s['elapsed_seconds']:.2f}초, 모델 로딩{load['load_seconds']:.2f}초는 별도다. "
        f"최대 allocated {s['peak_allocated_bytes']/1e9:.3f}GB, reserved {s['peak_reserved_bytes']/1e9:.3f}GB, "
        f"관측 최소 GPU 여유 {s['minimum_gpu_free_bytes']/1e9:.3f}GB였다. "
        f"총{s['forward_calls']} forward는 기본{s['condition_rows']}회에 smoke·generation/훅·복원{s['forward_calls']-s['condition_rows']}회를 더한 값이며 cap1,000회·1,800초 이내다. 단일 GPU 프로세스만 사용했다.", '',
        '모델 `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, '
        'seed20261029, RTX PRO 6000 Blackwell Max-Q, eager/batch1, 최종 prefill chat-template 위치다. '
        '실제 환경·코드/설정 SHA·모델 dtype·입력 위치·자원은 run metadata와 요약에 보존했다.', '',
        '- [실행 전 등록](../docs/ROUND3A_PREREGISTRATION.md) · [수치 기준과 예산](../configs/round3a.yaml)',
        '- [실측 요약](../logs/summary/round3a_p0.json) · [CPU 원시 재계산 검사](../logs/summary/round3a_p0_analysis.json)',
        '- [실행 코드](../scripts/run_round3a_p0.py) · [정확한 대비/수치 gate](../src/neuron_lab/round3a.py)',
        '- 원시 JSONL·baseline activation/logit cache·전체 prompt manifest는 로컬 `runs/`에만 보존한다.',
        '- CPU 재계산: `.venv/bin/python scripts/analyze_round3a_p0.py --check`.',
        '- 공개 paired 숫자만으로 gate 재계산: `.venv/bin/python scripts/replay_round3a_p0.py`. 원시 inference·전달량·hook을 독립 재검증하는 것은 아니다.',
        '- 보고서·SVG 최신성: `.venv/bin/python scripts/build_round3a_report.py --check`.', '',
        'CPU toy 시험은 대비식·분할·gate의 동작 검사이며 실제12B 결과와 구별한다. '
        f"실제 P0의 원시 SHA·등록 hash·과거 결과 보호·행렬 완결성과 예산 등 {len(a['checks'])}개 CPU replay 검사는 통과했다. "
        '새 연구의 Novelty나 출판 가능성은 이 비교만으로 확정하지 않는다.', '']
    return '\n'.join(rows)

def figure(s):
    cs=s['qualification']['candidates']
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','svg.hashsalt':'round3a-p0-fixed','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.1),gridspec_kw={'width_ratios':[1.25,1]})
    ys=np.arange(len(cs));labels=[f"{r['id']}  L{r['layer']} N{r['neuron']}" for r in cs]
    ax=axes[0]
    for y,r in zip(ys,cs):
        ax.plot([r['mean_abs_signed_effect_FP32'],r['mean_abs_signed_effect_BF16']],[y,y],color='#A0A0A0',lw=1)
    for key,lab,color,marker in [('mean_abs_signed_effect_BF16','BF16','#B36124','o'),('mean_abs_signed_effect_FP32','FP32 upcast','#245A81','s')]:
        ax.scatter([r[key] for r in cs],ys,color=color,marker=marker,s=52,label=lab,zorder=3)
    ax.set(xscale='log',xlim=(.00008,.08),yticks=ys,yticklabels=labels,xlabel='Mean |signed effect d| (nats; log scale)',title='Effect magnitude at the same nominal dose')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.legend(loc='lower left',fontsize=10)
    ax=axes[1]
    ax.axvline(0,color='#777777',lw=1);ax.axvline(.7,color='#333333',lw=1,ls='--')
    ax.scatter([r['spearman_signed_effects'] for r in cs],ys,s=52,color='#245A81',marker='D',zorder=3)
    for y,r in zip(ys,cs):ax.text(r['spearman_signed_effects']+.035,y,f"{r['spearman_signed_effects']:.3f}",va='center',fontsize=10)
    ax.text(.71,3.48,'Gate: 0.70',fontsize=10,ha='right')
    ax.set(xlim=(-.55,1),ylim=(-.5,3.7),yticks=ys,yticklabels=[r['id'] for r in cs],xlabel='Spearman: BF16 versus FP32 signed effects',title='Rank stability (32 paired variants)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.suptitle('R3-A P0: 8 generated source families; numerical qualification',fontsize=13)
    fig.tight_layout(rect=[0,.06,1,.92]);fig.text(.5,.018,'Same BF16-stored values upcast to FP32. Descriptive gates; no discovery or confirmation.',ha='center',fontsize=10)
    return fig

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    s=json.loads(SOURCE.read_text());a=json.loads(ANALYSIS.read_text())
    text=render(s,a);fig=figure(s);stream=io.BytesIO();fig.savefig(stream,format='svg',metadata={'Date':None})
    raw=b'\n'.join(x.rstrip() for x in stream.getvalue().splitlines())+b'\n'
    svg=ROOT/'figures/round3a_p0_precision.svg';preview=ROOT/'runs/round3a_p0_precision.png'
    source={str(p.relative_to(ROOT)):digest(p) for p in (SOURCE,ANALYSIS,ROOT/'logs/summary/round3a_p0_model_load.json',ROOT/'configs/round3a.yaml',Path(__file__))}
    meta={'schema_version':1,'source_sha256':source,'svg_sha256':hashlib.sha256(raw).hexdigest(),
          'scope':'P0 measured effects; 8 source families, 32 derivatives per coordinate; no inferential p values',
          'units':'signed semantic logit-margin change in nats; left mean absolute effect on log scale, right Spearman',
          'new_GPU_inference_executed':False}
    metadata=ROOT/'figures/round3a_p0_metadata.json'
    if args.check:
        if REPORT.read_text()!=text or svg.read_bytes()!=raw or json.loads(metadata.read_text())!=meta:raise SystemExit('R3-A report/figure is stale')
    else:
        REPORT.write_text(text);svg.write_bytes(raw);metadata.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        fig.savefig(preview,dpi=150)
    plt.close(fig)
    print(json.dumps({'status':'PASS','report':str(REPORT),'new_GPU_inference_executed':False}))

if __name__=='__main__':main()
