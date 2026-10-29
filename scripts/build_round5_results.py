#!/usr/bin/env python3
"""Render the actual R5 bounded-search outcome from its summary JSON."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"reports/ROUND5_RESULTS_KO.md"
SOURCE=ROOT/"logs/summary/round5_discovery.json"
FIGURE=ROOT/"figures/round5_discovery_damage.svg"
META=ROOT/"figures/round5_discovery_damage_metadata.json"
NAMES={"direct_only":"직접 표","composition_only":"두 단계 표","dual_route":"양쪽 표","code_map_only":"코드 변환만"}


def render():
    s=json.loads(SOURCE.read_text())
    replay=json.loads((ROOT/"logs/summary/round5_replay.json").read_text())
    a=s["analysis"]
    assert s["status"]=="PASS" and replay["status"]=="PASS"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==replay["source_result_sha256"]
    stopped=a["gate_outcome"].startswith("STOPPED")
    title=("**등록한 차단 자격 gate에서 중단했다. 이 조건의 모집 명제는 NOT_IDENTIFIABLE이며, 회복·모집은 미측정이다.**"
           if stopped else "**Discovery 차단 쌍을 선택했다. 독립 calibration 자격을 통과하기 전에는 모집을 주장할 수 없다.**")
    lines=["# R5 실측 결과 — 지속 차단 아래 피드백에 따른 경로 모집", "",title,"",
        "GPT Pro Deep Research·Claude Opus 5.5 Extra Research와 검토한 2번 주제의 실제 12B 실험이다. "
        "연구 명제는 **같은 1차 차단에서 오류 이력만 바꾸어도 새 질의의 인과적 계산 의존성이 달라지는가**다. "
        "이번 결과는 그 명제를 검정할 수 있는 차단 쌍을 먼저 찾은 단계의 결과다.","",
        f"판정: **{a['gate_outcome']}**. 평가한 후보 **{s['candidate_count_evaluated']}개**, 적격 쌍 **{a['eligible_pair_count']}개**.","",
        "## 실제 실행과 기술 검증", "",
        f"- Run: `{s['run_id']}`. 모델 `{s['model_id']}`, revision `{s['model_revision']}`.",
        f"- 실제 전체 forward **{s['actual_forward_calls']:,}회** = 기술 검증 **{s['technical_forward_calls']}회** + scientific 관측 **{s['scientific_rows']:,}행**. "
        "한 관측마다 캐시 없이 전체 문맥을 다시 계산했다.",
        f"- 측정 시간 **{s['elapsed_seconds']:.2f}초**, peak allocated **{s['peak_allocated_bytes']/1e9:.2f}GB**, peak reserved **{s['peak_reserved_bytes']/1e9:.2f}GB**. "
        "GPU GB는 10진 단위다. 모델 로딩 시간은 별도 model-load JSON에 있다.",
        f"- 첫 50회 완료까지 **{s['first50']['seconds']:.2f}초**. BF16/FP32 기술 검증과 서로 다른 입력 길이를 포함하므로 동일 길이 scientific 처리율로 해석하지 않는다.",
        "- 실제 BF16·전체 FP32 upcast에서 읽기 전용/identity 훅, 모든 토큰 zero, query-only zero와 prefix 보존, 같은 모듈의 동시 차단, 동일 forward 복원, 훅 제거, tied-head/softcap 재생이 통과했다.",
        f"- 측정 seed `{s['measurement_seed']}`, corpus seed `{s['corpus_seed']}`. 입력 manifest SHA256 `{s['manifest_canonical_sha256']}`.",
        f"- 독립 discovery **{s['discovery_episodes']} episode**, episode당 고정된 질의 3개×4 witness. Calibration 16 / confirmation 48 episode의 모델 출력은 열지 않았다.",
        f"- 추론 전 공개 등록: [`{replay['pre_inference_public_registration_commit'][:12]}`](https://github.com/PigeonLabs/Gemma4_neuron_LAB/commit/{replay['pre_inference_public_registration_commit']}). "
        "11개 허용 파일의 원격 내용 해시를 다시 조회했다.","",
        "## Healthy 과제 준비", "",
        "| 제공 정보 | 제한된 A/B/C/D 정확도 | 전체 어휘 top1 정답 코드 정확도 |", "|---|---:|---:|"]
    for w,name in NAMES.items():
        lines.append(f"| {name} | {a['healthy_accuracy'][w]:.2%} | {s['healthy_unrestricted_top1_accuracy'][w]:.2%} |")
    lines += ["", "각 witness는 24개 질의이나 독립 통계 단위는 8개 episode다. 주 readout은 네 코드만의 argmax다. "
        "전체 어휘에서의 top1을 별도로 제시했으며, 자유 생성의 일반적인 품질 회복을 측정하지 않았다.","",
        "## 고정한 차단 후보의 결과", "",
        "층 15/22/26/35×크기 32/128×활성 대비 두 방향의 16개 native 집합이다. "
        "후보는 건강한 discovery 단일 경로 witness에서 query 구간 평균 절대 활성값의 정규화된 대비로만 선정했다. "
        "차단은 MLP.down_proj 입력 좌표를 모든 토큰에서 0으로 만든다. 상대 활성 대비가 손상 또는 의미 기능을 보장하지 않는다.","",
        "손상은 healthy−차단 정확도다. 양수는 악화, 음수는 개선이다. 아래 퍼센트포인트(pp)와 CI는 "
        "선택에 사용한 discovery의 기술 통계이며 확증 CI가 아니다. Episode 안의 질의를 먼저 평균한 뒤 8 episode를 10000회 재표집했다.","",
        "| 후보 | 양쪽 표 손상 pp [95% CI] | 직접 표 손상 pp | 두 단계 손상 pp | 코드 변환 손상 pp | 방향별 경로 특이성 pp | 적격 |", "|---|---:|---:|---:|---:|---:|---|" ]
    for c in a["candidates"]:
        d=c["damage"];ci=d["dual_route"]["ci95"]
        lines.append(f"| {c['id']} | {100*d['dual_route']['point']:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | "
            f"{100*d['direct_only']['point']:.2f} | {100*d['composition_only']['point']:.2f} | "
            f"{100*d['code_map_only']['point']:.2f} | {100*c['route_selectivity'] if c['route_selectivity'] else 0.:.2f} | {'통과' if c['eligible'] else '실패'} |")
    lines += ["", "자격은 각각 dual 손상≥20pp, 반대 단일 경로 손상 차≥15pp, 코드 변환 손상≤5pp, "
        "두 집합의 dual 손상 차≤10pp, disjoint 좌표, discovery-fixed 오류 특징 AUC≥.70이다. "
        "점추정 자격을 통과하더라도 독립 calibration에서 손상 CI 하한>10pp 및 고정 오류 프로파일의 일반화를 다시 검증해야 한다.",""]
    if a["candidates"]:
        lines += ["![모든 후보의 discovery 손상과 경로 특이성](../figures/round5_discovery_damage.svg)",""]
    if stopped:
        lines += ["## 해석", "",
            "16개 차단 모두 양쪽 표 조건에서 healthy와 같은 24/24 정답을 유지해, 관측 정확도 손상이 0pp였다. "
            "이 8 episode를 재표집한 [0,0] 구간은 모집단의 손상이 정확히 0이라는 동등성 증명이 아니다.", "",
            "등록한 bounded search에서, 회복을 검정할 만큼 손상되고 두 경로에 대해 반대 특이성을 갖는 두 차단을 확보하지 못했다. "
            "따라서 **이 과제·후보 선정법·층·크기의 조합으로는 모집 명제를 식별할 수 없다**. 차단을 실제로 전달하지 못한 기술 실패와는 구별한다.","",
            "이 결과는 모델이 차단 아래 적응할 수 없다는 증거도, 기능적 경로가 존재하지 않는다는 증거도 아니다. "
            "건강한 동작의 활성 대비로 후보를 찾는 방식과 이번 차단 범위가 필요한 손상을 만들지 못했다는 한계에 해당한다. "
            "손상 자격이 없으므로 회복률이나 모집 효과를 계산하지 않았으며, dose·층·후보·과제·임계값을 사후에 바꾸지 않았다.","",
            "다음 연구가 필요하다면, 먼저 discovery에서 **양쪽 경로의 실제 인과적 병목을 찾아 두 손상 프로파일을 만들 수 있는가**를 별도 연구로 검증해야 한다. "
            "이 진단 결과를 R5 모집 가설의 지지로 합치지 않는다.",""]
    lines += ["## 남은 명제의 상태", "", "| 항목 | 상태 |", "|---|---|",
        f"| 실제 12B 기술/훅 검증 | PASS |", f"| 제한된 차단 자격 탐색 | {a['gate_outcome']} |",
        "| 독립 calibration / confirmation | NOT_EVALUATED |", "| 오류 이력에 따른 정답률 회복 | NOT_EVALUATED |",
        "| Query-time 모집 및 norm replay | NOT_EVALUATED |", "| 보편적인 적응 불가능성 | 검정하지 않음 |", "",
        f"CPU 재생은 raw **{replay['scientific_rows_checked']:,}행**의 정답·코드·margin·호출 수·native 후보 순위·선택 gate를 확인했다. "
        f"이전 연구 파일 **{replay['protected_prior_files_verified']}개**의 해시가 유지되었다. "
        "CPU 재생과 합성 단위 테스트는 실제 모델의 독립 복제가 아니다.","",
        "[실측 JSON](../logs/summary/round5_discovery.json) · [CPU 재생](../logs/summary/round5_replay.json) · "
        "[기술 검증](../logs/summary/round5_technical.json) · [사전 등록](ROUND5_EXECUTION_REGISTRATION_KO.md) · "
        "[협업 설계](ROUND5_DESIGN_KO.md)",""]
    return "\n".join(lines)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    s=json.loads(SOURCE.read_text());c=s["analysis"]["candidates"]
    if not c:
        return
    plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","svg.hashsalt":"R5-actual-discovery","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True,gridspec_kw={"width_ratios":[1.15,1]})
    y=np.arange(len(c));colors=["#2769A2" if r["orientation"]=="composition" else "#A76816" for r in c]
    markers=["o" if r["orientation"]=="composition" else "s" for r in c]
    for i,r in enumerate(c):
        d=r["damage"]["dual_route"];lo,hi=d["ci95"];point=d["point"]
        axes[0].errorbar(100*point,i,xerr=[[100*(point-lo)],[100*(hi-point)]],fmt=markers[i],color=colors[i],capsize=3,markersize=6)
        axes[1].plot(100*r["route_selectivity"],i,markers[i],color=colors[i],markersize=6)
    axes[0].set_yticks(y,[r["id"].replace("composition","comp") for r in c]);axes[0].invert_yaxis()
    for ax,threshold,title in ((axes[0],20,"Dual-route accuracy damage"),(axes[1],15,"Oriented single-route selectivity")):
        ax.axvline(0,color="#424851",linewidth=.8)
        ax.axvline(threshold,color="#424851",linestyle="--",linewidth=1,label=f"Discovery gate: {threshold} pp")
        ax.set_title(title,loc="left",fontweight="bold",pad=18)
        ax.set_xlabel("Percentage points (healthy minus lesion)")
        ax.xaxis.set_major_locator(MultipleLocator(10));ax.grid(axis="x",color="#DCE1E6",linewidth=.6)
        ax.set_axisbelow(True);ax.legend(loc="upper right",frameon=False,fontsize=10)
    p=[100*r["damage"]["dual_route"]["ci95"][0] for r in c]+[100*r["damage"]["dual_route"]["ci95"][1] for r in c]
    q=[100*r["route_selectivity"] for r in c]
    axes[0].set_xlim(min(-5,min(p)-5),max(25,max(p)+5));axes[1].set_xlim(min(-5,min(q)-5),max(20,max(q)+5))
    fig.suptitle("R5 fixed lesion search | 8 discovery episodes, 24 queries per witness",x=.04,ha="left",fontsize=15,fontweight="bold")
    fig.text(.04,.035,"Left: paired 95% episode-bootstrap intervals (10,000 resamples); descriptive discovery selection, not confirmation.\n"
             "Circle/blue: composition-biased activation rank. Square/gold: direct-biased rank. Every candidate is shown.",fontsize=10,color="#424851")
    fig.subplots_adjust(left=.23,right=.98,top=.88,bottom=.15,wspace=.12)
    fig.savefig(FIGURE,metadata={"Date":None});fig.savefig(ROOT/"runs/round5_discovery_damage_preview.png",dpi=150)
    plt.close(fig)
    META.write_text(json.dumps({"schema_version":1,"source":"logs/summary/round5_discovery.json",
        "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"figure":"figures/round5_discovery_damage.svg",
        "unit":"percentage points","statistical_unit":"independent episode","episodes":8,
        "queries_per_witness":24,"interval":"descriptive paired 95% bootstrap,10000 resamples,seed51991",
        "all_candidates_shown":len(c),"scope":"discovery selection; no confirmatory inference"},indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--check",action="store_true");a=p.parse_args()
    expected=render()
    if a.check:
        assert DEST.read_text()==expected,"R5 report stale"
        if json.loads(SOURCE.read_text())["analysis"]["candidates"]:
            assert json.loads(META.read_text())["source_sha256"]==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    else:
        DEST.write_text(expected);figure()
    print("PASS: R5 actual result report agrees with summary JSON.")


if __name__=="__main__":
    main()
