#!/usr/bin/env python3
"""Create scientific steering figures only from successful measured stages.

Missing validation/test measurements are omitted, never replaced with zeros.
Available raw files are hash-checked. All plotted values, count denominators,
source hashes and analytical caveats are recorded in steering_metadata.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

BLUE, GOLD = "#2563a6", "#bd8b26"
INK, MUTED, GRID = "#202a35", "#566372", "#e2e7ed"
ROLES = ("positive_control", "negative_control", "near_miss", "paraphrase", "out_of_distribution")
ROLE_LABELS = ("Positive\ncontrol", "Negative\ncontrol", "Near-miss", "Paraphrase", "OOD")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, name, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: a finite measured value is required")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name}: observation is outside the valid range")
    return float(value)


def count(value, name) -> int:
    finite(value, name, 1)
    if int(value) != value:
        raise ValueError(f"{name}: integer count required")
    return int(value)


def load_measured(stage: str, hypotheses: dict, hypotheses_sha: str):
    path = ROOT / f"logs/summary/{stage}.json"
    if not path.is_file():
        return None, {"status": "NOT_EVALUATED", "reason": f"No measured {stage} summary"}
    summary = json.loads(path.read_text())
    if summary.get("status") != "PASS":
        return None, {"status": summary.get("status", "NOT_EVALUATED"), "reason": "Stage did not record PASS"}
    if summary.get("stage") != stage or not re.fullmatch(r"\d{8}T\d{6}_" + stage, summary.get("run_id", "")):
        raise ValueError(f"{stage}: stage/run identity mismatch")
    if summary.get("model_revision") != hypotheses["model_revision"] or summary.get("hypotheses_sha256") != hypotheses_sha:
        raise ValueError(f"{stage}: frozen hypothesis or model revision mismatch")
    expected_split = {"steering": "discovery", "validation": "validation", "expanded": "test"}[stage]
    if summary.get("split") != expected_split:
        raise ValueError(f"{stage}: wrong data partition")
    count(summary.get("prompt_count"), stage + " prompt count")
    count(summary.get("candidate_count"), stage + " candidate count")
    if "95%" not in summary.get("confidence_interval", "") or "bootstrap" not in summary.get("confidence_interval", "").lower():
        raise ValueError(f"{stage}: cannot label confidence intervals without the recorded method")
    candidate_ids = {c["id"] for c in hypotheses["candidates"]}
    seen = set()
    for row in summary.get("aggregates", []):
        key = row.get("candidate_id"), row.get("condition")
        if key in seen or key[0] not in candidate_ids:
            raise ValueError(f"{stage}: duplicate or unknown candidate aggregate")
        seen.add(key)
        n = count(row.get("n_prompts"), "aggregate n_prompts")
        if n > summary["prompt_count"] or count(row.get("n_interventions"), "n_interventions") < n:
            raise ValueError(f"{stage}: aggregate count denominator mismatch")
        metric = row.get("metrics", {}).get("kl_base_to_changed", {})
        mean, low, high = [finite(metric.get(k), "KL " + k, 0) for k in ("mean", "ci_low", "ci_high")]
        if low > high or count(metric.get("n"), "KL prompt denominator") != n:
            raise ValueError(f"{stage}: confidence interval or denominator mismatch")
        if not low <= mean <= high:
            raise ValueError(f"{stage}: plotted mean is outside its recorded interval")
    provenance = {"status": "PASS", "source": path.relative_to(ROOT).as_posix(),
                  "source_sha256": digest(path), "run_id": summary["run_id"],
                  "model_revision": summary["model_revision"], "split": summary["split"],
                  "prompt_count": summary["prompt_count"], "candidate_count": summary["candidate_count"]}
    raw = ROOT / "runs" / summary["run_id"] / "interventions.jsonl"
    if raw.is_file():
        if digest(raw) != summary.get("raw_sha256"):
            raise ValueError(f"{stage}: raw intervention file hash mismatch")
        provenance["raw_sha256_verified"] = True
        # Independently reconcile prompt-weighted means, including seed averaging.
        groups = defaultdict(lambda: defaultdict(list))
        with raw.open() as handle:
            for line in handle:
                row = json.loads(line)
                groups[(row["candidate_id"], row["condition"])][row["prompt_id"]].append(
                    finite(row["kl_base_to_changed"], "raw KL", 0))
        for row in summary.get("aggregates", []):
            prompts = groups[(row["candidate_id"], row["condition"])]
            mean = float(np.mean([np.mean(values) for values in prompts.values()]))
            if len(prompts) != row["n_prompts"] or not math.isclose(mean, row["metrics"]["kl_base_to_changed"]["mean"], rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError(f"{stage}: published KL mean does not reproduce from raw prompt rows")
    else:
        provenance["raw_sha256_verified"] = False
        provenance["raw_check_scope"] = "Raw file absent; public summary used without an independent raw-file check"
    return summary, provenance


def theme() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8,
        "axes.titlesize": 11, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": .7, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "gemma4-neuron-lab-steering"})


def header(fig, title: str, subtitle: str) -> None:
    fig.text(.07, .96, title, va="top", fontsize=18, fontweight="bold")
    fig.text(.07, .895, subtitle, va="top", fontsize=10, color=MUTED)
    fig.text(.975, .96, "✽", ha="right", va="top", fontsize=17, color=GOLD)


def footer(fig, note: str, sources: str) -> None:
    fig.text(.07, .082, note, va="top", fontsize=9, color=MUTED)
    fig.text(.07, .032, sources, va="top", fontsize=8, color=MUTED)


def intervals(ax, x, mean, low, high, color, marker="o", label=None, fill=True, connect=True):
    if connect:
        ax.plot(x, mean, color=color, lw=1.5)
    ax.vlines(x, low, high, color=color, lw=1.2, alpha=.85)
    for xx, lo, hi in zip(x, low, high):
        ax.hlines([lo, hi], xx-.045, xx+.045, color=color, lw=1.2)
    ax.scatter(x, mean, s=58, marker=marker, facecolors=color if fill else "white",
               edgecolors=color, linewidths=1.5, zorder=4, label=label)


def dose_plot(summary: dict, candidate: dict, source: dict):
    rows = {r["condition"]: r for r in summary["aggregates"] if r["candidate_id"] == candidate["id"]}
    doses = []
    for condition, row in rows.items():
        match = re.fullmatch(r"add_([+-](?:\d+(?:\.\d+)?))sd", condition)
        if match:
            doses.append((float(match[1]), row))
    doses.sort(key=lambda value: value[0])
    if not doses:
        return None, {"status": "SKIPPED", "reason": "No measured dose rows for this primary candidate"}
    observed_doses = [dose for dose, _ in doses]
    if observed_doses != [-2., -1., -.5, .5, 1., 2.]:
        raise ValueError("Dose plot requires the frozen measured +/-0.5, +/-1, +/-2 SD sweep")
    ns = {row["n_prompts"] for _, row in doses}
    if len(ns) != 1:
        raise ValueError("Dose rows have incomparable prompt denominators")
    n = next(iter(ns))
    control = rows.get("random_direction_projected")
    baseline = rows.get("baseline")
    if baseline is None or baseline["metrics"]["kl_base_to_changed"]["mean"] != 0:
        raise ValueError("Measured baseline KL=0 is required before omitting it from the dose plot")
    bounds = [row["metrics"]["kl_base_to_changed"][key] for _, row in doses for key in ("mean", "ci_low", "ci_high")]
    if control:
        if control["n_prompts"] != n:
            raise ValueError("Projected-direction control and dose prompt counts differ")
        bounds.extend(control["metrics"]["kl_base_to_changed"][k] for k in ("mean", "ci_low", "ci_high"))
    positive = [value for value in bounds if value > 0]
    scale = "log" if all(value > 0 for value in bounds) else "symlog" if positive else "linear"
    fig, ax = plt.subplots(figsize=(11.8, 6.5))
    fig.subplots_adjust(left=.105, right=.965, bottom=.23, top=.73)
    header(fig, f"Steering strength and distribution change · {candidate['id']}",
           f"L{candidate['layer']} N{candidate['neuron']} · discovery n={n} prompts per dose\n"
           "Next-token logits at final chat-wrapper token; 95% prompt-bootstrap intervals")
    for negative in (True, False):
        selected = [(dose, row) for dose, row in doses if (dose < 0) == negative]
        metrics = [row["metrics"]["kl_base_to_changed"] for _, row in selected]
        intervals(ax, [dose for dose, _ in selected], [m["mean"] for m in metrics],
                  [m["ci_low"] for m in metrics], [m["ci_high"] for m in metrics], BLUE,
                  label="Candidate additive steering" if negative else None)
    if control:
        m = control["metrics"]["kl_base_to_changed"]
        intervals(ax, [1.], [m["mean"]], [m["ci_low"]], [m["ci_high"]], GOLD,
                  marker="D", fill=False, connect=False, label="Projected direction control (+1SD only)")
    if scale == "symlog":
        ax.set_yscale(scale, linthresh=min(positive)/10)
    else:
        ax.set_yscale(scale)
    ax.set(xlabel="Additive strength / discovery token activation SD",
           ylabel="Mean KL(baseline || intervention), nats", xlim=(-2.25, 2.25))
    ax.set_xticks([-2, -1, -.5, 0, .5, 1, 2])
    ax.axvline(0, color=MUTED, lw=.7, alpha=.6)
    ax.grid(axis="y", which="major")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.01, 1.025), fontsize=10)
    footer(fig, f"Baseline KL=0 omitted from {scale} axis; control seeds averaged within prompts. KL measures sensitivity, not semantic function.",
           f"Run {summary['run_id']} · summary SHA256 {source['source_sha256'][:12]} · BF16")
    metadata = {"candidate_id": candidate["id"], "layer": candidate["layer"], "neuron": candidate["neuron"],
        "sources": [source], "n_prompts_per_dose": n, "strength_sigma": observed_doses,
        "scale": scale, "baseline_kl": 0., "baseline_displayed": False,
        "control": "random_direction_projected at +1SD only" if control else "NOT_EVALUATED",
        "interval": summary["confidence_interval"], "plotted_rows": [row for _, row in doses] + ([control] if control else []),
        "caption": f"{candidate['id']} additive dose sweep on {n} discovery prompts; observed mean next-token KL and 95% prompt-bootstrap intervals, projected-direction control measured only at +1SD. Baseline KL=0 omitted; {scale} y-axis. Intervention is at the final chat-wrapper token."}
    return fig, metadata


def check_correlations(summary: dict, candidates: dict) -> dict:
    correlations = {}
    for row in summary.get("correlation", []):
        cid = row.get("candidate_id")
        if cid not in candidates or cid in correlations:
            raise ValueError("Unknown or duplicate held-out correlation candidate")
        observations = [o for o in summary.get("activation_observations", [])
                        if o["candidate_id"] == cid and o.get("role", "corpus") == "corpus"]
        if len({o["prompt_id"] for o in observations}) != len(observations):
            raise ValueError("Duplicate held-out corpus observation")
        category = candidates[cid]["target_category"]
        positive = [finite(o["prompt_mean"], "positive prompt mean") for o in observations if o["category"] == category]
        negative = [finite(o["prompt_mean"], "negative prompt mean") for o in observations if o["category"] != category]
        if len(positive) != count(row.get("n_positive"), "AUC positive count") or len(negative) != count(row.get("n_negative"), "AUC negative count"):
            raise ValueError("Held-out AUC counts differ from the underlying corpus observations")
        pairs = np.asarray(positive)[:, None] - np.asarray(negative)[None, :]
        independent_auc = float(((pairs > 0) + .5 * (pairs == 0)).mean())
        if not math.isclose(finite(row.get("auc"), "AUC", 0, 1), independent_auc, abs_tol=1e-12):
            raise ValueError("AUC does not reproduce from independent pairwise comparisons")
        correlations[cid] = {**row, "pairwise_auc_verified": True}
    return correlations


def auc_plot(measured: dict, sources: dict, candidates: dict):
    stages = [stage for stage in ("validation", "expanded") if measured.get(stage)]
    if not stages:
        return None, {"status": "SKIPPED", "reason": "No successful held-out validation/test stage"}
    checked = {stage: check_correlations(measured[stage], candidates) for stage in stages}
    ids = [cid for cid in candidates if any(cid in checked[stage] for stage in stages)]
    if not ids:
        return None, {"status": "SKIPPED", "reason": "No measured held-out AUC rows"}
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    fig.subplots_adjust(left=.10, right=.965, bottom=.25, top=.77)
    counts = sorted({(r["n_positive"], r["n_negative"]) for rows in checked.values() for r in rows.values()})
    count_label = "; ".join(f"n={p} positive / {n} negative" for p, n in counts)
    stage_label = "validation and untouched test" if len(stages) == 2 else "validation only; test not evaluated" if stages[0] == "validation" else "untouched test only; validation unavailable"
    header(fig, "Held-out activation selectivity", f"{len(ids)} frozen hypotheses · {stage_label}\n{count_label} per candidate / split; corpus prompts only")
    xbase = {cid: i for i, cid in enumerate(ids)}
    for stage in stages:
        offset = -.08 if stage == "validation" else .08
        color, marker = (BLUE, "o") if stage == "validation" else (GOLD, "s")
        rows = checked[stage]
        x = [xbase[cid] + offset for cid in ids if cid in rows]
        y = [rows[cid]["auc"] for cid in ids if cid in rows]
        ax.scatter(x, y, color=color, marker=marker, s=75, edgecolors="white", linewidths=.7,
                   label="Validation" if stage == "validation" else "Untouched test", zorder=3)
        for xx, yy in zip(x, y):
            ax.annotate(f"{yy:.3f}", (xx, yy), xytext=(-18, -17) if stage == "validation" else (4, 8),
                        textcoords="offset points", fontsize=10, color=color)
    ax.axhline(.5, color=MUTED, lw=1., linestyle="--", label="Chance rank (AUC=0.5)")
    ax.set_xticks(range(len(ids)), [f"{cid}\nL{candidates[cid]['layer']} N{candidates[cid]['neuron']}" for cid in ids])
    ax.set(ylabel="Target-vs-rest prompt-mean AUC", ylim=(0, 1.08), xlim=(-.45, len(ids)-.45))
    ax.set_yticks([0, .25, .5, .75, 1.])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.01, 1.01), ncols=3, fontsize=10)
    footer(fig, "Small positive samples; no AUC interval was measured. Activation separation is correlational and may reflect surface words.",
           " · ".join(f"{sources[stage]['run_id']} [{sources[stage]['source_sha256'][:10]}]" for stage in stages))
    return fig, {"sources": [sources[stage] for stage in stages], "candidate_ids": ids,
        "auc_rows": checked, "missing_stages": [stage for stage in ("validation", "expanded") if stage not in stages],
        "caption": f"Target-vs-rest prompt-mean activation AUC for {len(ids)} discovery-frozen hypotheses, showing only measured {stage_label}. {count_label} per candidate/split. No confidence interval is claimed; AUC was independently recomputed from pairwise prompt comparisons."}


def counterexample_plot(validation: dict | None, source: dict, candidates: dict, hypotheses_sha: str):
    if validation is None:
        return None, {"status": "SKIPPED", "reason": "No successful measured validation stage"}
    case_path = ROOT / "configs/validation_cases.json"
    if not case_path.is_file():
        return None, {"status": "SKIPPED", "reason": "Counterexample source cases are unavailable"}
    case_cfg = json.loads(case_path.read_text())
    if case_cfg.get("hypotheses_sha256") != hypotheses_sha:
        raise ValueError("Counterexample source cases do not match the frozen hypotheses")
    metadata_path = ROOT / "runs" / validation["run_id"] / "metadata.json"
    if metadata_path.is_file():
        recorded = json.loads(metadata_path.read_text()).get("source_sha256", {}).get("configs/validation_cases.json")
        if recorded is not None and recorded != digest(case_path):
            raise ValueError("Counterexample case source changed after the measured validation run")
    cases = {case["id"]: case for case in case_cfg["prompts"]}
    if len(cases) != len(case_cfg["prompts"]):
        raise ValueError("Duplicate counterexample source prompt")
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for observation in validation.get("activation_observations", []):
        case = cases.get(observation["prompt_id"])
        # A candidate's hook may observe another candidate's case; exclude it.
        if case is None or case.get("candidate_id") != observation.get("candidate_id"):
            continue
        cid, role = case["candidate_id"], case["role"]
        if cid not in candidates or role not in ROLES or observation.get("role") != role or observation.get("split") != "validation":
            raise ValueError("Counterexample observation/source role or partition mismatch")
        key = cid, case["id"]
        if key in seen:
            raise ValueError("Duplicate matching counterexample observation")
        seen.add(key)
        groups[cid][role].append({"prompt_id": case["id"], "source_id": case["source_id"],
            "role": role, "prompt_mean": finite(observation["prompt_mean"], "counterexample prompt mean")})
    ids = [cid for cid in candidates if cid in groups]
    if not ids:
        return None, {"status": "SKIPPED", "reason": "No measured candidate-matched counterexample observations"}
    for cid in ids:
        if any(len(groups[cid][role]) != 2 for role in ROLES):
            raise ValueError(f"{cid}: expected exactly two measured cases in each frozen counterexample role")
    cols = min(2, len(ids))
    rows = math.ceil(len(ids)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12.4, 3.4 + 2.75*rows), squeeze=False)
    fig.subplots_adjust(left=.085, right=.97, bottom=.22, top=.76, hspace=.70, wspace=.26)
    header(fig, "Counterexample activation checks", f"{len(ids)} frozen candidates · two measured prompts per role / candidate\n"
           "Mean activation over user-content tokens; independent activation scales in each panel")
    for ax, cid in zip(axes.flat, ids):
        source_points = {}
        for j, role in enumerate(ROLES):
            points = sorted(groups[cid][role], key=lambda row: row["prompt_id"])
            for k, point in enumerate(points):
                xx, yy = j + (-.07 if k == 0 else .07), point["prompt_mean"]
                ax.scatter(xx, yy, color=BLUE, s=46, zorder=3, marker="o" if k == 0 else "s")
                source_points.setdefault(point["source_id"], []).append((role, xx, yy))
        for family in source_points.values():
            related = [point for point in family if point[0] in {"positive_control", "paraphrase"}]
            if len(related) == 2:
                ax.plot([point[1] for point in related], [point[2] for point in related],
                        color=MUTED, lw=.8, linestyle=":", alpha=.65, zorder=1)
        ax.axhline(0, color=MUTED, lw=.7, alpha=.6)
        ax.set_xticks(range(len(ROLES)), ROLE_LABELS)
        ax.tick_params(axis="x", labelsize=9)
        ax.set(xlim=(-.4, len(ROLES)-.6), ylabel="Mean user-token activation")
        ax.set_title(f"{cid} · L{candidates[cid]['layer']} N{candidates[cid]['neuron']}", loc="left", pad=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", alpha=.65)
        ax.set_axisbelow(True)
        ax.margins(y=.18)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    footer(fig, "Dots show individual prompts. Dotted lines join original/paraphrase families; these pairs are not independent replicates.",
           f"Run {validation['run_id']} · summary SHA256 {source['source_sha256'][:12]} · case source SHA256 {digest(case_path)[:12]}")
    total = sum(len(points) for roles in groups.values() for points in roles.values())
    return fig, {"sources": [source], "case_source": case_path.relative_to(ROOT).as_posix(),
        "case_source_sha256": digest(case_path), "candidate_ids": ids, "observations": dict(groups),
        "n_measured_matched_prompts": total, "n_original_source_families": len({point["source_id"] for roles in groups.values() for points in roles.values() for point in points}),
        "matching": "Join on prompt_id AND matching candidate_id AND role; other candidates' hooks on the same case are excluded",
        "caption": f"User-content prompt-mean activation for {total} measured candidate-matched counterexample cases: two prompts per role for each of {len(ids)} hypotheses. Panel-specific activation scales; dotted lines indicate related original/paraphrase pairs, which are not independent replicates."}


def save_chart(fig, name: str, info: dict) -> dict:
    output, preview = ROOT / "figures", ROOT / "runs/figure_preview"
    output.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    svg, png = output / (name + ".svg"), preview / (name + ".png")
    svg_temp, png_temp = svg.with_suffix(".svg.tmp"), png.with_suffix(".png.tmp")
    description = json.dumps(info, ensure_ascii=False, sort_keys=True)
    fig.savefig(svg_temp, format="svg", metadata={"Date": None, "Description": description})
    normalized = "\n".join(line.rstrip() for line in svg_temp.read_text().splitlines()) + "\n"
    svg_temp.write_text(normalized)
    fig.savefig(png_temp, format="png", dpi=140, metadata={"Description": description})
    svg_temp.replace(svg)
    png_temp.replace(png)
    plt.close(fig)
    return {"status": "GENERATED", **info, "svg": svg.relative_to(ROOT).as_posix(),
            "preview_png": png.relative_to(ROOT).as_posix(), "svg_sha256": digest(svg),
            "rendered_visual_qa": "PENDING: inspect the PNG preview before publication"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Check measured source identities and denominators without drawing")
    args = parser.parse_args()
    hypotheses_path = ROOT / "configs/hypotheses.json"
    if not hypotheses_path.is_file():
        print("No frozen hypotheses; steering figures skipped. No placeholder data generated.")
        return 0
    hypotheses = json.loads(hypotheses_path.read_text())
    if not re.fullmatch(r"[a-f0-9]{40}", hypotheses.get("model_revision", "")):
        raise ValueError("Missing pinned model revision in hypotheses")
    candidates = {candidate["id"]: candidate for candidate in hypotheses["candidates"]}
    if len(candidates) != len(hypotheses["candidates"]):
        raise ValueError("Duplicate frozen candidate IDs")
    hypotheses_sha = digest(hypotheses_path)
    measured, sources = {}, {}
    for stage in ("steering", "validation", "expanded"):
        measured[stage], sources[stage] = load_measured(stage, hypotheses, hypotheses_sha)
    if not any(measured.values()):
        print("No successful measured steering/held-out stages; all figures skipped.")
        return 0
    if args.validate_only:
        for stage in ("validation", "expanded"):
            if measured[stage]:
                check_correlations(measured[stage], candidates)
        print(json.dumps({"status": "PASS", "sources": sources}, ensure_ascii=False))
        return 0
    theme()
    charts = {}
    for cid, candidate in candidates.items():
        if candidate.get("group") != "primary":
            continue
        name = "steering_dose_" + cid
        if measured["steering"]:
            fig, info = dose_plot(measured["steering"], candidate, sources["steering"])
            charts[name] = save_chart(fig, name, info) if fig is not None else info
        else:
            charts[name] = {"status": "SKIPPED", "reason": "No successful discovery steering stage"}
    for name, make in (("heldout_auc", lambda: auc_plot(measured, sources, candidates)),
                       ("counterexample_activation", lambda: counterexample_plot(measured["validation"], sources["validation"], candidates, hypotheses_sha))):
        fig, info = make()
        charts[name] = save_chart(fig, name, info) if fig is not None else info
    metadata = {"hypotheses_source": "configs/hypotheses.json", "hypotheses_sha256": hypotheses_sha,
                "model_revision": hypotheses["model_revision"], "sources": sources, "charts": charts,
                "generated_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "validation": "Successful measured stages only; raw-file hashes and prompt-weighted KL means checked when raw files exist; held-out AUC independently checked by prompt pairs; candidate-specific case joins enforced",
                "rendered_visual_qa": "PENDING: inspect runs/figure_preview/*.png before publication"}
    output = ROOT / "figures/steering_metadata.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"generated": [name for name, row in charts.items() if row["status"] == "GENERATED"],
                      "skipped": {name: row["reason"] for name, row in charts.items() if row["status"] == "SKIPPED"},
                      "metadata": output.relative_to(ROOT).as_posix()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"Steering figure generation rejected inconsistent measured evidence: {exc}", file=sys.stderr)
        sys.exit(1)
