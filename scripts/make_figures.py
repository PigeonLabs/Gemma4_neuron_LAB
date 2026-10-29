#!/usr/bin/env python3
"""Regenerate README research figures from successful measured summary files.

Usage: .venv/bin/python scripts/make_figures.py [--input logs/summary/survey.json]

No fixture data, interpolation, inferred missing cells, or placeholder plots are
generated. The public summary is sufficient to reproduce the plots. Available
local Parquet inputs are additionally checked against recorded SHA256 hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs/matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2563a6"
GOLD = "#bd8b26"
INK = "#202a35"
MUTED = "#566372"
GRID = "#e2e7ed"
LABELS = {
    "general_knowledge": "Knowledge", "math_logic": "Math / logic",
    "programming": "Code", "science": "Science",
    "language_translation": "Translation", "literature": "Literature",
    "social_interaction": "Social", "persona": "Persona",
    "emotion_tone": "Tone", "uncertainty": "Uncertainty",
    "benign_refusal": "Benign requests", "reasoning": "Reasoning",
    "long_context": "Retrieval", "format_compliance": "Format",
    "irrelevant_control": "Control",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite(value, name: str, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric observation")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its valid range")
    return float(value)


def choose_source(explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Measured summary does not exist: {explicit}")
        return explicit
    for stage in ("survey", "pilot"):
        candidate = ROOT / f"logs/summary/{stage}.json"
        if candidate.is_file() and json.loads(candidate.read_text()).get("status") == "PASS":
            return candidate
    return None


def validate_summary(path: Path) -> tuple[dict, dict]:
    """Check plotted values and denominators before any output is replaced."""
    summary = json.loads(path.read_text())
    if summary.get("status") != "PASS" or summary.get("stage") not in {"pilot", "survey"}:
        raise ValueError("Only successful measured pilot/survey summaries may be plotted")
    if summary.get("split") != "discovery":
        raise ValueError("Activation figures require the explicitly recorded discovery split")
    run_id = summary.get("run_id", "")
    if not re.fullmatch(r"\d{8}T\d{6}_(?:pilot|survey)", run_id):
        raise ValueError("Missing or invalid measured run identity")
    for key in ("model_revision", "corpus_sha256"):
        if not re.fullmatch(r"[a-f0-9]{40}" if key == "model_revision" else r"[a-f0-9]{64}", summary.get(key, "")):
            raise ValueError(f"Missing or invalid {key}")
    if not isinstance(summary.get("token_scope"), str) or "user-content" not in summary["token_scope"]:
        raise ValueError("Token scope must explicitly identify measured user-content tokens")
    layers = summary.get("layer_statistics")
    if not isinstance(layers, list) or not layers:
        raise ValueError("No measured layer rows")
    if positive_int(summary.get("layers"), "layers") != len(layers):
        raise ValueError("Layer denominator differs from observed rows")
    prompts = positive_int(summary.get("prompts"), "prompts")
    tokens = positive_int(summary.get("tokens_per_layer"), "tokens_per_layer")
    category_counts = summary.get("category_prompt_counts", {})
    if not category_counts or sum(positive_int(v, f"category count {k}") for k, v in category_counts.items()) != prompts:
        raise ValueError("Category prompt counts do not reconcile with the prompt denominator")
    if positive_int(summary.get("categories"), "categories") != len(category_counts):
        raise ValueError("Category denominator differs from observed categories")
    layer_map = {}
    for row in layers:
        layer = row.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 or layer in layer_map:
            raise ValueError("Layer IDs must be distinct nonnegative integers")
        layer_map[layer] = row
        positive_int(row.get("neurons"), f"layer {layer} neurons")
        if row.get("token_count") != tokens or row.get("prompt_count") != prompts:
            raise ValueError("Layer rows use different token or prompt denominators")
        p10, p50, p90 = [finite(row.get(k), k, 0) for k in ("std_p10", "std_p50", "std_p90")]
        if not p10 <= p50 <= p90:
            raise ValueError("Layer SD percentiles are unordered")
        finite(row.get("mean_abs_activation"), "mean_abs_activation", 0)
        finite(row.get("frequency_mean"), "frequency_mean", 0, 1)
        sample_n = positive_int(row.get("quantile_sample_count"), "quantile_sample_count")
        if sample_n > tokens:
            raise ValueError("Quantile reservoir exceeds the observed token population")
        if set(row.get("category_deviation", {})) != set(category_counts):
            raise ValueError("Heatmap has a missing or extra category; no zero filling is allowed")
        for cat, value in row["category_deviation"].items():
            finite(value, f"category_deviation {cat}", 0)
    if sum(row["neurons"] for row in layers) != summary.get("neuron_count"):
        raise ValueError("Neuron denominator does not reconcile with layer rows")
    seen_candidates = set()
    for row in summary.get("top_candidates", []):
        coord = (row.get("layer"), row.get("neuron"))
        if coord in seen_candidates or coord[0] not in layer_map:
            raise ValueError("Candidate has a duplicate or unmeasured coordinate")
        seen_candidates.add(coord)
        if isinstance(coord[1], bool) or not isinstance(coord[1], int) or not 0 <= coord[1] < layer_map[coord[0]]["neurons"]:
            raise ValueError("Candidate neuron lies outside the measured layer")
        if row.get("model_revision") != summary["model_revision"] or row.get("run_id") != run_id:
            raise ValueError("Candidate provenance differs from its activation survey")
        if row.get("category") not in category_counts or set(row.get("category_means", {})) != set(category_counts):
            raise ValueError("Candidate categories differ from the measured population")
        for key in ("selectivity_z", "selectivity_gap", "std", "cross_prompt_std"):
            finite(row.get(key), key, 0)
        for key in ("mean", "min", "max"):
            finite(row.get(key), key)
        samples = row.get("sampled_activations")
        if not isinstance(samples, list) or len(samples) != row.get("quantile_sample_count"):
            raise ValueError("Candidate histogram values differ from their reservoir denominator")
        if len(samples) != layer_map[coord[0]]["quantile_sample_count"]:
            raise ValueError("Candidate and layer quantile denominators differ")
        for sample in samples:
            finite(sample, "sampled_activation", row["min"] - 1e-6, row["max"] + 1e-6)
        qs = [finite(row.get("quantiles", {}).get(k), k) for k in ("q05", "q50", "q95")]
        if not row["min"] <= qs[0] <= qs[1] <= qs[2] <= row["max"]:
            raise ValueError("Candidate quantiles are unordered or outside the observed range")
        if not np.allclose(np.quantile(samples, [.05, .5, .95]), qs, rtol=2e-5, atol=2e-5):
            raise ValueError("Candidate quantiles do not reproduce from the published reservoir")
        cat_means = np.asarray([finite(row["category_means"][c], "candidate category mean") for c in sorted(category_counts)])
        gap = float(cat_means.max() - (cat_means.sum() - cat_means.max()) / (len(cat_means) - 1)) if len(cat_means) > 1 else 0.
        if not math.isclose(gap, row["selectivity_gap"], rel_tol=2e-5, abs_tol=2e-5):
            raise ValueError("Candidate selectivity gap does not reconcile with published category means")
        z = gap / max(row["cross_prompt_std"], 1e-8)
        if not math.isclose(z, row["selectivity_z"], rel_tol=3e-5, abs_tol=3e-5):
            raise ValueError("Candidate descriptive standardized gap is inconsistent")
    # Public summary regeneration remains possible without distributing raw data.
    local_checks = {"verified": [], "unavailable": []}
    for relative, expected in summary.get("artifact_hashes", {}).items():
        local = ROOT / "runs" / run_id / relative
        if ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError("Unsafe artifact path in measurement provenance")
        if not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("Invalid raw artifact digest")
        if local.is_file():
            if digest(local) != expected:
                raise ValueError(f"Local raw measurement artifact hash mismatch: {relative}")
            local_checks["verified"].append(relative)
        else:
            local_checks["unavailable"].append(relative)
    return summary, local_checks


def theme() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .8, "axes.titlesize": 11,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "grid.linewidth": .7, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none", "svg.hashsalt": "gemma4-neuron-lab",
    })


def header(fig, title: str, subtitle: str) -> None:
    fig.text(.065, .96, title, fontsize=18, fontweight="bold", va="top")
    fig.text(.065, .905, subtitle, fontsize=10, color=MUTED, va="top")
    fig.text(.975, .96, "✽", ha="right", va="top", fontsize=17, color=GOLD)


def scope_line(s: dict) -> str:
    return f"Discovery: {s['prompts']:,} prompts · {s['categories']} categories · {s['tokens_per_layer']:,} user tokens/layer"


def footer(fig, s: dict, source_digest: str, note: str) -> None:
    fig.text(.065, .075, note, fontsize=9, color=MUTED, va="top")
    fig.text(.065, .034, f"Run {s['run_id']} · summary SHA256 {source_digest[:12]} · BF16",
             fontsize=8, color=MUTED, va="top")


def layer_variability(s: dict, source_digest: str):
    rows = sorted(s["layer_statistics"], key=lambda r: r["layer"])
    x = [r["layer"] for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(left=.10, right=.96, bottom=.22, top=.79)
    header(fig, "Layer-wise activation variability", scope_line(s))
    ax.fill_between(x, [r["std_p10"] for r in rows], [r["std_p90"] for r in rows],
                    color=BLUE, alpha=.16, label="10th–90th percentiles across neurons")
    ax.plot(x, [r["std_p50"] for r in rows], color=BLUE, marker="o", ms=3.5,
            linewidth=1.6, label="Median across neurons")
    ax.set(xlabel="Transformer layer (zero-based)", ylabel="Token activation SD", ylim=(0, None))
    ax.set_xticks(sorted(set([x[0], *x[::max(1, len(x)//8)], x[-1]])))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    footer(fig, s, source_digest, "Population SD over user tokens; shaded neuron spread is not a confidence interval.")
    return fig, "Per-layer median and 10th–90th percentiles of per-neuron population SD over measured user-content tokens."


def category_heatmap(s: dict, source_digest: str):
    rows = sorted(s["layer_statistics"], key=lambda r: r["layer"])
    categories = list(s["category_prompt_counts"])
    matrix = np.asarray([[r["category_deviation"][c] for c in categories] for r in rows])
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    fig.subplots_adjust(left=.09, right=.90, bottom=.26, top=.82)
    header(fig, "Category activation patterns by layer", scope_line(s))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", origin="upper", vmin=0, cmap="cividis")
    ax.set_xticks(range(len(categories)), [LABELS.get(c, c.replace("_", " ")) for c in categories], rotation=45, ha="right")
    shown = sorted(set([0, *range(0, len(rows), max(1, len(rows)//12)), len(rows)-1]))
    ax.set_yticks(shown, [str(rows[i]["layer"]) for i in shown])
    ax.set_ylabel("Transformer layer (zero-based)")
    bar = fig.colorbar(image, ax=ax, fraction=.033, pad=.025)
    bar.set_label("Mean |category deviation| / neuron token SD", labelpad=11)
    footer(fig, s, source_digest,
           "Prompt-balanced category means; absolute deviations averaged across all neurons in each layer.")
    return fig, "Layer × category mean absolute deviation from the prompt-balanced grand mean, divided by each neuron's token SD (floor 1e-6)."


def candidate_scatter(s: dict, source_digest: str):
    candidates = s.get("top_candidates", [])
    if not candidates:
        return None, "Skipped: no measured candidate rows."
    fig, ax = plt.subplots(figsize=(11.5, 6.1))
    fig.subplots_adjust(left=.10, right=.96, bottom=.22, top=.78)
    header(fig, "Discovery candidate selectivity by layer", f"{scope_line(s)}\n{len(candidates)} displayed candidates from layer-wise top-8 pools")
    ax.scatter([r["layer"] for r in candidates], [r["selectivity_z"] for r in candidates],
               color=BLUE, edgecolors="white", linewidths=.6, s=48, alpha=.85)
    layer_ids = sorted(r["layer"] for r in s["layer_statistics"])
    ax.set(xlabel="Transformer layer (zero-based)", ylabel="Category gap / across-prompt SD", ylim=(0, None))
    ax.set_xlim(layer_ids[0] - 1, layer_ids[-1] + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    # Label a small deterministic set; do not manufacture a categorical color key.
    for j, row in enumerate(candidates[:4]):
        ax.annotate(f"L{row['layer']} N{row['neuron']}", (row["layer"], row["selectivity_z"]),
                    xytext=(6, 9 + j * 11), textcoords="offset points", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .6})
    ax.margins(y=.24)
    footer(fig, s, source_digest, "Descriptive discovery ranking after selection; these values are not hypothesis-test z-scores or causal evidence.")
    return fig, "Observed candidate standardized category gaps versus layer; only the recorded selected candidate pool is shown."


def candidate_histograms(s: dict, source_digest: str):
    candidates = s.get("top_candidates", [])[:8]
    if not candidates:
        return None, "Skipped: no measured sampled activations."
    ncols = min(4, len(candidates))
    nrows = math.ceil(len(candidates) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 + 2.3 * nrows), squeeze=False, sharey=True)
    fig.subplots_adjust(left=.065, right=.975, bottom=.20, top=.76, hspace=.57, wspace=.22)
    header(fig, "Sampled activation distributions of leading candidates",
           f"{scope_line(s)}\nUniform hash-priority token reservoir; approximate distributions")
    max_count = max(np.histogram(r["sampled_activations"], bins=12)[0].max() for r in candidates)
    for ax, row in zip(axes.flat, candidates):
        values = np.asarray(row["sampled_activations"])
        ax.hist(values, bins=12, color=BLUE, edgecolor="white", linewidth=.5)
        ax.axvline(np.median(values), color=GOLD, linestyle="--", linewidth=1.5)
        ax.set_title(f"L{row['layer']} · N{row['neuron']} · n={len(values)}\n{LABELS.get(row['category'], row['category'])}", loc="left", pad=8)
        ax.set_xlabel("Activation (panel-specific scale)", fontsize=9)
        ax.set_ylim(0, max_count * 1.12)
        ax.tick_params(axis="both", labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.grid(axis="y", alpha=.5)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Sampled user tokens", fontsize=10)
    for ax in list(axes.flat)[len(candidates):]:
        ax.set_visible(False)
    footer(fig, s, source_digest,
           "Each panel uses its own activation scale; compare shapes within panels. Dashed line: sample median. Not full-data histograms.")
    return fig, "Top-eight selected-neuron histograms from bounded uniform token reservoirs; panel-specific activation axes and common count axis."


def write_figure(fig, name: str, output: Path, preview: Path, provenance: dict, caption: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    svg, png = output / f"{name}.svg", preview / f"{name}.png"
    svg_tmp, png_tmp = svg.with_suffix(".svg.tmp"), png.with_suffix(".png.tmp")
    details = {**provenance, "caption": caption}
    fig.savefig(svg_tmp, format="svg", metadata={"Date": None, "Description": json.dumps(details, ensure_ascii=False)})
    svg_tmp.write_text("\n".join(line.rstrip() for line in svg_tmp.read_text().splitlines()) + "\n")
    fig.savefig(png_tmp, format="png", dpi=140, metadata={"Description": json.dumps(details, ensure_ascii=False)})
    svg_tmp.replace(svg)
    png_tmp.replace(png)
    plt.close(fig)
    return {"svg": str(svg.relative_to(ROOT)), "preview_png": str(png.relative_to(ROOT)),
            "svg_sha256": digest(svg), "caption": caption}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Successful measured pilot/survey summary; default prefers survey")
    parser.add_argument("--validate-only", action="store_true", help="Validate real source values without drawing or changing figures")
    args = parser.parse_args()
    path = choose_source(args.input)
    if path is None:
        print("No successful measured activation summary; figures skipped. No placeholder data generated.")
        return 0
    summary, raw_checks = validate_summary(path)
    source_digest = digest(path)
    provenance = {
        "source": path.resolve().relative_to(ROOT).as_posix(), "source_sha256": source_digest,
        "model_revision": summary["model_revision"], "run_id": summary["run_id"],
        "corpus_sha256": summary["corpus_sha256"], "split": summary["split"],
        "prompts": summary["prompts"], "categories": summary["categories"],
        "category_prompt_counts": summary["category_prompt_counts"],
        "tokens_per_layer": summary["tokens_per_layer"], "layers": summary["layers"],
        "neuron_count": summary["neuron_count"], "token_scope": summary["token_scope"],
        "raw_artifact_checks": raw_checks,
        "validation": "Finite values, source identities, denominators, percentile order, candidate gaps, and sampled quantiles reconciled.",
        "caveats": ["Small authored discovery corpus; selected candidates do not establish function or causality.",
                    "Across-neuron percentile bands are observed spread, not confidence intervals.",
                    "Histograms approximate token distributions with bounded uniform-priority reservoirs."],
    }
    if args.validate_only:
        print(json.dumps({"status": "PASS", **provenance}, ensure_ascii=False))
        return 0
    theme()
    charts = {}
    for name, render in (("layer_activation_variability", layer_variability),
                         ("category_activation_heatmap", category_heatmap),
                         ("candidate_selectivity", candidate_scatter),
                         ("candidate_activation_histograms", candidate_histograms)):
        fig, caption = render(summary, source_digest)
        charts[name] = {"status": "SKIPPED", "reason": caption} if fig is None else {
            "status": "GENERATED", **write_figure(fig, name, ROOT / "figures", ROOT / "runs/figure_preview", provenance, caption)}
    metadata = {**provenance, "generated_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "charts": charts, "rendered_visual_qa": "PENDING: inspect runs/figure_preview PNGs before publication"}
    destination = ROOT / "figures/metadata.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({"source": provenance["source"], "run_id": summary["run_id"],
                      "generated": [name for name, row in charts.items() if row["status"] == "GENERATED"],
                      "metadata": "figures/metadata.json"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"Figure generation rejected invalid or unavailable measured evidence: {exc}", file=sys.stderr)
        sys.exit(1)
