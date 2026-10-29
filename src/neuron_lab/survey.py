"""All-layer streaming observations with explicit user-token scope."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from .corpus import build_corpus, split_corpus, validate_corpus
from .hooks import ActivationHook
from .runtime import ROOT, Run, save_json
from .statistics import LayerStatistics, rank_candidates


def content_mask(runtime, text, inputs):
    rendered = runtime.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
    start = rendered.find(text.strip())
    if start < 0:
        raise ValueError("Cannot locate user text inside rendered template")
    end = start + len(text.strip())
    encoded = runtime.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != inputs["input_ids"][0].cpu().tolist():
        raise ValueError("Tokenizer offsets and runtime chat input differ")
    mask = [b > a and a < end and b > start for a,b in encoded["offset_mapping"]]
    return torch.tensor([mask], dtype=torch.bool, device=inputs["input_ids"].device)


@torch.inference_mode()
def collect(runtime, stage):
    cfg = yaml.safe_load((ROOT / f"configs/{stage}.yaml").read_text())
    run = Run(stage, cfg)
    rows = build_corpus()
    validate_corpus(rows)
    rows = split_corpus(rows, "discovery")
    if stage == "pilot":
        seen = set()
        selected = []
        for row in rows:
            if row["category"] not in seen:
                selected.append(row)
                seen.add(row["category"])
        rows = selected
    rows = rows[:cfg["max_prompts"]]
    layers = sorted(runtime.modules) if cfg["layers"] == "all" else cfg["layers"]
    accumulators = {l: LayerStatistics(l, reservoir_size=cfg["quantile_sample_tokens"],
                                      top_k=cfg["top_k_contexts"], seed=cfg["seed"]) for l in layers}
    corpus_digest = hashlib.sha256(json.dumps(build_corpus(), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    save_json(run.path / "prompts.json", rows)
    try:
        for i, row in enumerate(rows):
            run.check_budget()
            inputs = runtime.inputs(row["text"])
            mask = content_mask(runtime, row["text"], inputs)
            with ExitStack() as stack:
                for layer, accumulator in accumulators.items():
                    stack.enter_context(ActivationHook(runtime.modules[layer], capture=lambda x, a=accumulator: a.update(
                        x, row["id"], row["category"], token_ids=inputs["input_ids"], mask=mask)))
                runtime.logits(inputs)
            if (i + 1) % 5 == 0 or i == 0:
                print(json.dumps({"event": "activation_progress", "stage": stage, "completed": i+1, "total": len(rows)}), flush=True)
                save_json(run.path / "progress.json", {"completed_prompts": i+1, "total_prompts": len(rows)})
        stats_dir = run.path / "statistics"
        stats_dir.mkdir()
        candidates = []
        summaries = []
        source = {r["id"]: r for r in rows}
        for layer, accumulator in accumulators.items():
            run.check_budget()
            finalized = accumulator.finalize()
            columns = finalized["columns"]
            table = pa.Table.from_pydict({"layer": [layer]*finalized["width"], **columns})
            pq.write_table(table, stats_dir / f"layer_{layer:02d}.parquet", compression="zstd")
            arrays = {f"{kind}_{key}": np.asarray(value) for kind in ["top_contexts", "low_contexts"]
                      for key, value in finalized[kind].items()}
            arrays.update({"category_"+key: np.asarray(value, dtype=np.float32) for key,value in finalized["category_means"].items()})
            np.savez_compressed(stats_dir / f"contexts_{layer:02d}.npz", **arrays)
            local_candidates = rank_candidates([finalized], top_n=8)
            for candidate in local_candidates:
                n = candidate["neuron"]
                candidate.update(model_revision=runtime.cfg["revision"], run_id=run.id,
                                 cross_prompt_std=columns["cross_prompt_std"][n],
                                 quantiles={k: columns[k][n] for k in ["q05", "q50", "q95"]},
                                 quantile_sample_count=finalized["quantile_sample_count"],
                                 sampled_activations=accumulator._sample[:,n].cpu().tolist(),
                                 category_means={cat:vals[n] for cat,vals in finalized["category_means"].items()})
                for kind in ["top", "low"]:
                    evidence = finalized[kind+"_contexts"]
                    candidate[kind+"_contexts"] = []
                    for j in range(len(evidence["values"])):
                        prompt_id = finalized["prompt_ids"][evidence["prompt"][j][n]]
                        candidate[kind+"_contexts"].append({"prompt_id": prompt_id,
                            "category": source[prompt_id]["category"], "prompt": source[prompt_id]["text"],
                            "activation": evidence["values"][j][n], "position": evidence["position"][j][n],
                            "token": runtime.tokenizer.decode([evidence["token"][j][n]])})
            candidates.extend(local_candidates)
            std = np.asarray(columns["std"])
            means = np.asarray(columns["mean"])
            pattern = {}
            for cat, values in finalized["category_means"].items():
                deviation = (np.asarray(values) - np.asarray(columns["prompt_mean"])) / np.maximum(std, 1e-6)
                pattern[cat] = float(np.abs(deviation).mean())
            summaries.append({"layer": layer, "neurons": finalized["width"], "token_count": finalized["count"],
                              "prompt_count": finalized["prompt_count"], "std_p10": float(np.quantile(std,.1)),
                              "std_p50": float(np.median(std)), "std_p90": float(np.quantile(std,.9)),
                              "mean_abs_activation": float(np.abs(means).mean()),
                              "frequency_mean": float(np.mean(columns["frequency"])),
                              "category_deviation": pattern, "quantile_sample_count": finalized["quantile_sample_count"]})
            # Release layer GPU state once finalized.
            del accumulators[layer].__dict__["_sample"]
        candidates.sort(key=lambda x: (-x["selectivity_z"],x["layer"],x["neuron"]))
        save_json(run.path / "candidates.json", candidates)
        summary = {"model_revision": runtime.cfg["revision"], "layers": len(layers),
                   "neuron_count": sum(s["neurons"] for s in summaries), "prompts": len(rows),
                   "categories": len({r["category"] for r in rows}), "split": "discovery",
                   "token_scope": "user-content tokens only, offset-verified; no chat wrappers or assistant generation",
                   "tokens_per_layer": summaries[0]["token_count"], "corpus_sha256": corpus_digest,
                   "layer_statistics": summaries, "top_candidates": candidates[:24],
                   "category_prompt_counts": {c: sum(r["category"]==c for r in rows) for c in sorted({r["category"] for r in rows})},
                   "quantile_method": "uniform hash-priority reservoir, 64 user-token positions per layer",
                   "artifact_hashes": {p.relative_to(run.path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stats_dir.glob("*.parquet"))},
                   "resume_policy": "Completed stage artifacts are retained; interrupted within-stage collection restarts from the frozen corpus, never double-counts prompts"}
        if stage == "survey":
            save_json(ROOT / "catalog/public/candidates.json", candidates[:48])
        return run.finish(summary)
    except Exception as exc:
        run.finish({"error": repr(exc), "completed_prompts": len(next(iter(accumulators.values())).prompt_ids)}, "FAIL")
        raise
