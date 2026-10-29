"""Pinned native HF model adapter and bounded local run records."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def source_hashes() -> dict:
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/*"), *ROOT.glob("scripts/*.py")])
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def environment() -> dict:
    names = ["torch", "transformers", "accelerate", "huggingface_hub", "numpy", "pyarrow", "matplotlib"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in names},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_build": torch.version.cuda,
            "disk_free_bytes": shutil.disk_usage(ROOT).free}


class Run:
    def __init__(self, stage: str, config: dict):
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        self.id = now.strftime("%Y%m%dT%H%M%S") + "_" + stage
        self.path = ROOT / "runs" / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.config = config
        self.resource_limits = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
        self.minimum_gpu_free = torch.cuda.mem_get_info()[0]
        self.minimum_disk_free = shutil.disk_usage(ROOT).free
        self.meta = {"run_id": self.id, "stage": stage, "started_kst": now.isoformat(),
                     "status": "RUNNING", "config": config, "source_sha256": source_hashes(),
                     "environment": environment()}
        torch.cuda.reset_peak_memory_stats()
        self.db = sqlite3.connect(ROOT / "runs/lab.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, stage TEXT, status TEXT, summary TEXT)")
        self.db.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (self.id, stage, "RUNNING", "{}"))
        self.db.commit()
        save_json(self.path / "metadata.json", self.meta)

    def check_budget(self) -> None:
        if time.monotonic() - self.started > self.config.get("max_seconds", 1800):
            raise TimeoutError("Stage time budget exceeded")
        disk_free = shutil.disk_usage(ROOT).free
        gpu_free = torch.cuda.mem_get_info()[0]
        self.minimum_gpu_free = min(self.minimum_gpu_free, gpu_free)
        self.minimum_disk_free = min(self.minimum_disk_free, disk_free)
        if disk_free < self.resource_limits["min_free_disk_gb"] * 1e9:
            raise RuntimeError("Project disk free space below configured floor")
        if torch.cuda.max_memory_reserved() > self.resource_limits["max_gpu_memory_gb"] * 1e9:
            raise RuntimeError("GPU stage memory exceeded configured cap")
        if gpu_free < 4e9:
            raise RuntimeError("Less than 4GB GPU headroom, possibly due to another job; stop without terminating it")
        cap = self.config.get("max_run_disk_gb", 5) * 1e9
        if sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file()) > cap:
            raise RuntimeError("Run artifact disk budget exceeded")

    def finish(self, summary: dict, status="PASS") -> dict:
        torch.cuda.synchronize()
        self.meta.update(status=status, elapsed_seconds=time.monotonic() - self.started,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         ended_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                         minimum_gpu_free_bytes=self.minimum_gpu_free, minimum_disk_free_bytes=self.minimum_disk_free)
        summary = {**summary, **{k: self.meta[k] for k in ["run_id", "stage", "status", "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "minimum_gpu_free_bytes", "minimum_disk_free_bytes"]}}
        save_json(self.path / "metadata.json", self.meta)
        save_json(self.path / "summary.json", summary)
        save_json(ROOT / "logs/summary" / (self.meta["stage"] + ".json"), summary)
        self.db.execute("UPDATE runs SET status=?, summary=? WHERE id=?", (status, json.dumps(summary), self.id))
        self.db.commit()
        self.db.close()
        return summary


class ModelRuntime:
    def __init__(self, audit_path: Path | None = None):
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        self.lock_file = (ROOT / "runs/gpu.lock").open("a+")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A BF16-capable CUDA GPU is required; CPU/quantized fallback is not an experiment")
        free, total = torch.cuda.mem_get_info()
        if free < 34e9:
            raise RuntimeError("Less than 34GB free GPU memory; existing jobs remain untouched")
        seed = self.cfg["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(6)
        local = Path((ROOT / "runs/model_path.txt").read_text().strip())
        if local.name != self.cfg["revision"]:
            raise RuntimeError("Local cache path does not match pinned revision")
        started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            local, dtype=torch.bfloat16, device_map={"": self.cfg["device"]},
            attn_implementation=self.cfg["attn_implementation"],
            local_files_only=True, trust_remote_code=False,
        ).eval().requires_grad_(False)
        self.modules = {}
        self.module_names = {}
        for name, module in self.model.named_modules():
            match = re.search(r"\.layers\.(\d+)\.mlp\.down_proj$", name)
            if match:
                layer = int(match.group(1))
                self.modules[layer] = module
                self.module_names[layer] = name
        tc = self.model.config.text_config
        expected = list(range(tc.num_hidden_layers))
        if sorted(self.modules) != expected:
            raise RuntimeError("Runtime modules do not match config layers")
        if any(m.in_features != tc.intermediate_size for m in self.modules.values()):
            raise RuntimeError("Runtime intermediate size differs from config")
        dtypes = sorted({str(p.dtype) for p in self.model.parameters()})
        if dtypes != ["torch.bfloat16"]:
            raise RuntimeError(f"Unexpected parameter dtypes: {dtypes}")
        audit = {"status": "PASS", "model_id": self.cfg["model_id"], "revision": self.cfg["revision"],
                 "runtime_class": type(self.model).__name__, "layers": len(self.modules),
                 "intermediate_size": tc.intermediate_size, "neuron_count": sum(m.in_features for m in self.modules.values()),
                 "parameter_count": sum(p.numel() for p in self.model.parameters()),
                 "parameter_dtypes": dtypes, "hook_modules": self.module_names,
                 "chat_template_sha256": hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(),
                 "load_seconds": time.monotonic() - started, "allocated_bytes": torch.cuda.memory_allocated(),
                 "gpu_free_bytes_before": free, "gpu_total_bytes": total, "environment": environment(),
                 "source_sha256": source_hashes(), "config": self.cfg}
        save_json(audit_path or ROOT / "logs/summary/model_load.json", audit)
        save_json(ROOT / "runs/runtime_config.json", self.model.config.to_dict())
        self.smoke_passed = False
        self.initial_source_hashes = source_hashes()
        print(json.dumps({"event": "model_loaded", "layers": len(self.modules), "neuron_count": audit["neuron_count"], "seconds": audit["load_seconds"]}), flush=True)

    def verify_sources(self):
        current = source_hashes()
        changed = [p for p,h in self.initial_source_hashes.items() if p.startswith("src/") and current.get(p) != h]
        if changed:
            raise RuntimeError("Worker source changed on disk; restart before running: " + ", ".join(changed))

    def inputs(self, text: str) -> dict:
        messages = [{"role": "user", "content": text}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                    return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
        if inputs["input_ids"].shape[-1] > self.cfg["max_input_tokens"]:
            raise ValueError("Prompt exceeds input token budget; no silent truncation")
        return {k: v.to(self.cfg["device"]) for k, v in inputs.items()}

    @torch.inference_mode()
    def logits(self, inputs: dict) -> torch.Tensor:
        return self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()

    @torch.inference_mode()
    def generate(self, inputs: dict, max_new_tokens: int = 48) -> dict:
        length = inputs["input_ids"].shape[1]
        tokens = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                     use_cache=True, temperature=None, top_p=None, top_k=None)
        ids = tokens[0, length:].cpu().tolist()
        return {"token_ids": ids, "text": self.tokenizer.decode(ids, skip_special_tokens=True)}


def gpu_inventory() -> list[str]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    return result.stdout.strip().splitlines() if result.returncode == 0 else [result.stderr.strip()]
