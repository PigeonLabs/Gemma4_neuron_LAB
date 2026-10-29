"""Download only the pinned model's public inference files, within budget."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi, snapshot_download
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
    api = HfApi()
    info = api.model_info(cfg["model_id"], revision=cfg["revision"], files_metadata=True)
    if info.sha != cfg["revision"]:
        raise RuntimeError("Resolved revision differs from pinned revision")
    names = [f.rfilename for f in info.siblings if f.rfilename.endswith((".json", ".jinja", ".model", ".safetensors"))]
    total = sum(f.size or 0 for f in info.siblings if f.rfilename in names)
    free = shutil.disk_usage(ROOT).free
    if total > cfg["max_download_gb"] * 1e9:
        raise RuntimeError(f"Download {total} bytes exceeds model budget")
    if free - total < cfg["min_free_disk_gb"] * 1e9:
        raise RuntimeError("Insufficient free project disk space")
    audit = {"model_id": cfg["model_id"], "revision": info.sha, "gated": info.gated,
             "private": info.private, "download_bytes": total, "files": names,
             "status": "DOWNLOADING", "started_unix": time.time()}
    target = ROOT / "logs/summary/model_download.json"
    target.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"revision": info.sha, "bytes": total, "file_count": len(names)}), flush=True)
    snapshot = snapshot_download(cfg["model_id"], revision=info.sha,
                                 cache_dir=ROOT / cfg["cache_dir"], allow_patterns=names,
                                 max_workers=4)
    audit.update(status="PASS", elapsed_seconds=time.time() - audit["started_unix"])
    target.write_text(json.dumps(audit, indent=2) + "\n")
    (ROOT / "runs/model_path.txt").write_text(snapshot + "\n")
    print(json.dumps({"status": "PASS", "elapsed_seconds": audit["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
