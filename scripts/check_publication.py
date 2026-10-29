"""Create an explicit, scanned UTF-8 allowlist for GitHub checkpoint commits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["README.md", "AGENTS.md", "MASTER.md", ".gitignore", "pyproject.toml", "requirements.lock.txt",
            "src/**/*.py", "tests/**/*.py", "scripts/*.py", "scripts/*.sh", "configs/*.yaml", "configs/*.json",
            "docs/*.md", "reports/*.md", "figures/*.svg", "logs/summary/*.json", "catalog/public/*.json",
            "catalog/public/*.md", "references/*audit*.json", "explorer/*.py", "explorer/*.md"]
SECRET = re.compile(r"(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def manifest() -> list[dict]:
    files = sorted({p for pattern in PATTERNS for p in ROOT.glob(pattern) if p.is_file()})
    result = []
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Refusing symbolic link {path.relative_to(ROOT)}")
        raw = path.read_bytes()
        if len(raw) > 1_000_000:
            raise ValueError(f"Oversized public file {path.relative_to(ROOT)}")
        content = raw.decode("utf-8")
        if SECRET.search(content):
            raise ValueError(f"Potential secret in {path.relative_to(ROOT)}")
        result.append({"path": path.relative_to(ROOT).as_posix(), "mode": "100644", "type": "blob",
                       "content": content, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    if sum(x["bytes"] for x in result) > 10_000_000:
        raise ValueError("Checkpoint exceeds 10MB publication limit")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    entries = manifest()
    if args.manifest:
        print(json.dumps(entries))
    else:
        print(json.dumps({"status": "PASS", "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
                          "files": [{k: v for k, v in e.items() if k != "content"} for e in entries]}, indent=2))
