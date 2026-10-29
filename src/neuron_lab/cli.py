"""Single-GPU worker; local jobs keep one pinned model resident."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from .runtime import ROOT, ModelRuntime, save_json


def worker(idle_timeout: int):
    jobs = ROOT / "runs/jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    save_json(ROOT / "runs/worker.json", {"status": "STARTING", "pid": os.getpid()})
    try:
        runtime = ModelRuntime()
    except Exception as exc:
        save_json(ROOT / "runs/worker.json", {"status": "FAIL", "pid": os.getpid(), "error": repr(exc)})
        raise
    save_json(ROOT / "runs/worker.json", {"status": "READY", "pid": os.getpid()})
    last = time.monotonic()
    while time.monotonic() - last < idle_timeout:
        pending = sorted(jobs.glob("*.pending.json"))
        if not pending:
            time.sleep(0.5)
            continue
        path = pending[0]
        active = path.with_name(path.name.replace(".pending.json", ".running.json"))
        path.rename(active)
        job = json.loads(active.read_text())
        save_json(ROOT / "runs/worker.json", {"status": "BUSY", "pid": os.getpid(), "stage": job["stage"]})
        print(json.dumps({"event": "stage_start", "stage": job["stage"]}), flush=True)
        if job["stage"] == "shutdown":
            active.rename(active.with_name(active.name.replace(".running.json", ".done.json")))
            break
        try:
            runtime.verify_sources()
            if job["stage"] == "smoke":
                from .smoke import smoke
                result = smoke(runtime)
            elif job["stage"] in {"pilot", "survey"}:
                if not runtime.smoke_passed:
                    raise RuntimeError("Smoke test has not passed in this worker")
                from .survey import collect
                result = collect(runtime, job["stage"])
            elif job["stage"] in {"steering", "validation", "expanded"}:
                if not runtime.smoke_passed:
                    raise RuntimeError("Smoke test has not passed in this worker")
                from .steering import execute
                result = execute(runtime, job["stage"])
            elif job["stage"] == "live":
                if not runtime.smoke_passed:
                    raise RuntimeError("Smoke test has not passed in this worker")
                from .steering import live
                result = live(runtime, job)
            else:
                raise ValueError("Unknown stage")
            job["result"] = result
            suffix = ".done.json"
        except Exception as exc:
            traceback.print_exc()
            job["error"] = repr(exc)
            suffix = ".failed.json"
        save_json(active, job)
        active.rename(active.with_name(active.name.replace(".running.json", suffix)))
        print(json.dumps({"event": "stage_end", "stage": job["stage"], "status": job.get("result", {}).get("status", "FAIL")}), flush=True)
        last = time.monotonic()
        save_json(ROOT / "runs/worker.json", {"status": "READY", "pid": os.getpid()})
    save_json(ROOT / "runs/worker.json", {"status": "STOPPED", "pid": os.getpid()})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--idle-timeout", type=int, default=3600)
    p = sub.add_parser("submit")
    p.add_argument("stage", choices=["smoke", "pilot", "survey", "steering", "validation", "expanded", "shutdown"])
    p = sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.idle_timeout)
    elif args.command == "submit":
        jobs = ROOT / "runs/jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        path = jobs / f"{time.time_ns()}_{args.stage}.pending.json"
        save_json(path, {"stage": args.stage, "submitted_unix": time.time()})
        print(path)
    elif args.command == "status":
        for path in sorted((ROOT / "runs/jobs").glob("*.json")):
            data = json.loads(path.read_text())
            print(json.dumps({"job": path.name, "stage": data["stage"], "status": data.get("result", {}).get("status"), "error": data.get("error")}))


if __name__ == "__main__":
    main()
