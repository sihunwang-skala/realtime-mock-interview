"""Structured test record (see docs/TEST_RECORD_SPEC.md). Public part never contains text from the real recordings."""
from __future__ import annotations
import json, pathlib, platform, subprocess, time
ROOT = pathlib.Path(__file__).resolve().parents[2]; R = ROOT / "results"

def env():
    try: commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: commit = "?"
    return {"git_commit": commit, "hardware": "Apple M2 Pro, 32 GB", "os": platform.platform(), "python": platform.python_version(), "network": "loopback / local models; cloud only where stated",
            "other_load": "no other benchmark running unless stated"}

def write(test_id: str, public: dict, local: dict | None = None):
    rec = {"test_id": test_id, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env(), **public}
    (R / f"mini_{test_id}.record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    if local is not None: (R / f"mini_{test_id}_REAL_local.json").write_text(json.dumps(local, ensure_ascii=False, indent=1))
    print(f"[record] {test_id} written ({'+local' if local is not None else 'public only'})", flush=True)
