"""Step 7 check: is every cell of the experiment complete?"""
import json
from collections import Counter

from src.config import RESULTS_LOG

with open(RESULTS_LOG, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

latest_ok = {}
for r in rows:
    if not r["error"]:
        latest_ok[(r["task_id"], r["setting"], r["level"], r["mode"])] = r

cells = Counter((r["mode"], r["level"], r["setting"], r["complexity"]) for r in latest_ok.values())
print("Completed runs per cell (expected 50 each):")
for cell in sorted(cells):
    print(f"  {' | '.join(cell):55} {cells[cell]}")

replays = [r for r in latest_ok.values() if r["mode"] == "replay"]
print(f"\nTotal completed: {len(latest_ok)} (expected 600)")
print(f"Error rows in log: {sum(1 for r in rows if r['error'])}")
print(f"Replays diverged: {sum(r['replay_diverged'] for r in replays)} of {len(replays)}")