"""Step 8: analysis of the experiment results.

Prints five tables and saves each as a CSV file in analysis/.
"""
import csv
import json
import math
import random
from collections import defaultdict
from statistics import mean

from src.config import PROJECT_ROOT, RESULTS_LOG

OUT_DIR = PROJECT_ROOT / "analysis"
LONG_ANSWER_WORDS = 5          # sensitivity rule, fixed before the full run
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
SETTINGS = ("single_agent", "multi_agent")
COMPLEXITIES = ("single_step", "multi_step")


# ---------- loading ----------
def load_results() -> dict:
    """One row per (task, setting, level, mode): the latest successful run."""
    latest = {}
    with open(RESULTS_LOG, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["error"]:
                latest[(r["task_id"], r["setting"], r["level"], r["mode"])] = r
    return latest


def is_long_answer(r: dict) -> bool:
    return min(len(g.split()) for g in r["gold"]) > LONG_ANSWER_WORDS


# ---------- statistics ----------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: list[float]) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of paired differences."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    means = sorted(mean(rng.choices(diffs, k=n)) for _ in range(BOOTSTRAP_SAMPLES))
    return means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment for multiple comparisons."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ---------- tables ----------
def cell_table(rows: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["mode"], r["level"], r["setting"], r["complexity"])].append(r)

    table = []
    for (mode, level, setting, complexity), rs in sorted(cells.items()):
        words_original = sum(r["words_original"] for r in rs)
        words_sent = sum(r["words_sent"] for r in rs)
        table.append({
            "mode": mode, "level": level, "setting": setting, "complexity": complexity,
            "n": len(rs),
            "EM": mean(r["em"] for r in rs),
            "F1": mean(r["f1"] for r in rs),
            "all_gold_retrieved": mean(r["all_gold_retrieved"] for r in rs),
            "messages_per_task": mean(r["n_messages"] for r in rs),
            "words_sent_per_task": mean(r["words_sent"] for r in rs),
            "compression": words_sent / words_original if words_original else float("nan"),
            "diverged": mean(r["replay_diverged"] for r in rs) if mode == "replay" else "",
        })
    return table


def paired_table(results: dict, mode: str, keep=lambda summary_row: True) -> list[dict]:
    """Full (live) vs short_summary (given mode), paired by task, per setting x complexity."""
    table = []
    for setting in SETTINGS:
        for complexity in COMPLEXITIES:
            pairs = []
            for (task_id, s, level, m), full in results.items():
                if (s, level, m) != (setting, "full", "live") or full["complexity"] != complexity:
                    continue
                summary = results.get((task_id, setting, "short_summary", mode))
                if summary and keep(summary):
                    pairs.append((full, summary))
            if not pairs:
                continue

            b = sum(f["em"] == 1 and s["em"] == 0 for f, s in pairs)
            c = sum(f["em"] == 0 and s["em"] == 1 for f, s in pairs)
            f1_diffs = [s["f1"] - f["f1"] for f, s in pairs]
            low, high = bootstrap_ci(f1_diffs)
            table.append({
                "setting": setting, "complexity": complexity, "n_pairs": len(pairs),
                "EM_full": mean(f["em"] for f, _ in pairs),
                "EM_summary": mean(s["em"] for _, s in pairs),
                "EM_change": mean(s["em"] - f["em"] for f, s in pairs),
                "lost_b": b, "gained_c": c,
                "mcnemar_p": mcnemar_exact(b, c),
                "F1_change": mean(f1_diffs), "F1_CI_low": low, "F1_CI_high": high,
                "lost_but_gold_retrieved": sum(
                    s["all_gold_retrieved"] for f, s in pairs if f["em"] == 1 and s["em"] == 0),
            })

    for row, adjusted in zip(table, holm([row["mcnemar_p"] for row in table])):
        row["holm_p"] = adjusted
    return table


# ---------- output ----------
def show_and_save(title: str, filename: str, table: list[dict]):
    print(f"\n=== {title} ===")
    if not table:
        print("  (no data)")
        return
    columns = list(table[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    widths = [max(len(c), *(len(fmt(r[c])) for r in table)) for c in columns]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for r in table:
        print("  " + "  ".join(fmt(r[c]).ljust(w) for c, w in zip(columns, widths)))

    with open(OUT_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: fmt(r[c]) for c in columns} for r in table)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results = load_results()
    rows = list(results.values())
    print(f"Loaded {len(rows)} results.")

    show_and_save("1. Cell summary", "1_cells.csv", cell_table(rows))
    show_and_save("2. Full vs short summary (LIVE)", "2_paired_live.csv",
                  paired_table(results, "live"))
    show_and_save("3. Full vs short summary (REPLAY)", "3_paired_replay.csv",
                  paired_table(results, "replay"))
    show_and_save("4. Replay, excluding diverged tasks", "4_paired_replay_no_diverged.csv",
                  paired_table(results, "replay", keep=lambda r: not r["replay_diverged"]))
    show_and_save(f"5. Live, excluding answers over {LONG_ANSWER_WORDS} words (sensitivity)",
                  "5_paired_live_short_answers.csv",
                  paired_table(results, "live", keep=lambda r: not is_long_answer(r)))
    print(f"\nTables saved to {OUT_DIR}")


if __name__ == "__main__":
    main()