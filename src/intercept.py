"""Step 5: the message interception layer.

Every message between agents goes through MessageBus.pass_message(), which:
  1. takes the sender's original text (in replay mode: the FROZEN original logged in
     the most recent full-context live run of the same task and setting),
  2. applies the communication level (full / short_summary),
  3. logs original and transmitted text to the message log,
  4. returns the transmitted text to the receiver.

Replay: logged messages are matched in order by (sender, receiver). If the run asks for
a different message than the log holds, asks for more, or stops early, the task is
flagged replay_diverged and continues live from that point.
"""
import json
import time
import uuid
from pathlib import Path

from .levels import Transformer, word_count

MODES = ("live", "replay")


def load_frozen_messages(log_path: Path) -> dict[tuple[str, str], list[dict]]:
    """For replay: the messages of the most recent full-context live run of each
    (task_id, setting), in the order they were sent."""
    log_path = Path(log_path)
    if not log_path.exists():
        return {}

    latest: dict[tuple[str, str], tuple[str, list[dict]]] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            message = json.loads(line)
            if message["level"] != "full" or message["mode"] != "live":
                continue
            key = (message["task_id"], message["setting"])
            if key not in latest or latest[key][0] != message["run_id"]:
                latest[key] = (message["run_id"], [])
            latest[key][1].append(message)

    return {
        key: sorted(messages, key=lambda m: m["step"])
        for key, (_, messages) in latest.items()
    }


class MessageBus:
    """Drop-in replacement for DirectChannel: same pass_message() method, plus the
    communication level, logging and replay."""

    def __init__(self, level: str, transformer: Transformer, log_path: Path, mode: str = "live"):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        if transformer.level != level:
            raise ValueError(f"Transformer is for {transformer.level!r}, not {level!r}")
        self.level = level
        self.transform = transformer
        self.mode = mode
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_id = None

    # ---------- per task ----------
    def start_task(self, task: dict, setting: str, frozen: list[dict] | None = None):
        """Reset for a new task. In replay mode, pass that task's frozen messages."""
        self.run_id = uuid.uuid4().hex
        self.task_id = task["task_id"]
        self.complexity = task["complexity"]
        self.setting = setting
        self.step = 0
        self.frozen = list(frozen or [])
        self.diverged = False
        self.n_messages = 0
        self.n_replayed = 0
        self.words_original = 0
        self.words_sent = 0

    def stats(self) -> dict:
        """Totals for the current task, to store with its result."""
        return {
            "run_id": self.run_id,
            "n_messages": self.n_messages,
            "n_replayed": self.n_replayed,
            "words_original": self.words_original,
            "words_sent": self.words_sent,
            "replay_diverged": self.mode == "replay" and (self.diverged or bool(self.frozen)),
        }

    def _take_frozen(self, sender: str, receiver: str) -> str | None:
        """In replay mode, return the next frozen original if it matches this message."""
        if self.mode != "replay" or self.diverged:
            return None
        if not self.frozen:
            self.diverged = True
            return None
        next_message = self.frozen[0]
        if (next_message["sender"], next_message["receiver"]) == (sender, receiver):
            self.frozen.pop(0)
            return next_message["original"]
        self.diverged = True
        return None

    # ---------- the interception point ----------
    async def pass_message(self, sender: str, receiver: str,
                           content: str | None = None, produce=None) -> str:
        if self.task_id is None:
            raise RuntimeError("Call start_task() before passing messages.")

        frozen = self._take_frozen(sender, receiver)
        replayed = frozen is not None
        if replayed:
            original = frozen
        elif content is not None:
            original = content
        else:
            original = await produce()
        original = str(original)

        sent = await self.transform(original)

        record = {
            "run_id": self.run_id,
            "ts": time.time(),
            "mode": self.mode,
            "level": self.level,
            "setting": self.setting,
            "complexity": self.complexity,
            "task_id": self.task_id,
            "step": self.step,
            "sender": sender,
            "receiver": receiver,
            "original": original,
            "sent": sent,
            "words_original": word_count(original),
            "words_sent": word_count(sent),
            "replayed": replayed,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.step += 1
        self.n_messages += 1
        self.n_replayed += int(replayed)
        self.words_original += record["words_original"]
        self.words_sent += record["words_sent"]
        return sent