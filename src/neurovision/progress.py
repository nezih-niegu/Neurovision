"""
neurovision.progress
====================
A dependency-free progress bar with an ETA.

Degrades to one plain line per update when stdout is not a terminal, because
these runs are usually redirected into a log file and a carriage-return bar
turns such a file into an unreadable single line.
"""

from __future__ import annotations

import shutil
import sys
import time


def _fmt(seconds: float) -> str:
    if seconds != seconds or seconds < 0:      # NaN or nonsense
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m:02d}:{s:02d}"


class Progress:
    """Usage:

        bar = Progress(total, "fitting")
        for ...:
            ...
            bar.update(note="sub-hc1")
        bar.close()

    The ETA is a running mean over completed items, which suits this workload
    because the items are near-identical in cost. It is deliberately not an
    exponential moving average: with a handful of slow outliers, a plain mean
    gives a steadier and more honest estimate.
    """

    def __init__(self, total: int, label: str = "", stream=None,
                 min_interval: float = 0.2):
        self.total = max(int(total), 1)
        self.label = label
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.min_interval = min_interval
        self.n = 0
        self.t0 = time.time()
        self._last = 0.0

    def update(self, step: int = 1, note: str = ""):
        self.n += step
        now = time.time()
        done = self.n >= self.total
        if not done and self.tty and now - self._last < self.min_interval:
            return
        self._last = now
        self._draw(note, done)

    def _draw(self, note: str, done: bool):
        elapsed = time.time() - self.t0
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.n / self.total
        head = f"{self.label} " if self.label else ""
        tail = (f"{self.n}/{self.total} {pct:5.1f}%  "
                f"elapsed {_fmt(elapsed)}  eta {_fmt(eta)}"
                + (f"  {note}" if note else ""))

        if not self.tty:
            self.stream.write(f"  {head}{tail}\n")
            self.stream.flush()
            return

        width = shutil.get_terminal_size((100, 20)).columns
        barw = max(10, min(30, width - len(head) - len(tail) - 6))
        filled = int(barw * self.n / self.total)
        bar = "#" * filled + "." * (barw - filled)
        line = f"\r  {head}[{bar}] {tail}"
        self.stream.write(line[: width - 1].ljust(width - 1))
        self.stream.flush()
        if done:
            self.stream.write("\n")
            self.stream.flush()

    def close(self):
        if self.tty and self.n < self.total:
            self.stream.write("\n")
            self.stream.flush()
