"""Console formatting and transcript capture shared by the inspections."""

from contextlib import redirect_stdout
from io import StringIO
import sys


class Tee:
    def __init__(self, terminal, transcript):
        self.terminal = terminal
        self.transcript = transcript

    def write(self, text):
        self.terminal.write(text)
        self.transcript.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.transcript.flush()

    def isatty(self):
        return self.terminal.isatty()


def capture_console(transcript=None):
    if transcript is None:
        transcript = StringIO()
    return transcript, redirect_stdout(Tee(sys.stdout, transcript))


def range_summary(label, exceeded, total, radius):
    percent = 100 * exceeded / total if total else 0
    print(f"{exceeded:,}/{total:,} {label} > {radius:g}m --> {percent:.2f}%")

def print_histogram(histogram, total, label):
    print(f"\n{label}")
    print("    Count | Amount of targets |     in % | visual")
    print("----------+-------------------+----------+----------------------------------------")
    for count in range(max(histogram, default=0) + 1):
        amount = histogram[count]
        percent = 100 * amount / total
        bars = "I" * int(round(percent)) or ("." if amount else "")
        print(f"{count:9d} | {amount:17,d} | {percent:7.2f}% | {bars}")


