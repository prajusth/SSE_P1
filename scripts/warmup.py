"""
warmup.py — Heat up the CPU before measurements.

Why: Cold hardware has lower resistance = lower energy readings.
The course slides say to run a CPU-intensive task for at least 1 min
(5 min recommended) before starting energy measurements.
We use Fibonacci computation as a simple CPU burner.
"""

import time


def fibonacci_burn(n=100000):
    """Compute Fibonacci to burn CPU cycles."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def warmup(seconds=120):
    """Run CPU-intensive work for the given number of seconds."""
    print(f"  CPU warm-up for {seconds}s...")
    end_time = time.time() + seconds
    while time.time() < end_time:
        fibonacci_burn()
    print("  Warm-up done.")


if __name__ == "__main__":
    # If you run this file directly: python3 scripts/warmup.py
    warmup(120)
