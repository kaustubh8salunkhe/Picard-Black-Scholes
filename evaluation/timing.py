import time
import numpy as np
import torch


def measure_latency(callable_fn, warmup=20, repetitions=100):
    for _ in range(warmup):
        _ = callable_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for _ in range(repetitions):
        t0 = time.perf_counter_ns()
        _ = callable_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        timings.append((t1 - t0) / 1e6)  # Convert ns to ms

    timings = np.array(timings)
    return {
        "median_ms": float(np.median(timings)),
        "p5_ms": float(np.percentile(timings, 5)),
        "p95_ms": float(np.percentile(timings, 95)),
    }
