"""Mede só o tempo do modelo (sem HTTP), comparando sklearn e ONNX.

Uso (depois de treinar e de rodar python -m triage.optimize):
    poetry run python scripts/benchmark_model.py --n 2000
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmark_latency import PHRASES, percentile

from triage.predict import load_model, predict_text

RESULTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"


def measure(backend: str, n: int, warmup: int) -> dict:
    """Chama predict_text n vezes, uma frase por vez, como a API faz."""
    model, _ = load_model(backend=backend)
    latencies = []
    for i in range(warmup + n):
        start = time.perf_counter()
        predict_text(model, PHRASES[i % len(PHRASES)])
        if i >= warmup:
            latencies.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(percentile(latencies, 50), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "p99_ms": round(percentile(latencies, 99), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="predições medidas")
    parser.add_argument("--warmup", type=int, default=50, help="predições de aquecimento")
    args = parser.parse_args()

    summary = {"measured_at": datetime.now(UTC).isoformat(), "n_predictions": args.n}
    for backend in ("sklearn", "onnx"):
        summary[backend] = measure(backend, args.n, args.warmup)
        print(backend, summary[backend])
    summary["speedup_p50"] = round(summary["sklearn"]["p50_ms"] / summary["onnx"]["p50_ms"], 1)
    print(f"ONNX é {summary['speedup_p50']}x mais rápido no p50")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "latency_model_only.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Resultado salvo em {out_path}")


if __name__ == "__main__":
    main()
