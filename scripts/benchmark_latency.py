"""Mede a latência da API de triagem (p50, p95, p99).

Uso (com a API no ar):
    poetry run python scripts/benchmark_latency.py --label baseline
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PHRASES = [
    "My chest hurts and I can't breathe",
    "I cut my hand and it is bleeding a lot",
    "I have acne on my face",
    "My back hurts after working all day",
    "I feel dizzy when I stand up",
    "I have a terrible headache since yesterday",
    "My knee hurts when I climb stairs",
    "I have had a cough for two weeks",
]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"


def percentile(values: list[float], p: float) -> float:
    """Percentil p (0 a 100) de uma lista de valores."""
    ordered = sorted(values)
    index = round(p / 100 * (len(ordered) - 1))
    return ordered[index]


def run(url: str, n_requests: int, warmup: int) -> list[float]:
    """Envia as requisições em sequência e devolve a latência de cada uma, em ms."""
    latencies = []
    with httpx.Client(base_url=url, timeout=10) as client:
        for i in range(warmup + n_requests):
            text = PHRASES[i % len(PHRASES)]
            start = time.perf_counter()
            response = client.post("/predict", json={"text": text})
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            if i >= warmup:  # as primeiras chamadas (aquecimento) não entram na conta
                latencies.append(elapsed_ms)
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=500, help="requisições medidas")
    parser.add_argument("--warmup", type=int, default=20, help="requisições de aquecimento")
    parser.add_argument("--label", default="baseline", help="nome do resultado salvo")
    args = parser.parse_args()

    latencies = run(args.url, args.n, args.warmup)
    summary = {
        "label": args.label,
        "measured_at": datetime.now(UTC).isoformat(),
        "url": args.url,
        "n_requests": args.n,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"latency_{args.label}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{args.n} requisições → média {summary['mean_ms']} ms")
    print(f"p50 {summary['p50_ms']} ms | p95 {summary['p95_ms']} ms | p99 {summary['p99_ms']} ms")
    print(f"Resultado salvo em {out_path}")


if __name__ == "__main__":
    main()
