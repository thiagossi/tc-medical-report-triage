"""Gera tráfego na API para popular os painéis do Grafana.

Mistura frases válidas com algumas inválidas (texto vazio ou longo demais), para que o
painel de taxa de erro também mostre algo.

Uso (com a stack no ar):
    poetry run python scripts/load_test.py --seconds 120
"""

import argparse
import random
import time

import httpx

PHRASES = [
    "My chest hurts and I can't breathe",
    "I cut my hand and it is bleeding a lot",
    "There is a deep wound on my leg",
    "I feel dizzy when I stand up",
    "I have a terrible headache since yesterday",
    "I have had a cough for two weeks",
    "My ear hurts a lot",
    "I have acne on my face",
    "My back hurts after working all day",
    "My knee hurts when I climb stairs",
    "My hair is falling out",
]
INVALID_PAYLOADS = [{"text": ""}, {"text": "   "}, {"text": "a" * 1001}, {}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--seconds", type=int, default=120, help="duração do teste")
    parser.add_argument("--rps", type=float, default=10, help="requisições por segundo")
    parser.add_argument("--error-rate", type=float, default=0.1, help="fração de inválidas")
    args = parser.parse_args()

    random.seed(42)
    counts: dict[int, int] = {}
    deadline = time.monotonic() + args.seconds
    with httpx.Client(base_url=args.url, timeout=10) as client:
        while time.monotonic() < deadline:
            if random.random() < args.error_rate:
                payload = random.choice(INVALID_PAYLOADS)
            else:
                payload = {"text": random.choice(PHRASES)}
            status = client.post("/predict", json=payload).status_code
            counts[status] = counts.get(status, 0) + 1
            if random.random() < 0.05:
                client.get("/health")
            time.sleep(1 / args.rps)

    print("Requisições por status:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
