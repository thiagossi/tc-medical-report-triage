# Imagem da API de triagem.
# Estágio 1 (builder): instala as dependências com o Poetry.
# Estágio 2 (final): leva só o ambiente pronto e o código, sem o Poetry.

FROM python:3.11-slim AS builder

ENV POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIP_NO_CACHE_DIR=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

COPY src ./src
RUN poetry install --only main


FROM python:3.11-slim

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

RUN useradd --create-home appuser

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

RUN mkdir -p /app/models && chown appuser /app/models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "triage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
