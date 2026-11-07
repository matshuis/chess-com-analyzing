FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-cache

COPY az_chess.py az_network.py az_mcts.py az_rules.py az_benchmark.py az_parallel.py az_replay.py az_selfplay.py az_training.py train.py ./
RUN mkdir -p /app/runs && chown 1000:1000 /app/runs
USER 1000:1000

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "train.py"]
