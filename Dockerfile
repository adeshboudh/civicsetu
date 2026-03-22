FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --no-dev

# Bake model into image — eliminates cold start on first request
RUN uv run python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
print('Model cached.')
"

COPY . .

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "civicsetu.api.main:app", \
     "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
