# Stage 1: Build Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend & Final Image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Copy minimal files first
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --no-dev

# Bake model into image
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); print('Model cached.')"

# Copy remaining files
COPY . .

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/out ./frontend/out

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "civicsetu.api.main:app", \
     "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
