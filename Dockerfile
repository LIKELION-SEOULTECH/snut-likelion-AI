FROM python:3.9-slim

WORKDIR /app

# 시스템 라이브러리 최소화
RUN apt-get update && apt-get install -y --no-install-recommends fonts-nanum && rm -rf /var/lib/apt/lists/*

# 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pandas numpy transformers accelerate fastapi uvicorn

COPY app.py .
COPY kluebert_intent_out/ ./kluebert_intent_out/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]