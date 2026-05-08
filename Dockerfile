FROM python:3.10-slim

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements-serving.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY configs ./configs
COPY models ./models
COPY data/processed ./data/processed
COPY data/splits ./data/splits

RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
