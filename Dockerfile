FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config

RUN python -m src.train_baseline

EXPOSE 8010
CMD ["uvicorn", "src.serve_model:app", "--host", "0.0.0.0", "--port", "8010"]
