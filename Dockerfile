# syntax=docker/dockerfile:1.2
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar dependencias
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Copiar el proyecto completo
COPY . .

# Declarar el módulo principal
ENV APP_MODULE="challenge.api:app"

EXPOSE 8000

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]