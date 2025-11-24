# syntax=docker/dockerfile:1.2
FROM python:3.10-slim
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY challenge/ /app/challenge/
COPY artifacts/ /app/artifacts/
COPY . /app/

# Exponer el puerto en el que correrá la API
ENV PORT 8080

# Comando para iniciar la API usando Uvicorn
# El flag --host 0.0.0.0 es necesario para que sea accesible dentro del contenedor.
CMD uvicorn challenge.api:app --host 0.0.0.0 --port $PORT