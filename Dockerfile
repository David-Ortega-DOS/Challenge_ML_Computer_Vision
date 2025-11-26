# syntax=docker/dockerfile:1.2
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

COPY requirements-docker.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --default-timeout=300 -r requirements-docker.txt

COPY ./challenge /app/challenge
COPY ./artifacts /app/artifacts


ENV APP_MODULE="challenge.api"

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "challenge.api:app"]