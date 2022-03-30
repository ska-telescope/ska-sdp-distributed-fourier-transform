FROM python:3.9-slim as build

ENV POETRY_HOME=/opt/poetry

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python -

COPY . ./
RUN /opt/poetry/bin/poetry export --without-hashes -o requirements.txt

FROM python:3.9-slim

WORKDIR /install
RUN apt-get update && apt-get install -y gcc

COPY --from=build requirements.txt ./
RUN pip install --no-cache-dir --no-compile -r requirements.txt

WORKDIR /app
COPY . ./
