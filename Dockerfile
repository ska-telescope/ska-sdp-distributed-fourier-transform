FROM python:3.10-slim as build

COPY . ./
RUN pip install poetry && \
    poetry export --without-hashes -o requirements.txt

FROM python:3.10-slim

WORKDIR /install
RUN apt-get update && apt-get install -y gcc

COPY --from=build requirements.txt ./
RUN pip install --no-cache-dir --no-compile -r requirements.txt
RUN pip install jupyterlab

WORKDIR /app
ENV PYTHONPATH=$PYTHONPATH:/app/src

COPY . ./
