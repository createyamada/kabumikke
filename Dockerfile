FROM python:3.9-slim

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt

RUN set -eux \
    && apt-get update \
    && apt-get install -y \
        libgomp1 \
        build-essential \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --trusted-host pypi.python.org -r /src/requirements.txt \
    && apt-get remove --purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip 

COPY . /src

EXPOSE 8888

CMD ["uvicorn", "src.app.api.app:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "8888"]
