# ベースイメージ
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係のコピーとインストール
COPY requirements.txt /app/requirements.txt

RUN set -eux \
    && apt-get update \
    && apt-get install -y \
        libgomp1 \
        build-essential \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --trusted-host pypi.python.org -r /app/requirements.txt \
    && apt-get remove --purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

# アプリケーションコードをコピー
COPY ./app /app

# ポートを公開
EXPOSE 8888

# コンテナ起動時に実行するコマンド
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888"]
