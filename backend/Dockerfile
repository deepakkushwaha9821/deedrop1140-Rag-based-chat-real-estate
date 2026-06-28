FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
	libglib2.0-0 \
	libgl1 \
	libsm6 \
	libxext6 \
	libxrender1 \
	libxcb1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 7860"]