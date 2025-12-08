FROM python:3.10-slim

WORKDIR /app

#System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#Copy requirements 
COPY requirements.txt .

#Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy relevant code folders
COPY data ./data
COPY models ./models
COPY src ./src
COPY utils ./utils
COPY config ./config
COPY pyproject.toml .
COPY README.md .
