FROM python:3.10-slim

WORKDIR /app

# Set PYTHONPATH so src module can be imported
ENV PYTHONPATH=/app

#System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

#Copy requirements 
COPY requirements.txt .

#Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Install DVC with all extras to support various remotes
RUN pip install --no-cache-dir "dvc[all]"

#Copy relevant code folders
COPY src ./src
COPY README.md .
