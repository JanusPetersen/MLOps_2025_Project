FROM python:3.10-slim

WORKDIR /app

# Set PYTHONPATH so src module can be imported
ENV PYTHONPATH=/app

#System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#Copy requirements 
COPY requirements.txt .

#Install protobuf first to ensure correct version
RUN pip install --no-cache-dir "protobuf<5.0.0,>=3.20.0"

#Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy relevant code folders
COPY src ./src
COPY README.md .
