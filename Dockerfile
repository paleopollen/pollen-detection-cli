FROM python:3.9-slim

WORKDIR /usr/src/app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create data directory
RUN mkdir -p /data

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src/ /usr/src/app/

# Set the entrypoint
ENTRYPOINT ["python3", "pollen_detection_cli.py"]
