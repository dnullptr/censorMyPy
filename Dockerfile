# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_clean.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_clean.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port for web server
EXPOSE 8000

# Default command
CMD ["python", "server.py"]
