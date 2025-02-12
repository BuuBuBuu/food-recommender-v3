# Use Python 3.9 slim image as base with platform specification
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Linux
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app /app/app

# Create data directory
RUN mkdir -p data

# IMPORTANT: Copy only the required model files
COPY data/tfidf_vectorizer.pkl /app/data/
COPY app/models/production_ranking_model_enhanced.pkl /app/app/models/

# Explicitly set the Python path
ENV PYTHONPATH=/app

# Set environment variables for Linux
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
