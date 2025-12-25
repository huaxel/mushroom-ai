# Builder stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Create a non-root user
RUN useradd -m -r appuser && chown appuser /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app files
COPY . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Start FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]