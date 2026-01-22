# API container
FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install --no-cache-dir .

# Copy source code
COPY src/ src/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "src"]
