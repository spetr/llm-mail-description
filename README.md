# LLM Mail Description

Email analysis service using local LLM with constrained JSON output.

## Features

- Analyzes emails and extracts structured information (summaries, keywords, categories, tone)
- Uses TensorRT-LLM with Triton Inference Server for high-throughput GPU inference
- Constrained decoding ensures valid JSON output matching the schema
- Micro-batching for efficient processing of concurrent requests
- Dynamic schema configuration via YAML

## Quick Start

```bash
# Install Task runner
# https://taskfile.dev

# Start services (requires NVIDIA GPU)
task up

# Or development mode (no GPU, mock inference)
task up:dev

# Test the API
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "Dear Sir, Please find attached invoice #12345 for $500."}'
```

## Documentation

- [Configuration Guide](docs/configuration.md)
- [Development Guide](docs/development.md)
- [Deployment Guide](docs/deployment.md)
