# Deployment

## Prerequisites

- Docker with Compose v2
- NVIDIA GPU with drivers
- NVIDIA Container Toolkit
- ~50GB disk space for model
- [Task](https://taskfile.dev) (go-task) for running commands

## Quick Start

```bash
# 1. Clone and configure
git clone <repo>
cd llm-mail-description
cp .env.example .env

# 2. Set Hugging Face token (for Llama)
echo "HF_TOKEN=your_token_here" >> .env

# 3. Start services
task up

# 4. Check status
task logs
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI application |
| `triton-gpu-0` | 8001 (gRPC), 8002 (metrics) | Triton on GPU 0 |
| `triton-gpu-1` | 8011 (gRPC), 8012 (metrics) | Triton on GPU 1 |

## Step-by-Step

### 1. Download Model

First run downloads and converts the model:

```bash
task init-model
```

This takes 30-60 minutes depending on network and GPU.

### 2. Start Services

```bash
task up
```

Wait for health checks to pass:

```bash
docker compose ps
```

### 3. Verify

```bash
# Health check
curl http://localhost:8000/health

# Readiness (includes Triton status)
curl http://localhost:8000/ready

# Test analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "Dear Sir, Please find attached invoice #12345 for $500."}'
```

## Development Mode

Run without GPU (mock inference):

```bash
task up:dev
```

## Monitoring

### Logs

```bash
# API logs
task logs

# Triton logs
task logs:triton

# All logs
docker compose logs -f
```

### Metrics

Triton exposes Prometheus metrics:

- GPU 0: http://localhost:8002/metrics
- GPU 1: http://localhost:8012/metrics

### Stats Endpoint

```bash
curl http://localhost:8000/stats
```

## Troubleshooting

### Model init fails

```bash
# Check logs
docker compose logs model-init

# Common issues:
# - Invalid HF_TOKEN
# - Insufficient disk space
# - Network issues
```

### Triton won't start

```bash
# Check GPU availability
nvidia-smi

# Check container logs
docker compose logs triton-gpu-0
```

### High latency

1. Check batch processor stats: `curl /stats`
2. Verify both GPUs are used
3. Consider increasing `max_batch_size`

## Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set appropriate `LOG_LEVEL`
- [ ] Configure reverse proxy (nginx)
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backup for model volume
- [ ] Review resource limits in docker-compose.yaml
