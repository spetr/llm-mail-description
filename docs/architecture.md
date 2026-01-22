# Architecture

## Overview

MailBrain is a service for analyzing emails using a local LLM with constrained JSON output (structured generation).

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐                                          │
│   │   model-init     │ ──▶ Downloads & converts model (once)    │
│   └────────┬─────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐      ┌──────────────────┐               │
│   │  triton-gpu-0    │      │  triton-gpu-1    │               │
│   │  (TensorRT-LLM)  │      │  (TensorRT-LLM)  │               │
│   └────────┬─────────┘      └────────┬─────────┘               │
│            │                         │                          │
│            └────────────┬────────────┘                          │
│                         │ gRPC                                   │
│                         ▼                                        │
│                  ┌─────────────┐                                │
│                  │   FastAPI   │                                │
│                  │   + Batch   │                                │
│                  │  Processor  │                                │
│                  └─────────────┘                                │
│                         │                                        │
└─────────────────────────┼────────────────────────────────────────┘
                          │ HTTP :8000
                          ▼
                      Clients
```

## Components

### FastAPI Application (`src/api/`)

- HTTP endpoint for email analysis
- Request validation
- Response formatting

### Micro-Batch Processor (`src/batch/`)

- Collects incoming requests into batches
- Configurable batch size and wait time
- Transparent to clients - they send single requests
- Abstracted protocol for future queue backends (RabbitMQ)

### Inference Backend (`src/inference/`)

- Protocol abstraction for different backends
- Triton client implementation
- Prompt rendering with Jinja2
- JSON schema for constrained decoding

### Schema Loader (`src/schema/`)

- Loads output schema from YAML
- Dynamically generates Pydantic model
- Used for validation and constrained decoding

### Configuration (`src/config/`)

- Pydantic Settings for environment variables
- YAML config loading (app.yaml, schema.yaml)

## Data Flow

```
1. Client sends POST /analyze with email content
           │
           ▼
2. FastAPI validates request
           │
           ▼
3. Request added to batch queue
           │
           ▼
4. Batch processor waits for:
   - max_batch_size (16) requests, OR
   - max_wait_ms (50ms) timeout
           │
           ▼
5. Batch sent to Triton (round-robin between GPUs)
           │
           ▼
6. TensorRT-LLM generates JSON with constrained decoding
           │
           ▼
7. Response validated against Pydantic model
           │
           ▼
8. Client receives response
```

## Scaling

### Horizontal Scaling

- Add more Triton instances (GPU containers)
- Update `app.yaml` with new URLs
- Load balancing is built-in (round-robin)

### Future: Message Queue

The batch processor protocol allows swapping implementation:

```python
# Current: In-memory queue
batch_processor = MicroBatchProcessor(...)

# Future: RabbitMQ
batch_processor = RabbitMQProcessor(...)
```

No changes needed in API layer.
