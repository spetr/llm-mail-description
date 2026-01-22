# Development Guide

## Setup

```bash
# Clone repository
git clone <repo>
cd mailbrain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install with dev dependencies
task dev
```

## Running Locally

### Without GPU (Mock Mode)

```bash
# Start API with mock inference
task run

# Or with Docker
task up:dev
```

Mock mode returns valid responses matching the schema, useful for API development.

### With GPU

```bash
task up
```

## Project Structure

```
src/
├── api/                 # FastAPI application
│   ├── app.py          # App factory, lifespan
│   ├── dependencies.py # Dependency injection
│   ├── exceptions.py   # Custom HTTP exceptions
│   └── routes/         # Endpoint handlers
│       ├── analyze.py  # POST /analyze
│       └── health.py   # Health endpoints
│
├── batch/              # Batch processing
│   ├── protocol.py     # Abstract interface
│   └── processor.py    # Micro-batch implementation
│
├── inference/          # LLM inference
│   ├── protocol.py     # Abstract interface
│   ├── triton.py       # Triton client
│   └── prompt.py       # Jinja2 prompt rendering
│
├── schema/             # Dynamic schema
│   └── loader.py       # YAML → Pydantic
│
├── config/             # Configuration
│   └── settings.py     # Pydantic Settings
│
├── logging/            # Structured logging
│   └── setup.py        # structlog config
│
└── __main__.py         # Entry point
```

## Adding Features

### New Endpoint

1. Create route file in `src/api/routes/`
2. Register in `src/api/routes/__init__.py`

```python
# src/api/routes/my_endpoint.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint():
    return {"hello": "world"}
```

### New Inference Backend

1. Implement `InferenceBackend` protocol
2. Add factory logic in `src/api/app.py`

```python
# src/inference/vllm.py
from src.inference.protocol import InferenceBackend

class VLLMBackend(InferenceBackend):
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def analyze_batch(self, emails: list[str]) -> list[BaseModel]: ...
    async def health_check(self) -> bool: ...
    def get_stats(self) -> dict[str, Any]: ...
```

### New Schema Field

1. Add to `config/schema.yaml`
2. Update prompt in `config/prompts/analyze.jinja`
3. Restart service

## Testing

```bash
# Run all tests
task test

# With coverage
task test:cov

# Specific test
pytest tests/unit/test_schema_loader.py -v
```

## Code Quality

```bash
# Lint
task lint

# Format
task format

# Type check
task typecheck

# All checks at once
task check
```

## Debugging

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG task run
```

### Inspect Batch Processing

```bash
# Watch stats
watch -n1 'curl -s localhost:8000/stats | jq'
```

### Test Prompt Rendering

```python
from src.schema.loader import load_schema
from src.inference.prompt import PromptManager
from pathlib import Path

_, config = load_schema(Path("config/schema.yaml"))
pm = PromptManager(Path("config/prompts"))
pm.load_templates()

prompt = pm.render_analyze_prompt("Test email", config)
print(prompt)
```
