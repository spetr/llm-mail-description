# Configuration

## Overview

Configuration is split into:

| File | Purpose | Reload |
|------|---------|--------|
| `.env` | Secrets, environment | Restart required |
| `config/app.yaml` | Infrastructure settings | Restart required |
| `config/schema.yaml` | Output JSON schema | Restart required |
| `config/prompts/*.jinja` | Prompt templates | Restart required |

## Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face token (for gated models) | - |
| `ENVIRONMENT` | `development` or `production` | `development` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `CONFIG_PATH` | Path to config directory | `./config` |

## app.yaml

Main application configuration.

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

batching:
  max_batch_size: 16    # Max emails per batch
  max_wait_ms: 50       # Max wait before processing

inference:
  backend: "triton"
  triton_urls:
    - "triton-gpu-0:8001"
    - "triton-gpu-1:8001"
  model_name: "llama-3.1-8b"
  timeout_seconds: 30.0

model:
  hf_repo: "meta-llama/Llama-3.1-8B-Instruct"
  max_input_tokens: 4096
  max_output_tokens: 512

output:
  language: "en"
```

## schema.yaml

Defines the output JSON structure. Changes here affect:
- Pydantic model generation
- Constrained decoding
- Response validation

```yaml
output:
  language: "en"

fields:
  short_summary:
    type: string
    max_length: 100
    description: "One sentence summary"

  detailed_summary:
    type: string
    max_length: 500
    description: "2-4 sentences for RAG"

  keywords:
    type: array
    item_type: string
    min_items: 3
    max_items: 10

  # Categories use 'groups' for organized prompt display
  categories:
    type: array
    item_type: enum
    min_items: 1
    groups:
      financial:
        - invoice
        - payment_request
        - refund
      sales:
        - order_confirmation
        - shipping_notification
      # Add more groups...

  tone:
    type: enum
    enum_values:
      - formal
      - informal
      - friendly
      - angry

  detected_language:
    type: string
```

### Field Types

| Type | Parameters |
|------|------------|
| `string` | `max_length` |
| `enum` | `enum_values` (required) |
| `array` | `item_type`, `min_items`, `max_items`, `enum_values` or `groups` |

### Using Groups

For better prompt organization, use `groups` instead of flat `enum_values`:

```yaml
categories:
  type: array
  item_type: enum
  groups:
    financial:
      - invoice
      - payment_request
    sales:
      - order_confirmation
```

Groups are:
- Automatically flattened into `enum_values` for Pydantic validation
- Displayed grouped in the prompt for better LLM comprehension

## Prompt Templates

Located in `config/prompts/`. Uses Jinja2 syntax.

### analyze.jinja

Available variables:
- `{{ email_content }}` - The email text
- `{{ output_language }}` - Target language (e.g., "en")
- `{{ categories }}` - Flat list of all categories
- `{{ category_groups }}` - Dict of group_name -> [values]
- `{{ tones }}` - List of available tones

Example using groups:

```jinja
=== CATEGORIES ===
{% for group_name, values in category_groups.items() %}
{{ group_name }}: {{ values | join(", ") }}
{% endfor %}
```
