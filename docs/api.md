# API Reference

Base URL: `http://localhost:8000`

## Endpoints

### POST /analyze

Analyze a single email.

**Request:**

```json
{
  "content": "Email content here (plain text or markdown)"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | Email content (1-50000 chars) |

**Response:**

```json
{
  "analysis": {
    "short_summary": "Invoice #12345 for $500 from Acme Corp",
    "detailed_summary": "This email contains an invoice from Acme Corp for services rendered in January. The total amount is $500 with payment due in 30 days.",
    "keywords": ["invoice", "payment", "acme", "january"],
    "categories": ["invoice"],
    "tone": "formal",
    "detected_language": "en"
  },
  "processing_time_ms": 245.32
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 422 | Invalid input |
| 500 | Inference error |
| 503 | Service unavailable |

**Example:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Dear Team,\n\nPlease review the attached quarterly report.\n\nBest regards,\nJohn"
  }'
```

---

### GET /health

Basic liveness check.

**Response:**

```json
{
  "status": "ok"
}
```

---

### GET /ready

Readiness check including backend status.

**Response:**

```json
{
  "ready": true,
  "details": {
    "batch_processor": {
      "running": true,
      "queue_size": 0
    }
  }
}
```

---

### GET /stats

Service statistics.

**Response:**

```json
{
  "batch_processor": {
    "queue_size": 2,
    "total_requests": 1523,
    "total_batches": 156,
    "total_errors": 0,
    "avg_batch_size": 9.76,
    "running": true
  }
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

## Rate Limiting

No built-in rate limiting. Implement at reverse proxy level if needed.

## OpenAPI

Interactive documentation available at:
- Swagger UI: http://localhost:8000/docs (development only)
- ReDoc: http://localhost:8000/redoc (development only)
