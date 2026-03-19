# serving/

End-to-end serving layer for the melanoma skin cancer classifier. Runs two Docker containers
(FastAPI inference API + Streamlit UI) with a single command.

## Quick start

```bash
# 1. Place a trained checkpoint at models/best_model.pt (relative to repo root)
# 2. From the serving/ directory:
docker compose up --build
```

- UI: http://localhost:7777
- API: http://localhost:8000
- Health check: `curl http://localhost:8000/health`

## Requirements

| Requirement | Notes |
|-------------|-------|
| Docker + Docker Compose v2 | `docker compose` (not `compose`) |
| Trained checkpoint | `models/best_model.pt` at repo root (see checkpoint schema below) |
| Internet access on first build | API container downloads the HuggingFace model config for preprocessing |

### Checkpoint schema

The file at `MODEL_PATH` must be a PyTorch `.pt` saved with `torch.save` containing these keys:

```python
{
    "model_state_dict": dict,   # state dict of DinoV3Classifier
    "class_names":      list,   # e.g. ["benign", "malignant"]
    "model_name":       str,    # HuggingFace model ID used at training time
    "image_size":       int,    # e.g. 224
    "freeze_backbone":  bool,
}
```

This is the `best_model.pt` written automatically by `src/mse_mlops/train.py` after each
training run.

## Configuration

All settings are passed as environment variables. Override them in a `.env` file placed next
to `compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/best_model.pt` | Path inside the API container to the checkpoint |
| `DOCTOR_PASSWORD` | `doctor123` | Password for doctor-only UI tabs — **change in production** |

Example `.env`:
```
DOCTOR_PASSWORD=my_secure_password
MODEL_PATH=/models/best_model.pt
```

## Folder structure

```
serving/
├── AGENT.md            AI/developer guide for this subtree
├── Architecture.md     Component diagram, data flows, API reference
├── compose.yml  Two-service orchestration
├── api/                FastAPI inference API
│   ├── main.py         Route definitions and feedback store I/O
│   ├── model.py        Model loading, preprocessing, inference
│   ├── pyproject.toml  Dependencies (managed with uv)
│   └── Dockerfile
└── ui/
    ├── app.py          Streamlit 3-tab application
    ├── pyproject.toml  Dependencies (managed with uv)
    └── Dockerfile
```

Feedback data (JSONL + images) is stored in a named Docker volume (`feedback_data`) mounted
at `/feedback` inside the API container. It persists across container restarts.

## Running without Docker (development)

**API:**
```bash
cd serving/api
MODEL_PATH=../../models/best_model.pt uv run uvicorn main:app --reload --port 8000
```

**UI (separate terminal):**
```bash
cd serving/ui
API_URL=http://localhost:8000 DOCTOR_PASSWORD=doctor123 uv run streamlit run app.py --server.port 7777
```

## Further reading

- [`Architecture.md`](./Architecture.md) — component diagram, all data flows, full API reference
- [`AGENT.md`](./AGENT.md) — design decisions, extension guide, common pitfalls
- [`api/README.md`](./api/README.md) — API routes, model details, how to add endpoints
- [`ui/README.md`](./ui/README.md) — UI tabs, authentication, how to extend the UI
