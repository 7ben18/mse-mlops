# serving/ui/

Streamlit web application for the melanoma skin cancer classifier. Three-tab UI serving two
audiences: patients (public, no login) and doctors (password-protected).

## Tabs

### Tab 1 — Skin Analysis (public)

Anyone can upload a skin lesion photo (JPG/PNG) and receive an AI prediction.

- Displays the image, a benign/malignant verdict, confidence score, and per-class probability bars.
- Includes a medical disclaimer.
- Calls `POST /predict` on the API; the prediction is automatically saved to the feedback store.

### Tab 2 — Review & Label (doctors only)

Shows all AI predictions that have not yet received a verified ground-truth label.

- Fetches the list via `GET /feedback` and filters for `label=null`.
- The doctor selects a label (benign / malignant) and submits via `POST /feedback`.
- Entry is updated in the feedback store and removed from the queue on next refresh.

### Tab 3 — Bulk Dataset Upload (doctors only)

Two upload modes:

**Single image** — upload one image with a label; calls `POST /upload-labeled`.

**ZIP + label sheet** — upload a `.zip` archive of images alongside a CSV or Excel file.
- Label sheet must have at minimum two columns: `filename` and `label`.
- `filename` must match the image filename inside the ZIP (not a path, just the base name).
- `label` must be `benign` or `malignant`.
- Unmatched images (not in the sheet) are listed and skipped.
- Each matched image is uploaded individually to `POST /upload-labeled` with a progress bar.

## Doctor authentication

Tabs 2 and 3 are gated behind a password prompt. The password is set via the `DOCTOR_PASSWORD`
environment variable (default: `doctor123`). Change this in production via a `.env` file.

Authentication state is stored in `st.session_state.doctor_auth`. It persists for the browser
session and is cleared by the "Log out" button. The API itself has no authentication — it
relies on network isolation (both containers share a Docker internal network).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | Base URL of the FastAPI API |
| `DOCTOR_PASSWORD` | `doctor123` | Password for doctor-only tabs |

Inside Docker Compose the default `API_URL` is `http://api:8000` (set in `compose.yml`).

## Dependencies

Managed with `uv` via `pyproject.toml`:

| Package | Purpose |
|---------|---------|
| `streamlit` | UI framework |
| `requests` | HTTP calls to the API |
| `pillow` | Image preview rendering |
| `pandas` | Parsing CSV/Excel label sheets |
| `openpyxl` | Excel (`.xlsx` / `.xls`) reading backend for pandas |

## Running locally

```bash
cd serving/ui
API_URL=http://localhost:8000 DOCTOR_PASSWORD=doctor123 uv run streamlit run app.py --server.port 7777
```

The API must be running before the UI starts (or Tab 1 will show an error on first prediction).

## Adding a new tab

1. Add a tab label to the `st.tabs([...])` call in `app.py`.
2. Write the tab body inside `with tabN:`.
3. For doctor-only tabs, add this guard at the top of the block:
   ```python
   if not st.session_state.doctor_auth:
       _doctor_login_form("unique_key")
   else:
       # tab content
   ```
4. Document the tab in the table above.

## Changing the password mechanism

The password check is in `_doctor_login_form()` in `app.py`. To switch to a more robust
mechanism (e.g. OAuth, a users table), replace that function. The guard pattern in each
doctor tab (`if not st.session_state.doctor_auth`) stays the same.
