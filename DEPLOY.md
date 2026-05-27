# Deploy the dashboard

The FastAPI service is meant for Docker / container deploys (see `Dockerfile` and `docker-compose.yml`). The dashboard side ships easily to Streamlit Cloud.

## Streamlit Community Cloud (free, ~3 min)

1. [streamlit.io/cloud](https://streamlit.io/cloud) → **Sign in with GitHub** (use `nanettetada`).
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `nanettetada/microfinance-credit-risk`
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
4. **Deploy**. Build will train a small model on first run via `tests/conftest.py`'s fixture if no model exists.

## Notes

- Auto-rebuilds on every push to `main`.
- The dashboard reads the model directly via `src.predict`, so no separate API needs to be up.
- For the FastAPI service, use `docker-compose up` locally or push the Docker image to any container registry (Fly.io, Render, AWS App Runner, etc.).
