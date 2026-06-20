# India Weather ML Pipeline

ML workflow for India temperature and weather forecasting using IMD historical and
real-time data.

## Goal

Train models from historical IMD data and expose predictions through an API used by
the weather app.

## Data sources

- IMD daily gridded maximum temperature data, 1.0 x 1.0 degree, 1951-2024.
- IMD daily gridded rainfall data, 0.25 x 0.25 degree, 1901-2024.
- IMD real-time max/min temperature data for daily refresh.

Some IMD datasets are binary grid files, so this repo keeps ingestion separate from
feature engineering. The initial baseline uses a seasonal climatology-style model,
then can be upgraded to LightGBM/XGBoost and sequence models.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_baseline
uvicorn src.serve_model:app --reload --port 8010
```

## Deploy to kind

Prerequisites:

- Docker
- kind
- kubectl pointed at your kind cluster

Manual deployment:

```powershell
.\scripts\deploy-kind.ps1 -ClusterName kind
kubectl -n imd-weather port-forward svc/imd-weather-ml 8010:8010
```

Then verify `http://127.0.0.1:8010/health`.

GitHub Actions deployment:

- Uses `.github/workflows/deploy-kind.yml`.
- Requires a self-hosted runner on the machine that can access your kind cluster.
- The workflow trains the latest baseline model before building the Docker image.

## Forecast honesty

Short-range forecasts can be evaluated as weather forecasts. Predictions many
months ahead should be treated as probabilistic climate outlooks, not exact daily
weather guarantees.
