# MLflow Integration for Model Tracking

This guide explains how to use MLflow to track your model's performance and efficiency metrics.

## Overview

MLflow is integrated into the training and serving pipeline to automatically track:
- **Training metrics**: Training time, dataset statistics, memory usage, CPU usage
- **Model parameters**: Model type, number of states, lookup table size
- **Inference performance**: Prediction time per request, throughput
- **System resources**: Memory and CPU consumption during training

## Starting MLflow UI

### Local Development

```powershell
# Install MLflow (already in requirements.txt)
pip install -r requirements.txt

# Start MLflow UI
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Docker/Kubernetes Deployment

The MLflow server can be deployed as a service in Kubernetes:

```bash
kubectl apply -f k8s/20-mlflow.yaml
kubectl -n imd-weather port-forward svc/mlflow 5000:5000
```

Then access at http://localhost:5000

## Key Metrics Tracked

### Training Metrics
- **training_time_seconds**: Total time to train the model
- **memory_used_mb**: Memory consumed during training
- **memory_percent**: Percentage of system memory used
- **cpu_percent**: CPU usage during training
- **mean_tmax, std_tmax, min_tmax, max_tmax**: Data statistics

### Model Parameters
- **model_type**: Type of model (baseline, climatology, etc.)
- **target_variable**: What the model predicts (tmax_c)
- **num_states**: Number of geographic states in training data
- **training_data_rows**: Total rows in training dataset
- **num_lookup_entries**: Size of lookup table for predictions

### Inference Metrics (Logged per prediction)
- **inference_time_ms**: Time taken for a single prediction in milliseconds
- **forecast_state**: State for which prediction was made
- **forecast_type**: Type of forecast (short-range, extended-range, climate-outlook)

## Accessing Metrics Programmatically

### From the API

Get the latest model metrics via the `/metrics` endpoint:

```bash
curl http://127.0.0.1:8010/metrics
```

Response example:
```json
{
  "status": "ok",
  "latest_run_id": "abc123def456",
  "parameters": {
    "model_type": "seasonal_climatology",
    "num_states": "5",
    "training_data_rows": "73050"
  },
  "metrics": {
    "training_time_seconds": 2.34,
    "memory_used_mb": 456.78,
    "mean_tmax": 25.43,
    "inference_time_ms": 0.45
  },
  "tags": {
    "model_stage": "baseline",
    "data_source": "IMD"
  }
}
```

### From Python

```python
import mlflow

# Search for all training runs
runs = mlflow.search_runs()

# Get latest run details
latest_run = runs.iloc[0]
print(f"Training time: {latest_run.data.metrics['training_time_seconds']}s")
print(f"Memory used: {latest_run.data.metrics['memory_used_mb']}MB")
```

## Comparing Model Versions

MLflow automatically tracks all training runs. Use the UI to:
1. Compare metrics across different runs
2. Identify best model based on metrics
3. Track model improvements over time
4. Export models for production deployment

## CI/CD Integration

The GitHub Actions workflow automatically logs metrics when:
- `python -m src.train_baseline` runs → Training metrics logged
- `/predict` endpoint is called → Inference metrics logged

No additional configuration needed—metrics are automatically captured.

## Performance Monitoring

Monitor real-time inference performance:

```bash
# In one terminal, start the API
uvicorn src.serve_model:app --reload --port 8010

# In another, start MLflow UI
mlflow ui

# Make predictions
curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Delhi",
    "district": "Central Delhi",
    "target_date": "2026-07-01"
  }'

# Check metrics
curl http://127.0.0.1:8010/metrics
```

Each prediction is logged to MLflow, allowing you to track:
- Average inference time
- Performance trends
- Resource usage patterns
- Prediction distribution by state/region

## Next Steps

1. **Deploy MLflow Server**: Set up persistent MLflow tracking server for production
2. **Model Registry**: Use MLflow Model Registry to manage model versions and promotions
3. **Alerts**: Set up alerts on inference time or accuracy degradation
4. **Dashboards**: Create custom MLflow dashboards for stakeholder reporting
