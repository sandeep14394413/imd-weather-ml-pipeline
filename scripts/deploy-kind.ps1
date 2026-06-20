param(
    [string]$ClusterName = "kind",
    [string]$Namespace = "imd-weather"
)

$ErrorActionPreference = "Stop"
$ImageTag = Get-Date -Format "yyyyMMddHHmmss"

.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m src.train_baseline

docker build -t imd-weather-ml:$ImageTag .
kind load docker-image imd-weather-ml:$ImageTag --name $ClusterName

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/10-ml.yaml
kubectl -n $Namespace set image deployment/imd-weather-ml ml=imd-weather-ml:$ImageTag
kubectl -n $Namespace rollout status deployment/imd-weather-ml --timeout=180s

Write-Host "Port forward ML API with:"
Write-Host "kubectl -n $Namespace port-forward svc/imd-weather-ml 8010:8010"
