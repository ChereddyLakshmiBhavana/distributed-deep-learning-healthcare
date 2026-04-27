# Start containers for local deployment testing
param(
    [string]$ComposeFile = "docker-compose.yml"
)

Write-Host "Building and starting containers (detached)..."
docker compose -f $ComposeFile up --build -d

Write-Host "Waiting for backend /health..."
for ($i=0; $i -lt 60; $i++) {
    try {
        $r = Invoke-RestMethod http://localhost:5000/health -Method Get -TimeoutSec 2
        Write-Host "/health OK"; break
    } catch { Start-Sleep -Seconds 1 }
}

Write-Host "To run smoke tests use: python scripts/ci_smoke_test.py"