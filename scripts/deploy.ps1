param(
    [string]$ComposeFile = "docker-compose.yml"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path ".env")) {
    if (-not (Test-Path ".env.template")) {
        throw "Missing .env.template"
    }
    Copy-Item ".env.template" ".env"
    Write-Host "Created .env from .env.template. Update it before real production use if needed."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed or not on PATH. Install Docker Desktop first."
}

Write-Host "Starting Docker Compose stack..."
docker compose -f $ComposeFile up --build -d

Write-Host "Waiting for backend health..."
for ($i = 1; $i -le 60; $i++) {
    try {
        $health = Invoke-RestMethod "http://localhost:5000/health" -TimeoutSec 3
        if ($health.status -eq "healthy") {
            Write-Host "Backend is healthy."
            break
        }
    } catch {
        Start-Sleep -Seconds 2
        continue
    }
    Start-Sleep -Seconds 2
}

if (-not $health -or $health.status -ne "healthy") {
    throw "Backend did not become healthy in time."
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $python) {
    Write-Host "Running smoke test helper..."
    & $python "scripts\ci_smoke_test.py"
} else {
    Write-Host "Skipping smoke test helper because .venv was not found."
}

Write-Host "Deployment helper completed successfully."