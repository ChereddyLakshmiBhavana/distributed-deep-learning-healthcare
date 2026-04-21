# Distributed Engineering Version

This repository now includes a practical distributed-engineering layer in addition to the existing pneumonia detection workflow.

## What was added

1. Multi-GPU and multi-node training preparation via the distributed runtime summary and the real DDP trainer.
2. A SQLite-backed async job queue for prediction and aggregation tasks.
3. A federated learning coordinator that can register sites, store model updates, and run FedAvg aggregation.
4. Persistent audit logging for jobs, nodes, and federated updates.
5. Container deployment files for backend and frontend services.

## Backend endpoints

- `GET /distributed/info`
- `GET /distributed/jobs`
- `POST /distributed/jobs`
- `POST /distributed/jobs/process-next`
- `POST /predict/async`
- `GET /distributed/federated/status`
- `POST /distributed/federated/register`
- `POST /distributed/federated/update`
- `POST /distributed/federated/aggregate`

## Distributed worker

Run the worker from the backend folder to process queued jobs:

```powershell
python backend/distributed_worker.py
```

## Distributed training launcher

Inspect the runtime and launch real DDP training with:

```powershell
torchrun --standalone --nproc_per_node=<gpus> backend/distributed_ddp_train.py
```

The notebook also uses the same DDP-ready logic when distributed environment variables are present.

## Docker

Use the compose file at the repository root to run the backend and frontend as services.

## Quick Start

1. Start backend and frontend:

```powershell
docker compose up --build backend frontend
```

2. Start distributed worker (second terminal):

```powershell
docker compose up --build worker
```

3. Verify runtime:

```powershell
Invoke-WebRequest http://localhost:5000/distributed/info -UseBasicParsing
Invoke-WebRequest http://localhost:5000/distributed/federated/status -UseBasicParsing
```

4. Queue async prediction (example):

```json
POST /predict/async
{
	"image": "<base64>",
	"model": "random_forest_model",
	"priority": 1
}
```

5. Poll job status:

```powershell
Invoke-WebRequest http://localhost:5000/distributed/jobs/<job_id> -UseBasicParsing
```
