# LogStream Processing Service

The LogStream processing service is the batch ML engine of the platform. It runs inside an ECS Fargate task, takes a log ID range from the ingestion service, clusters the matching logs, updates the database, detects anomalies, and marks the batch as completed.

## What This Service Does

This repository is responsible for turning newly ingested raw logs into enriched operational data.

At a high level, it:

- receives a batch range through ECS environment variables
- fetches the matching unprocessed logs from PostgreSQL
- generates semantic embeddings
- predicts a `cluster_id` for each log
- stores embeddings and updates `logs.cluster_id`
- updates pattern summaries
- checks for anomalous cluster volume
- creates or refreshes incidents
- marks the batch `COMPLETED`

## Where It Fits In The System

High-level pipeline:

`Log producer -> SQS -> ingestion pipeline -> logs table -> batch_order -> ECS task -> processing service`

The ingestion service creates a row in `batch_order`, launches an ECS task, and passes:

- `BATCH_ID`
- `START_LOG_ID`
- `END_LOG_ID`

This service uses those values to process exactly one batch.

## What The ECS Task Actually Runs

The Docker container entrypoint is defined in [Dockerfile](/d:/PERSONAL/Log_Stream App/logstream-processing-service/Dockerfile:37):

```dockerfile
CMD ["python", "scripts/run_incremental_batch.py"]
```

So the ECS task is simply this repository’s container starting `scripts/run_incremental_batch.py`.

## Batch Execution Flow

### 1. Read batch metadata

The script reads:

- `BATCH_ID`
- `START_LOG_ID`
- `END_LOG_ID`

from environment variables injected by ECS.

### 2. Load current model state

The batch runner loads the currently active production artifacts, including:

- clustering model
- feature pipeline
- semantic vector centroids
- volume anomaly model

This allows each batch to continue from the latest learned state.

### 3. Fetch the assigned logs

The script queries PostgreSQL for logs:

- between `START_LOG_ID` and `END_LOG_ID`
- where `level IN ('error', 'warning')`
- where `cluster_id IS NULL`

This means it only processes warning/error logs that have not yet been clustered.

### 4. Generate embeddings and cluster the logs

For each log:

- the message and parsed data are combined
- a text embedding is generated
- features are built
- the model predicts a `cluster_id`

### 5. Persist the results

For each processed log, the service:

- inserts a row into `log_embeddings`
- updates the corresponding row in `logs`

This is the point where a raw log becomes ML-enriched.

### 6. Update pattern metadata

After clustering, the service refreshes pattern-level data so repeated log behaviors can be summarized and queried later.

### 7. Detect anomalies and create incidents

The service then:

- counts how many logs landed in each cluster for the batch
- stores cluster volume history
- runs anomaly detection on that history
- creates or refreshes incidents for anomalous clusters

### 8. Mark the batch complete

At the end of a successful run, the script updates `batch_order` and sets:

- `status = 'COMPLETED'`
- `last_processed_timestamp = NOW()`

This is how the batch lifecycle is closed.

## Main Scripts

### `scripts/run_incremental_batch.py`

This is the production batch runner used by ECS.

Responsibilities:

- load batch context
- fetch logs
- cluster logs
- write results
- detect incidents
- mark the batch complete

### `scripts/run_training_batch.py`

This script is used for base training / model refresh work.

Responsibilities:

- train on a larger historical set
- build model artifacts
- prepare production-ready model state

### `scripts/validate_quality.py`

Utility script for checking clustering quality.

### `scripts/visualise_results.py`

Utility script for inspecting clustering output visually.

## Project Structure

```text
.
|-- Dockerfile
|-- requirements.txt
|-- scripts/
|   |-- run_incremental_batch.py
|   |-- run_training_batch.py
|   |-- validate_quality.py
|   `-- visualise_results.py
`-- src/
    |-- db/
    `-- ml/
```

Important code areas:

- [scripts/run_incremental_batch.py](/d:/PERSONAL/Log_Stream App/logstream-processing-service/scripts/run_incremental_batch.py:1): main ECS batch script
- [src/db/log_ops.py](/d:/PERSONAL/Log_Stream App/logstream-processing-service/src/db/log_ops.py:44): writes embeddings and updates `logs.cluster_id`
- [src/db/incident_ops.py](/d:/PERSONAL/Log_Stream App/logstream-processing-service/src/db/incident_ops.py:42): anomaly-driven incident creation
- [src/ml/](/d:/PERSONAL/Log_Stream App/logstream-processing-service/src/ml): ML models, pipeline, semantic grouping, anomaly logic

## Tech Stack

- Python 3.10
- Docker
- AWS ECS Fargate
- PostgreSQL
- SQLAlchemy
- pandas
- sentence-transformers
- river
- scikit-learn
- scipy
- PyTorch CPU build

## Data It Expects

This service expects the surrounding system to provide:

- a populated `logs` table
- a `batch_order` table
- model artifacts for incremental processing
- downstream tables such as `log_embeddings` and `incidents`

Important columns used by the batch runner:

- `logs.log_id`
- `logs.app_id`
- `logs.level`
- `logs.message`
- `logs.source`
- `logs.parsed_data`
- `logs.cluster_id`
- `batch_order.batchid`
- `batch_order.startlogid`
- `batch_order.endlogid`
- `batch_order.status`

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- access to the LogStream PostgreSQL database
- model files already available for incremental processing

### Install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run locally

Set environment variables first:

```bash
set BATCH_ID=1
set START_LOG_ID=1
set END_LOG_ID=500
python scripts/run_incremental_batch.py
```

### Train / refresh the model

```bash
python scripts/run_training_batch.py
```

### Build the Docker image

```bash
docker build -t logstream-processing-service .
```

## Configuration Notes

The ECS task passes:

- `BATCH_ID`
- `START_LOG_ID`
- `END_LOG_ID`

The code also depends on database connection settings and production model files.

One important implementation note: database settings are still hardcoded in [src/db/connection.py](/d:/PERSONAL/Log_Stream App/logstream-processing-service/src/db/connection.py:1). Moving those into environment variables would make the service cleaner and safer to deploy.

## Known Gaps

- Some environment-specific values are still hardcoded.
- ECS task-definition registration and deployment wiring live outside this repo.
- This repo contains both production batch logic and local utility/training scripts.

## README Scope

This README is meant to give a strong high-level understanding of the service:

- what it does
- how it is triggered
- what the ECS task runs
- how a batch moves from raw logs to `COMPLETED`

If the project grows, deeper schema docs, ML notes, and deployment runbooks should live in a separate `docs/` folder.
