# LogStream - Processing Service 🧠

A Python-based microservice that consumes raw logs, parses them into a structured format, and enriches them with ML-driven insights.

This service is the core data processing engine of LogStream. It pulls raw log messages from an SQS queue, applies parsing logic to structure the data, and invokes ML models to generate valuable metadata before persisting the final, enriched record to the database.

---

### Key Features

- **Asynchronous Processing**: Consumes logs from an SQS queue to handle workloads efficiently.
- **Intelligent Enrichment**: Integrates with ML models for log clustering, anomaly detection, and severity prediction.
- **Structured Data**: Transforms unstructured text logs into clean, queryable JSON format.
- **Scalable**: Designed to run as a pool of workers on AWS Fargate, scaling with the ingestion rate.

### Tech Stack

- **Language**: Python
- **Framework**: FastAPI
- **Containerization**: Docker
- **Deployment**: AWS Fargate
- **Database**: PostgreSQL (via RDS)

---

### Getting Started

_Instructions for local setup, environment variables, and running the service will be added here._

### Processing Logic

_Details on the log parsing and ML enrichment pipeline will be added here._
