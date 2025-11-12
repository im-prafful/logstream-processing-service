# LogStream - Processing Service 🧠

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in_development-orange.svg)

A Python-based microservice that consumes raw logs, parses them into a structured format, and enriches them with ML-driven insights.

This service is the core data processing engine of LogStream. It is designed to handle dynamic, streaming data by using an **online machine learning pipeline**. It can load new log batches (from S3 or a DB query), update its clustering model incrementally, and save the evolved model state for the next run.

---

## 🚀 Key Features

- **Asynchronous Processing:** Designed to consume logs in batches (e.g., from S3 or DB queries).
- **Dynamic ML Enrichment:** Uses `river` and `DenStream` to perform density-based clustering that evolves as new log patterns emerge.
- **Evolving Model:** The ML model state is saved after each run, allowing it to "remember" past data and adapt to new patterns incrementally.
- **Structured Data:** Transforms unstructured text logs into clean, queryable JSON format.
- **Scalable:** Designed to run as a containerized script on AWS Fargate, scaling with the ingestion rate.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Framework (Future):** FastAPI
- **Containerization:** Docker
- **Deployment:** AWS Fargate
- **Database:** PostgreSQL (via RDS)
- **Streaming ML:** `river` (using `DenStream`)
- **Text Embedding:** `sentence-transformers`
- **Database ORM:** `sqlalchemy`
- **Data Handling:** `pandas`

---

## 🏗️ Architecture & Data Flow

This service operates in scheduled batches to simulate a stream:

1.  **Load:** A scheduled job (e.g., cron) triggers the `run_incremental_batch.py` script.
2.  **State Load:** The script loads the "warm" `river_pipeline.pkl` and `denstream_model.pkl` from the `/models` directory.
3.  **Fetch New Data:** It queries the PostgreSQL DB for all logs where `cluster_id IS NULL`.
4.  **Process Stream (One-by-One):**
    - For each new log, it performs feature extraction (text embedding, JSON flattening).
    - The `river` pipeline standardizes and one-hot-encodes the features.
    - The `DenStream` model learns from the log (`learn_one`) and assigns a cluster ID (`predict_one`).
5.  **Update DB:** The script bulk-updates the database, writing the new `cluster_id` for all processed logs.
6.  **State Save:** The script saves the _newly evolved_ pipeline and model files back to `/models`, ready for the next run.

---

## 🧠 ML Processing Pipeline

The model's success depends on a robust, _streaming-compatible_ pre-processing pipeline.

| Input Attribute                  | Pre-processing                   | Why It's Ideal (The Analogy)                                                                                                                                                                     |
| :------------------------------- | :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`message`**                    | **Text Embedding**               | We convert text to a 384-dim numeric vector. This acts like a **GPS coordinate for meaning**, allowing the model to understand semantic similarity.                                              |
| **Vector (`vec_0`...`vec_383`)** | **Incremental Standardization**  | `river.preprocessing.StandardScaler` scales all numbers to the same range. It's a **live accountant** that updates its average with each log, unlike a batch scaler that needs all data at once. |
| **`level`**, **`source`**        | **Incremental One-Hot Encoding** | `river.preprocessing.OneHotEncoder` creates **"light switches"** (e.g., `is_error`, `is_api_backend`). It's non-biased and can _add new switches_ if a new `source` appears.                     |
| **`parsed_data`**                | **JSON Flattening**              | We extract keys like `method` _before_ the pipeline. This is like **unlocking a treasure chest** so the pipeline can see the individual features inside.                                         |

---

# All Python dependencies

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Poetry (recommended) or `pip`
- Docker (for containerization)
- Access to a running PostgreSQL database

### Local Installation

1.  **Clone the repository:**

    ```bash
    git clone <https://github.com/im-prafful/logstream-processing-service.git>
    cd logstream_processing_service
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory (and add `.env` to your `.gitignore`!).

    ```ini
    # .env
    DB_USER="<user_name>"
    DB_PASS="<db_password>"
    DB_NAME="<db_name>"
    DB_HOST="localhost"
    DB_PORT="5433"
    ```

4.  **Add `cluster_id` to your `logs` table:**
    You must add a column for storing the results.
    ```sql
    ALTER TABLE logs ADD COLUMN cluster_id INT;
    ```

---

## ⚙️ Usage

There are two main scripts to run this service.

### 1. Initial Model Training (Run Once)

You must run this script **one time** to create and save the very first version of your model. It will process an initial batch of logs (i.e., the first 10,000) and save `denstream_model.pkl` and `river_pipeline.pkl`.

```bash
python scripts/run_training_batch.py
```
