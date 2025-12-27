# Data Pipeline & AI Engineering (Build & Test)

## Overview
This project implements a scalable Data Engineering and Machine Learning pipeline using PySpark, Apache Iceberg, and MLflow.

The goal is to simulate a real-world scenario where data from two different corporate sources (Supply Chain and Financials) is ingested, entity-resolved (deduplicated), and harmonized into a "Golden Record" dataset. This data is stored in an Apache Iceberg table supporting transactional updates (Upserts) and is subsequently used to train a Logistic Regression model to predict corporate profitability.

This repository is designed to run entirely locally or within a CI/CD environment (like GitHub Actions) without requiring paid cloud accounts (AWS/GCP/Azure).

## Architecture
The pipeline follows these distinct stages:

1. **Ingestion:** Reads raw CSV data representing two simulated sources.
2. **Entity Resolution:** Uses a cleaning heuristic (normalizing names, removing suffixes like "Inc" or "LLC") to identify and merge records belonging to the same company.
3. **Storage (Lakehouse):** Upserts the harmonized data into an Apache Iceberg table. This allows us to update existing records (e.g., new revenue figures) while inserting new ones using the MERGE INTO command.
4. **Machine Learning:** Reads the clean data from Iceberg and trains a Spark ML model.
5. **Model Tracking:** Logs the model artifacts and performance metrics (AUC) to a local MLflow registry.

## Technologies Used
- **Language:** Python 3.9+
- **Processing:** Apache Spark 3.5.0
- **Storage Format:** Apache Iceberg 1.4.3 (configured with a local Hadoop Catalog)
- **Model Registry:** MLflow
- **Testing:** Pytest (Unit tests) and GitHub Actions (CI/CD)

## Project Structure
- `data_generator.py`: Generates 12,000+ synthetic records using the Faker library.
- `etl_ml_pipeline.py`: The main script that runs the ETL, Iceberg Upsert, and ML Training.
- `tests/`: Contains unit tests for the entity resolution logic.
- `warehouse/`: The local directory acting as the Iceberg data lake.
- `mlruns/`: The local directory where MLflow stores model metrics.
- `.github/workflows/ci_cd.yml`: Configuration for the automated build pipeline.

## Setup and Execution

### Prerequisites
1. **Java 17**: This is strictly required for Spark 3.5 and the Iceberg runtime to function correctly.
2. **Python 3.9** or higher.

### 1. Installation
Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate Data
Run the generator script to create the raw CSV files in the data_source folder. This will create overlapping records to test the deduplication logic.

```bash
python data_generator.py
```

### 3. Run the Pipeline
Execute the main script. On the first run, Spark will automatically download the necessary Iceberg JAR files.

```bash
python etl_ml_pipeline.py
```

If successful, you will see output indicating the number of records stored in Iceberg and the final AUC score of the trained model.

### 4. Run Tests
To verify the entity resolution logic works as expected:

```bash
pytest tests/
```


## Implementation Notes
### Local Warehouse: 
The Iceberg catalog is configured as type=hadoop pointing to the local warehouse/ directory. This simulates S3/Cloud storage behavior on a local disk.
### Memory Optimization: 
The Iceberg write operation is performed without partitioning by unique ID. Partitioning by high-cardinality columns on small local machines (like GitHub Codespaces) causes OutOfMemory errors due to the number of open file buffers.
### Java Compatibility: 
Recent environments often default to Java 21, which is incompatible with the current Spark/Hadoop security manager. The CI/CD pipeline and local instructions strictly enforce Java 17.