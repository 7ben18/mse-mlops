# MLOps Pipeline

This pipeline covers the full lifecycle of a machine learning model — from raw data ingestion to production serving and continuous retraining. It is built around skin cancer classification using the HAM10000 dataset from Kaggle.

The pipeline is divided into five stages:

1. **Data** — Ingest raw images, version datasets with DVC, and track code changes with Git
2. **Training** — Train and tune a CNN classifier using PyTorch and Ray Tune
3. **Experimentation** — Log runs and export the best model to ONNX, registered in MLFlow
4. **Serving** — Deploy via Docker, expose predictions through FastAPI, and surface them in a Streamlit UI
5. **Feedback Loop** — Monitor production performance, collect user corrections, curate new data, and feed it back into the pipeline

```mermaid
flowchart TD
    DS["Data Source\nKaggle"]
    RS["Raw Storage\nLocal / S3"]
    DVC["Data Versioning\nDVC"]
    GIT["Code Versioning\nGit"]
    TRAIN["Training\nPyTorch"]
    HPT["HP Tuning\nRay Tune"]
    EXP["Experiment Tracking\nMLFlow"]
    ME["Model Export\nONNX"]
    MR["Model Registry\nMLFlow"]
    DEP["Deployment\nDocker"]
    API["Inference API\nFastAPI"]
    UI["UI\nStreamlit"]
    MON["Monitoring\nMLFlow"]
    FS["Feedback Store"]
    CUR["Curation"]

    DS --> RS --> DVC
    GIT --> TRAIN
    DVC --> TRAIN
    TRAIN --> HPT
    TRAIN --> DVC
    HPT --> EXP
    HPT --> ME
    ME --> MR
    EXP -.-> MR
    MR --> DEP
    DEP --> API
    API --> UI
    API --> MON
    MON -.-> EXP
    UI --> FS
    FS --> CUR
    CUR --> DVC

    classDef data    fill:#006d77,stroke:#004e57,color:#fff
    classDef code    fill:#4a4e69,stroke:#2f3150,color:#fff
    classDef train   fill:#1d6a96,stroke:#0f4a6e,color:#fff
    classDef exp     fill:#6a0572,stroke:#45024d,color:#fff
    classDef serve   fill:#e07a5f,stroke:#b85c43,color:#fff
    classDef monitor fill:#f2cc8f,stroke:#c9a227,color:#222

    class DS,RS,DVC data
    class GIT code
    class TRAIN,HPT train
    class EXP,ME,MR exp
    class DEP,API,UI serve
    class MON,FS,CUR monitor
```

## Node Descriptions

- **Data Source (Kaggle)** — Raw skin cancer images from a public dataset
- **Raw Storage** — Unprocessed images stored locally or in the cloud
- **Data Versioning (DVC)** — Tracks and reproduces dataset snapshots
- **Code Versioning (Git)** — Version control for all training code
- **Training (PyTorch)** — Fine-tunes a CNN classifier on the dataset
- **HP Tuning (Ray Tune)** — Distributed hyperparameter optimization
- **Experiment Tracking (MLFlow)** — Logs runs, metrics, and artifacts
- **Model Export (ONNX)** — Serializes the model in a framework-agnostic format
- **Model Registry (MLFlow)** — Stores and stages versioned production models
- **Deployment (Docker)** — Containerizes the model for reproducible serving
- **Inference API (FastAPI)** — Serves real-time REST predictions
- **UI (Streamlit)** — Interactive frontend for end users
- **Monitoring (MLFlow)** — Tracks model performance in production
- **Feedback Store** — Collects and persists user corrections
- **Curation** — Labels and prepares corrected data for retraining
