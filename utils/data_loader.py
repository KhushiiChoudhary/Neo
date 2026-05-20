"""
All data ingestion methods. Each function returns a pd.DataFrame.
The rest of the app never knows how the data arrived.
"""
from __future__ import annotations

import io
import pandas as pd


# ── sample datasets (sklearn, no internet required) ───────────────────────────

SAMPLE_DATASETS: dict[str, dict] = {
    "Titanic: survival prediction (classification)": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "description": "891 passengers · 12 columns · predict Survived",
    },
    "Iris: flower species (classification)": {
        "loader": "iris",
        "description": "150 rows · 4 features · predict species",
    },
    "Wine: wine quality (classification)": {
        "loader": "wine",
        "description": "178 rows · 13 features · predict wine class",
    },
    "Breast Cancer: diagnosis (classification)": {
        "loader": "breast_cancer",
        "description": "569 rows · 30 features · predict malignant/benign",
    },
    "Diabetes: disease progression (regression)": {
        "loader": "diabetes",
        "description": "442 rows · 10 features · predict disease progression score",
    },
    "California Housing: home prices (regression)": {
        "loader": "california_housing",
        "description": "20,640 rows · 8 features · predict median house value",
    },
}


def load_sample(name: str) -> pd.DataFrame:
    entry = SAMPLE_DATASETS[name]

    if "url" in entry:
        return pd.read_csv(entry["url"])

    loader_key = entry["loader"]
    if loader_key == "iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
    elif loader_key == "wine":
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
    elif loader_key == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
    elif loader_key == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
    elif loader_key == "california_housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
    else:
        raise ValueError(f"Unknown sample dataset: {loader_key}")

    df = data.frame.copy()
    # rename target column to something readable
    target_map = {
        "iris": "species",
        "wine": "wine_class",
        "breast_cancer": "diagnosis",
        "diabetes": "disease_progression",
        "california_housing": "median_house_value",
    }
    if "target" in df.columns and loader_key in target_map:
        df = df.rename(columns={"target": target_map[loader_key]})
    return df


# ── URL ───────────────────────────────────────────────────────────────────────

def load_from_url(url: str) -> pd.DataFrame:
    """
    Load a CSV from any public URL.
    Converts common sharing URLs to direct download links automatically.
    """
    url = url.strip()

    # Google Sheets: convert share URL → export CSV URL
    if "docs.google.com/spreadsheets" in url and "/export" not in url:
        sheet_id = url.split("/d/")[1].split("/")[0]
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    # GitHub blob → raw
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    return pd.read_csv(url)


# ── paste ─────────────────────────────────────────────────────────────────────

def load_from_text(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text.strip()))


# ── database ──────────────────────────────────────────────────────────────────

def load_from_database(connection_string: str, query: str) -> pd.DataFrame:
    """
    Supports SQLite (sqlite:///path/to/file.db) and PostgreSQL
    (postgresql://user:pass@host:5432/dbname) via SQLAlchemy.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise ImportError("sqlalchemy is required for database connections.") from exc

    engine = create_engine(connection_string)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)
