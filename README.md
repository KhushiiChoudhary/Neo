# Neo: Autonomous AutoML Agent

An autonomous ML engineer in your browser. Upload data, describe your goal in plain English, and Neo handles everything end-to-end: understanding the data, engineering features, running experiments, tuning models, and returning the best result with a full explanation.

---

## What it does

```
You: "Predict which customers will churn"

Neo: Analyzed your dataset. Target column: churn_flag (classification).

     Engineering 4 new features: tenure_per_product, spend_velocity...

     Tuning 4 models (25 Optuna trials each):
       LogisticRegression   AUC: 0.81
       RandomForest         AUC: 0.91  <- best
       XGBoost              AUC: 0.89
       LightGBM             AUC: 0.88

     Best model: RandomForest (AUC 0.91)
     Top drivers: tenure, total_spend, login_frequency

     Here's your model, inference package, and report.
```

---

## Features

- **5 data input methods**: upload CSV, paste URL, paste raw text, sample datasets, or database connection (SQLite / PostgreSQL)
- **Human-in-the-loop confirmation**: review and override the target column and problem type before any training starts
- **Data quality checks**: flags class imbalance, data leakage risk, high null columns, and near-constant features before training
- **Feature Engineering Agent**: GPT-4o suggests and creates new features from your existing columns
- **4 tuned models**: Logistic Regression / Ridge, Random Forest, XGBoost, LightGBM; each with 25 Optuna trials
- **Baseline benchmark**: always trains a naive baseline first so results are meaningful
- **SHAP explainability**: feature importance bar chart for every run
- **Confidence plot**: probability distribution (classification) or actual vs predicted scatter (regression)
- **Plain English report**: GPT-4o writes a business-readable summary of what won and why
- **MLflow experiment tracking**: every trial logged; one-click to open the full dashboard
- **Inference package download**: zip containing `best_model.pkl` + `predict.py` + `serve.py` (FastAPI) + `README`

---

## Architecture

```
Streamlit UI
     |
     v
LangGraph StateGraph
     |
     |-- Data Agent         profile -> GPT-4o target ID -> preprocessing
     |-- Feature Agent      GPT-4o feature suggestions -> pandas execution
     |-- Experiment Agent   Optuna x 4 models -> MLflow logging
     +-- Reporter Agent     SHAP -> confidence plot -> GPT-4o report -> inference zip
```

---

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph |
| LLM | OpenAI GPT-4o |
| ML | scikit-learn, XGBoost, LightGBM |
| Tuning | Optuna (Bayesian, 25 trials/model) |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| Deployment target | Streamlit Community Cloud |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/KhushiiChoudhary/Neo.git
cd Neo
```

**2. Create a virtual environment and install dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Add your OpenAI API key**
```bash
cp .env.example .env
# then edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

**4. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Using the app

1. **Pick a data source**: upload a CSV, paste a URL, use a built-in sample dataset, or connect a database
2. **Describe your goal** in plain English (e.g. "Predict survival", "Forecast house prices")
3. **Confirm the target column**: review what GPT-4o identified and override if needed
4. **Watch the agents work**: live status updates stream into the chat as each agent runs
5. **Review results**: SHAP plot, confidence chart, model comparison table, plain English report
6. **Download**: grab the model (`.pkl`) or the full inference package (`.zip` with FastAPI server)

---

## Project structure

```
Neo/
|-- app.py                    # Streamlit UI, 4-stage chat flow
|-- requirements.txt
|-- .env                      # OPENAI_API_KEY (not committed)
|-- .streamlit/
|   +-- config.toml           # Theme (light, brown accent)
|-- agents/
|   |-- data_agent.py         # Profile + GPT-4o target ID + preprocessing
|   |-- feature_agent.py      # GPT-4o feature engineering
|   |-- experiment_agent.py   # Optuna tuning for 4 models + baseline
|   +-- reporter_agent.py     # SHAP + confidence plot + GPT-4o report
|-- graph/
|   +-- pipeline.py           # LangGraph: Data -> Features -> Experiment -> Reporter
|-- utils/
|   |-- llm.py                # OpenAI wrapper (chat + chat_json)
|   |-- mlflow_tracker.py     # MLflow run logger
|   |-- data_loader.py        # 5 data input methods
|   |-- data_quality.py       # Pre-training data quality checks
|   +-- inference_generator.py# Generates predict.py + serve.py
+-- prompts/
    |-- data_agent.txt
    |-- feature_agent.txt
    |-- experiment_agent.txt
    +-- reporter_agent.txt
```

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Set `OPENAI_API_KEY` as a secret in the app settings
4. Deploy and get a public URL instantly

---

## Local MLflow dashboard

Every model run is automatically logged. To browse all experiments:

```bash
mlflow ui --port 5001
```

Then open [http://localhost:5001](http://localhost:5001). Or click the **"Open MLflow tracker"** button inside the app.
