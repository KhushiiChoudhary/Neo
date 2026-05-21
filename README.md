# Neo: Autonomous AutoML Agent

An autonomous ML engineer in your browser. Upload data, describe your goal in plain English, and Neo handles everything end-to-end: profiling the data, detecting quality issues and leakage, engineering new features, tuning four models with cross-validation, explaining what drove the result, and handing you a ready-to-deploy inference package.

---

## What it does

```
You: "Predict which customers will churn"

Neo: Analyzed your dataset. Target column: churn_flag (classification).

     Engineering 4 new features: tenure_per_product, spend_velocity...

     Tuning 4 models (25 Optuna trials each):
       LogisticRegression   AUC: 0.81  cv: 0.80 +/-0.02
       RandomForest         AUC: 0.91  cv: 0.90 +/-0.01  <- best
       XGBoost              AUC: 0.89  cv: 0.88 +/-0.02
       LightGBM             AUC: 0.88  cv: 0.87 +/-0.02

     Best model: RandomForest (AUC 0.91)
     Top drivers: tenure, total_spend, login_frequency

     Here is your model, inference package, and report.
```

---

## Features

**Data input**
- 5 input methods: upload CSV, paste a URL, paste raw text, built-in sample datasets, or a live database connection (SQLite / PostgreSQL)

**Pre-training analysis**
- Data quality checks: class imbalance, high null columns, near-constant features
- Leakage detection using both Pearson correlation and mutual information (catches numeric and categorical leakage)
- Class distribution chart and imbalance warning on the confirmation screen
- Human-in-the-loop confirmation: review and override the target column and problem type before any training starts

**Training pipeline**
- Feature Engineering Agent: GPT-5.4 suggests and creates new features from existing columns
- 4 tuned models: Logistic Regression / Ridge, Random Forest, XGBoost, LightGBM
- Each model gets 25 Optuna Bayesian trials + 5-fold cross-validation
- Naive baseline always trained first so improvements are meaningful
- Live progress: every agent step streams into the UI as it happens

**Results and explainability**
- Color-coded model leaderboard with CV mean and std
- SHAP feature importance chart
- Confusion matrix and ROC curve (classification)
- Residual analysis: residuals vs predicted + residual histogram (regression)
- Confidence distribution / actual-vs-predicted plot
- Plain English report written by the LLM
- Prediction sandbox: enter your own inputs and get a live prediction with confidence

**Downloads**
- Trained model (`.pkl`)
- Styled HTML report
- Inference package (`.zip`): `best_model.pkl` + `predict.py` + `serve.py` (FastAPI) + `README`
- MLflow experiment tracker (one-click launch)

---

## Architecture

```
Streamlit UI  (4-stage flow: Upload -> Confirm -> Training -> Results)
     |
     v
LangGraph StateGraph
     |
     |-- Data Agent         profile -> GPT-5.4 target ID -> preprocessing
     |-- Feature Agent      GPT-5.4 feature suggestions -> pandas execution
     |-- Experiment Agent   Optuna x 4 models + 5-fold CV -> MLflow logging
     +-- Reporter Agent     SHAP -> confusion matrix -> ROC -> residuals -> report -> zip
```

---

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph |
| LLM | OpenAI GPT-5.4 |
| ML | scikit-learn, XGBoost, LightGBM |
| Tuning | Optuna (Bayesian, 25 trials/model) |
| Validation | 5-fold cross-validation |
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
# edit .env and add your key:
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
3. **Review the confirmation screen**: check the target column, class distribution, and any data quality warnings
4. **Watch the agents work**: every step streams live into the status panel
5. **Explore results**: SHAP plot, confusion matrix, ROC curve, model leaderboard with CV scores, plain English report
6. **Try the prediction sandbox**: enter custom feature values and get a live prediction with confidence
7. **Download**: model `.pkl`, styled HTML report, or the full inference package `.zip`

---

## Project structure

```
Neo/
|-- app.py                    # Streamlit UI, 4-stage flow
|-- requirements.txt
|-- .env                      # OPENAI_API_KEY (not committed)
|-- .streamlit/
|   +-- config.toml           # Theme (light, brown accent)
|-- agents/
|   |-- data_agent.py         # Profile + GPT-5.4 target ID + preprocessing
|   |-- feature_agent.py      # GPT-5.4 feature engineering
|   |-- experiment_agent.py   # Optuna tuning + 5-fold CV + MLflow logging
|   +-- reporter_agent.py     # SHAP + confusion matrix + ROC + residuals + report
|-- graph/
|   +-- pipeline.py           # LangGraph: Data -> Features -> Experiment -> Reporter
|-- utils/
|   |-- llm.py                # OpenAI wrapper (chat + chat_json)
|   |-- mlflow_tracker.py     # MLflow run logger
|   |-- data_loader.py        # 5 data input methods
|   |-- data_quality.py       # Leakage detection + pre-training quality checks
|   +-- inference_generator.py# Generates predict.py + serve.py (FastAPI)
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
3. In app settings, add `OPENAI_API_KEY` under **Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy and get a public URL instantly

---

## Local MLflow dashboard

Every model run is automatically logged. To browse all experiments:

```bash
mlflow ui --port 5001
```

Then open [http://localhost:5001](http://localhost:5001), or click **Open MLflow tracker** inside the app.
