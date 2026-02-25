# Corporate Bond Liquidity Scoring & Market Impact Model

> ML-Driven Liquidity Assessment and Pre-Trade Cost Estimation for US Credit

ML-powered corporate bond liquidity scoring system that engineers 40+ features from FINRA TRACE data (Roll spread, Amihud illiquidity, Kyle's lambda, inter-trade duration) to train XGBoost models predicting composite 0–100 liquidity scores. Includes a calibrated pre-trade market impact model with Bayesian shrinkage, a FastAPI scoring service, and a 5-tab Streamlit dashboard — with 87 passing tests.

---

## Quick Start

```bash
# 1. Clone & set up
git clone <repo-url>
cd "Liquidity Scoring"
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) add FRED API key for real macro data
cp .env.example .env

# 3. Run the pipeline
python3 run_pipeline.py --no-mlflow --fast   # ~3 min
python3 run_pipeline.py                       # full 2015–2023 run

# 4. Serve the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Launch the dashboard
streamlit run app.py

# 6. Run tests
python3 -m pytest -v
```

---

## Project Structure

```
├── config.py                    # Centralized settings (Pydantic)
├── run_pipeline.py              # 9-stage pipeline runner
├── app.py                       # Streamlit dashboard (5 tabs)
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── bond_reference.py    # Bond universe loader / generator
│   │   ├── trace_loader.py      # TRACE Enhanced loader / simulator
│   │   └── macro_factors.py     # FRED API macro factor downloader
│   ├── features/
│   │   ├── roll_measure.py      # Roll (1984) bid-ask spread estimator
│   │   ├── amihud.py            # Amihud (2002) illiquidity ratio
│   │   ├── price_impact.py      # Kyle (1985) lambda from signed flow
│   │   ├── trade_frequency.py   # Daily + 30d rolling trade statistics
│   │   ├── inter_trade_time.py  # Median / max inter-trade duration
│   │   └── feature_pipeline.py  # Full ~40-feature build + score target
│   ├── models/
│   │   ├── trainer.py           # XGBoost + LightGBM trainer + MLflow
│   │   ├── cross_validator.py   # Walk-forward TimeSeriesSplit CV
│   │   └── shap_explainer.py    # SHAP TreeExplainer + global importance
│   ├── impact/
│   │   ├── impact_calibrator.py # OLS MI(q) = α·σ·(q/ADV)^β per CUSIP
│   │   ├── bayesian_shrinkage.py# Shrink sparse CUSIPs toward sector prior
│   │   └── cost_estimator.py    # Pre-trade: bid-ask + impact + CI + time
│   ├── api/
│   │   ├── models.py            # Pydantic v2 request/response schemas
│   │   └── main.py              # FastAPI endpoints
│   └── integration/
│       ├── tca_connector.py     # Connector → TCA Engine
│       └── rfq_connector.py     # Connector → RFQ Pricing Engine
│
└── tests/
    ├── test_features.py
    ├── test_impact.py
    ├── test_models.py
    └── test_api.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status |
| `GET` | `/score/{cusip}` | Liquidity score for a single CUSIP |
| `POST` | `/score` | Batch scores for up to 500 CUSIPs |
| `GET` | `/pretrade/{cusip}/{size_mm}/{side}` | Pre-trade cost estimate |
| `POST` | `/pretrade` | Pre-trade estimate via request body |
| `GET` | `/universe/map` | Aggregated heatmap by rating × sector |
| `GET` | `/impact/{cusip}` | Calibrated impact params for a CUSIP |

Interactive docs: **http://localhost:8000/docs**

```bash
# Examples
curl http://localhost:8000/score/CUSIP000001
curl http://localhost:8000/pretrade/CUSIP000001/5.0/B
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML | XGBoost 2.x, LightGBM 4.x, scikit-learn |
| Features | Pandas, NumPy, SciPy, Statsmodels |
| Explainability | SHAP (TreeExplainer) |
| Feature Store | DuckDB |
| Model Registry | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Testing | pytest, pytest-asyncio (87 tests) |
