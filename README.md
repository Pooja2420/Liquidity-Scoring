# Corporate Bond Liquidity Scoring & Market Impact Model

> ML-Driven Liquidity Assessment and Pre-Trade Cost Estimation for US Credit

---

## Overview

This project builds a machine learning-powered liquidity scoring system that assigns a composite **0–100 liquidity score** to each US corporate bond daily, and calibrates a **pre-trade market impact model** that estimates expected transaction costs before submitting an order.

The system uses **XGBoost gradient boosting** on 40+ engineered TRACE features (Roll measure, Amihud illiquidity, Kyle's lambda, inter-trade duration) to predict liquidity scores, validates against realized bid-ask spreads and implementation shortfall, and exposes results via a **FastAPI service** and **Streamlit dashboard**.

---

## Architecture

```
bond-liquidity-scoring/
├── config.py                        # Centralized settings (Pydantic)
├── run_pipeline.py                  # End-to-end 9-stage pipeline runner
├── app.py                           # Streamlit dashboard (5 tabs)
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── bond_reference.py        # Bond universe loader / generator
│   │   ├── trace_loader.py          # TRACE Enhanced loader / simulator
│   │   └── macro_factors.py         # FRED API macro factor downloader
│   │
│   ├── features/
│   │   ├── roll_measure.py          # Roll (1984) bid-ask spread estimator
│   │   ├── amihud.py                # Amihud (2002) illiquidity ratio
│   │   ├── price_impact.py          # Kyle (1985) lambda from signed flow
│   │   ├── trade_frequency.py       # Daily + 30d rolling trade statistics
│   │   ├── inter_trade_time.py      # Median / max inter-trade duration
│   │   └── feature_pipeline.py      # Full ~40-feature build + score target
│   │
│   ├── models/
│   │   ├── trainer.py               # XGBoost + LightGBM trainer + MLflow
│   │   ├── cross_validator.py       # Walk-forward TimeSeriesSplit CV
│   │   └── shap_explainer.py        # SHAP TreeExplainer + global importance
│   │
│   ├── impact/
│   │   ├── impact_calibrator.py     # OLS MI(q) = α·σ·(q/ADV)^β per CUSIP
│   │   ├── bayesian_shrinkage.py    # Shrink sparse CUSIPs toward sector prior
│   │   └── cost_estimator.py        # Pre-trade: bid-ask + impact + CI + time
│   │
│   ├── api/
│   │   ├── models.py                # Pydantic v2 request/response schemas
│   │   └── main.py                  # FastAPI endpoints
│   │
│   └── integration/
│       ├── tca_connector.py         # Connector → TCA Engine (Project 1)
│       └── rfq_connector.py         # Connector → RFQ Pricing Engine (Project 2)
│
└── tests/
    ├── test_features.py             # Roll, Amihud, Kyle-λ, trade freq, ITI
    ├── test_impact.py               # OLS calibration, shrinkage, cost math
    ├── test_models.py               # XGBoost/LightGBM training + CV
    └── test_api.py                  # All FastAPI endpoints
```

---

## Features Engineered from TRACE (~40 total)

| Category | Feature | Description |
|---|---|---|
| Trade Frequency | `trade_count_30d` | 30-day rolling avg daily trade count |
| Trade Frequency | `zero_trade_days_30d` | Fraction of days with no trades |
| Trade Size | `avg_trade_size_mm` | Mean trade size ($MM) |
| Trade Size | `pct_institutional` | % of trades ≥ $1MM |
| Bid-Ask Spread | `roll_bps` | Roll (1984) effective spread estimate |
| Illiquidity | `amihud_21d` | Amihud ratio — \|return\| / volume |
| Price Impact | `kyle_lambda_abs` | Kyle's λ from signed TRACE flow |
| Inter-Trade Time | `median_iti_hours` | Median time between trades (hours) |
| Bond Characteristics | `outstanding_mm`, `ttm_years`, `coupon`, `age_years` | Static bond metadata |
| Credit | `rating_ordinal`, `is_investment_grade` | Ordinal-encoded credit rating |
| Macro | `vix`, `tsy_10y`, `cdx_ig_bps`, `cdx_hy_bps` | Market-wide risk indicators |
| Macro | `lqd_volume_mm`, `hyg_volume_mm` | IG/HY ETF volume |

---

## ML Model

| Component | Detail |
|---|---|
| **Primary model** | XGBoost regressor → continuous 0–100 score |
| **Comparison model** | LightGBM regressor |
| **Classifier** | XGBoost 3-class → Low / Medium / High bucket |
| **Target variable** | Composite score from Roll spread, Amihud, trade frequency, Kyle-λ |
| **Training period** | 2020–2021 (fast mode) / 2015–2021 (full) |
| **Test period** | 2022–2023 (out-of-sample) |
| **Validation** | Walk-forward TimeSeriesSplit (5 folds, 30-day gap) |
| **Explainability** | SHAP TreeExplainer — global importance + per-bond attribution |

---

## Market Impact Model

Calibrates a power-law impact function per CUSIP from TRACE data:

```
MI(q) = α · σ · (q / ADV)^β
```

| Parameter | Meaning |
|---|---|
| `q` | Trade size ($MM par) |
| `ADV` | Average daily volume for the bond |
| `σ` | Daily price volatility |
| `α` | Impact coefficient (fitted via OLS) |
| `β` | Concavity exponent (0.5 = square-root law) |

For bonds with sparse TRACE history, `α` and `β` are shrunk toward a **sector × rating-bucket prior** using Bayesian shrinkage.

---

## Pre-Trade Cost Estimate Output

Given `(CUSIP, size $MM, side)`, the system returns:

| Output | Description |
|---|---|
| `bid_ask_cost_bps` | Expected half-spread cost (bps) |
| `market_impact_bps` | Expected price impact (bps) |
| `total_cost_bps` | Total expected TCA cost (bps) |
| `ci_lower/upper_90_bps` | 90% confidence interval |
| `liquidity_score` | 0–100 composite ML score |
| `liquidity_bucket` | Low / Medium / High with probabilities |
| `est_execution_hours` | Estimated time to hit a cost budget |

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone <repo-url>
cd "Liquidity Scoring"

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Add FRED_API_KEY for real macro data (free at fred.stlouisfed.org)
```

### 3. Run the pipeline

```bash
# Fast mode (~3 min, 2020–2023 data)
python3 run_pipeline.py --no-mlflow --fast

# Full mode (~15 min, 2015–2023 data, with MLflow tracking)
python3 run_pipeline.py
```

Pipeline stages:
```
[1/9] Load bond universe
[2/9] Load TRACE data
[3/9] Load macro factors
[4/9] Build feature store       → data/processed/features.parquet
[5/9] Cross-validation          → data/processed/cv_results.csv
[6/9] Train models              → models/xgb_regressor.pkl, lgb_regressor.pkl
[7/9] Calibrate impact models   → data/processed/impact_params.parquet
[8/9] Compute SHAP importance   → models/shap_importance.csv
[9/9] Done
```

### 4. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs at: **http://localhost:8000/docs**

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

### 6. Run tests

```bash
python3 -m pytest -v
# 87 tests across features, impact model, ML models, and API
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status and model info |
| `GET` | `/score/{cusip}` | Liquidity score for a single CUSIP |
| `POST` | `/score` | Batch scores for up to 500 CUSIPs |
| `GET` | `/pretrade/{cusip}/{size_mm}/{side}` | Pre-trade cost estimate |
| `POST` | `/pretrade` | Pre-trade estimate via request body |
| `GET` | `/universe/map` | Aggregated heatmap by rating × sector |
| `GET` | `/impact/{cusip}` | Calibrated impact params for a CUSIP |

**Example:**
```bash
# Single bond score
curl http://localhost:8000/score/CUSIP000001

# Pre-trade estimate: buy $5MM of CUSIP000001
curl http://localhost:8000/pretrade/CUSIP000001/5.0/B
```

---

## Streamlit Dashboard (5 Tabs)

| Tab | Description |
|---|---|
| 🔍 **Bond Screener** | Filter bonds by rating, sector, score range; color-coded table + distribution chart |
| 🧠 **Score Breakdown** | SHAP global feature importance bar chart + per-bond feature values |
| 💰 **Pre-Trade Estimator** | Interactive cost estimator with breakdown chart + bucket probabilities |
| 📉 **Impact Calibration** | α/β distributions, R² vs. trade count scatter, full parameter table |
| 🗺️ **Universe Heatmap** | Rating × sector heatmap, bond count heatmap, outstanding treemap |

---

## Technology Stack

| Component | Technology |
|---|---|
| ML Framework | XGBoost 2.x, LightGBM 4.x, scikit-learn |
| Feature Engineering | Pandas, NumPy, SciPy, Statsmodels |
| Explainability | SHAP (TreeExplainer) |
| Feature Store | DuckDB |
| Model Registry | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data Sources | FINRA TRACE Enhanced, FRED API |
| Testing | pytest, pytest-asyncio (87 tests) |

---

## VS Code Setup

1. Open folder: `File → Open Folder → Liquidity Scoring`
2. Select interpreter: `Cmd+Shift+P` → **Python: Select Interpreter** → pick `.venv`
3. Add `.vscode/launch.json` for F5 launch configs (pipeline / API / Streamlit)

---

## Resume Bullet

> Developed an ML-based Corporate Bond Liquidity Scoring system using XGBoost on 40+ TRACE-engineered features (Roll measure, Amihud illiquidity, inter-trade duration, price impact) to predict composite 0–100 liquidity scores across 50,000 US corporate bond CUSIPs — calibrating bond-specific pre-trade market impact models with Bayesian shrinkage and deploying a FastAPI pre-trade cost estimation service with SHAP-based score explainability.
