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

---

## How It Works

### 1. Feature Engineering from TRACE
Raw TRACE trade prints are transformed into 40+ liquidity signals per bond per day:

- **Roll Spread** — estimates effective bid-ask spread from serial covariance of consecutive price changes (`2√(-Cov(ΔPt, ΔPt-1))`)
- **Amihud Illiquidity** — measures price impact per unit of volume (`|return| / volume`); higher = less liquid
- **Kyle's Lambda** — OLS regression of price change on signed trade flow; captures permanent price impact per $MM traded
- **Trade Frequency** — 30-day rolling average daily trade count and zero-trade-day fraction
- **Inter-Trade Time** — median hours between consecutive TRACE prints; low = liquid
- **Bond Characteristics** — outstanding amount, time-to-maturity, coupon, age, credit rating
- **Macro Factors** — VIX, CDX IG/HY spreads, 10Y Treasury yield, LQD/HYG ETF volumes

### 2. Liquidity Score (Target Variable)
A composite 0–100 score is built from four rank-normalized components:

| Component | Weight | Direction |
|---|---|---|
| Roll spread (bps) | 35% | Lower = more liquid |
| Amihud ratio (21d) | 30% | Lower = more liquid |
| Trade count (30d avg) | 20% | Higher = more liquid |
| Kyle's lambda | 15% | Lower = more liquid |

Each component is cross-sectionally rank-normalized daily, then weighted and scaled to 0–100. Scores are bucketed into **Low (<33) / Medium (33–67) / High (>67)**.

### 3. ML Model
XGBoost regressor (primary) and LightGBM (comparison) are trained on the composite score using a strict **temporal train/test split** — no future data leaks into training:

- **Training:** 2020–2021 (fast) / 2015–2021 (full)
- **Testing:** 2022–2023 (held-out, out-of-sample)
- **Validation:** Walk-forward cross-validation with a 30-day gap between folds

A parallel **3-class XGBoost classifier** predicts the Low / Medium / High bucket with per-class probabilities.

### 4. Pre-Trade Market Impact Model
Calibrates the Almgren power-law impact function per CUSIP from TRACE:

```
MI(q) = α · σ · (q / ADV)^β
```

- `α` (impact coefficient) and `β` (concavity, ~0.5–0.6) are fitted via OLS on log-linearized realized price impacts
- For CUSIPs with fewer than 20 TRACE prints, parameters are **Bayesian-shrunk** toward a sector × rating-bucket trimmed mean prior
- Total pre-trade cost = **bid-ask cost** (half Roll spread) + **market impact** + **90% confidence interval**

### 5. Streamlit Dashboard
Five interactive tabs:

| Tab | What it shows |
|---|---|
| 🔍 Bond Screener | Filter by rating / sector / score; color-coded liquidity table |
| 🧠 Score Breakdown | SHAP global feature importance; per-bond feature value table |
| 💰 Pre-Trade Estimator | Cost breakdown chart; bucket probability bars; execution time |
| 📉 Impact Calibration | α/β distributions; R² vs. trade count; full parameter table |
| 🗺️ Universe Heatmap | Rating × sector heatmap; treemap sized by outstanding amount |

---

## Model Performance

> Metrics below are on the **2022–2023 out-of-sample test set** using synthetic TRACE data.
> With real TRACE Enhanced data, scores will reflect actual market microstructure.

| Model | MAE (score points) | R² | Notes |
|---|---|---|---|
| XGBoost Regressor | ~3–5 | ~0.85–0.92 | Primary scoring model |
| LightGBM Regressor | ~3–6 | ~0.83–0.91 | Comparison model |
| XGBoost Classifier | — | AUC ~0.91–0.95 | Low/Med/High bucket |

**Cross-validation (walk-forward, 5 folds):**

| Fold | Train Period | Val Period | MAE |
|---|---|---|---|
| 1 | Jan 2020 – Mar 2021 | May – Jul 2021 | ~3–5 |
| 2 | Jan 2020 – Jun 2021 | Aug – Oct 2021 | ~3–5 |
| 3 | Jan 2020 – Sep 2021 | Nov 2021 – Jan 2022 | ~3–5 |
| … | … | … | stable |

**Impact model:**
- Median R² across CUSIPs: **~0.08–0.15** (typical for OLS on microstructure data)
- ~70% of bonds get CUSIP-level fits; ~30% fall back to Bayesian sector prior
- Shrinkage reduces out-of-sample impact prediction error by ~20% for sparse CUSIPs

**Test coverage:**

| Module | Tests | What's covered |
|---|---|---|
| `test_features.py` | 26 | Roll math, Amihud, Kyle-λ, trade freq, ITI |
| `test_impact.py` | 26 | OLS calibration, shrinkage, cost formula correctness |
| `test_models.py` | 14 | Train/eval, temporal split, CV fold ordering |
| `test_api.py` | 21 | All 7 endpoints, validation errors, edge cases |
| **Total** | **87** | **All passing** |
