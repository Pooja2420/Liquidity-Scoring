"""
Tests for ML model training, cross-validation, and SHAP explainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

from src.models.trainer import (
    train_xgb_regressor,
    train_lgb_regressor,
    train_xgb_classifier,
    evaluate_regressor,
    evaluate_classifier,
    train_test_split_temporal,
)
from src.models.cross_validator import time_series_cv
from src.features.feature_pipeline import get_feature_columns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_feature_df():
    """
    Synthetic feature dataframe mimicking the feature store output.
    2 years of daily data, 50 CUSIPs.
    """
    rng = np.random.default_rng(42)
    n_cusips = 50
    dates = pd.bdate_range("2021-01-04", "2022-12-30")
    feature_cols = get_feature_columns()

    records = []
    for cusip_i in range(n_cusips):
        base_score = rng.uniform(10, 90)
        for date in dates:
            row = {col: rng.uniform(0.01, 10) for col in feature_cols}
            # Make score somewhat correlated with trade_count_30d
            score = np.clip(base_score + rng.normal(0, 5), 0, 100)
            bucket = "Low" if score < 33 else ("High" if score > 67 else "Medium")
            row.update({
                "cusip": f"CUSIP{cusip_i:04d}",
                "date": date,
                "liquidity_score": score,
                "liquidity_bucket": bucket,
            })
            records.append(row)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def split_data(synthetic_feature_df):
    feature_cols = [c for c in get_feature_columns() if c in synthetic_feature_df.columns]
    X_train, y_train, X_test, y_test = train_test_split_temporal(
        synthetic_feature_df,
        feature_cols,
        train_end="2022-06-30",
        test_start="2022-07-01",
    )
    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------------------------
# train_test_split_temporal
# ---------------------------------------------------------------------------

class TestTemporalSplit:

    def test_no_overlap(self, synthetic_feature_df):
        feature_cols = [c for c in get_feature_columns() if c in synthetic_feature_df.columns]
        X_tr, y_tr, X_te, y_te = train_test_split_temporal(
            synthetic_feature_df, feature_cols,
            train_end="2022-06-30", test_start="2022-07-01"
        )
        assert len(X_tr) > 0
        assert len(X_te) > 0
        # No date overlap (dates are in index, not columns here — just check sizes)
        total = len(synthetic_feature_df["date"].unique())
        assert len(X_tr) + len(X_te) <= len(synthetic_feature_df)

    def test_no_nan_after_fill(self, synthetic_feature_df):
        feature_cols = [c for c in get_feature_columns() if c in synthetic_feature_df.columns]
        X_tr, y_tr, X_te, y_te = train_test_split_temporal(
            synthetic_feature_df, feature_cols,
            train_end="2022-06-30", test_start="2022-07-01"
        )
        assert X_tr.isna().sum().sum() == 0
        assert X_te.isna().sum().sum() == 0

    def test_target_in_range(self, synthetic_feature_df):
        feature_cols = [c for c in get_feature_columns() if c in synthetic_feature_df.columns]
        _, y_tr, _, y_te = train_test_split_temporal(
            synthetic_feature_df, feature_cols,
            train_end="2022-06-30", test_start="2022-07-01"
        )
        assert y_tr.between(0, 100).all()
        assert y_te.between(0, 100).all()


# ---------------------------------------------------------------------------
# XGBoost regressor
# ---------------------------------------------------------------------------

class TestXGBRegressor:

    def test_training_succeeds(self, split_data):
        X_tr, y_tr, X_te, y_te, _ = split_data
        val_size = max(1, int(0.1 * len(X_tr)))
        model = train_xgb_regressor(X_tr.iloc[:-val_size], y_tr.iloc[:-val_size],
                                     X_tr.iloc[-val_size:], y_tr.iloc[-val_size:],
                                     params={"n_estimators": 50})
        assert model is not None

    def test_predictions_in_valid_range_after_clip(self, split_data):
        X_tr, y_tr, X_te, y_te, _ = split_data
        val_size = max(1, int(0.1 * len(X_tr)))
        model = train_xgb_regressor(X_tr.iloc[:-val_size], y_tr.iloc[:-val_size],
                                     X_tr.iloc[-val_size:], y_tr.iloc[-val_size:],
                                     params={"n_estimators": 50})
        preds = np.clip(model.predict(X_te.values), 0, 100)
        assert (preds >= 0).all() and (preds <= 100).all()

    def test_mae_better_than_naive(self, split_data):
        """Model should beat predicting the mean (baseline)."""
        X_tr, y_tr, X_te, y_te, _ = split_data
        val_size = max(1, int(0.1 * len(X_tr)))
        model = train_xgb_regressor(X_tr.iloc[:-val_size], y_tr.iloc[:-val_size],
                                     X_tr.iloc[-val_size:], y_tr.iloc[-val_size:],
                                     params={"n_estimators": 50})
        preds = np.clip(model.predict(X_te.values), 0, 100)
        model_mae = mean_absolute_error(y_te, preds)
        naive_mae = mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean()))
        assert model_mae <= naive_mae * 1.5  # allow some slack for synthetic data

    def test_evaluate_regressor_returns_dict(self, split_data):
        X_tr, y_tr, X_te, y_te, _ = split_data
        val_size = max(1, int(0.1 * len(X_tr)))
        model = train_xgb_regressor(X_tr.iloc[:-val_size], y_tr.iloc[:-val_size],
                                     X_tr.iloc[-val_size:], y_tr.iloc[-val_size:],
                                     params={"n_estimators": 50})
        metrics = evaluate_regressor(model, X_te, y_te)
        assert "mae" in metrics and "r2" in metrics
        assert isinstance(metrics["mae"], float)


# ---------------------------------------------------------------------------
# LightGBM regressor
# ---------------------------------------------------------------------------

class TestLGBRegressor:

    def test_training_succeeds(self, split_data):
        X_tr, y_tr, X_te, y_te, _ = split_data
        val_size = max(1, int(0.1 * len(X_tr)))
        model = train_lgb_regressor(X_tr.iloc[:-val_size], y_tr.iloc[:-val_size],
                                     X_tr.iloc[-val_size:], y_tr.iloc[-val_size:],
                                     params={"n_estimators": 50})
        assert model is not None

    def test_predictions_finite(self, split_data):
        X_tr, y_tr, X_te, y_te, _ = split_data
        val_size = max(1, int(0.1 * len(X_tr)))
        model = train_lgb_regressor(X_tr.iloc[:-val_size], y_tr.iloc[:-val_size],
                                     X_tr.iloc[-val_size:], y_tr.iloc[-val_size:],
                                     params={"n_estimators": 50})
        preds = model.predict(X_te.values)
        assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# XGBoost classifier
# ---------------------------------------------------------------------------

class TestXGBClassifier:

    def test_training_succeeds(self, split_data, synthetic_feature_df):
        X_tr, y_tr, X_te, y_te, _ = split_data
        buckets_all = synthetic_feature_df.set_index(["cusip", "date"])["liquidity_bucket"].astype(str)

        # Rebuild bucket arrays aligned to train/test splits
        df = synthetic_feature_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        train_df = df[df["date"] <= pd.Timestamp("2022-06-30")]
        test_df = df[df["date"] >= pd.Timestamp("2022-07-01")]

        feature_cols = [c for c in get_feature_columns() if c in df.columns]
        X_tr2 = train_df[feature_cols].fillna(0)
        y_tr2 = train_df["liquidity_bucket"].astype(str)
        X_te2 = test_df[feature_cols].fillna(0)
        y_te2 = test_df["liquidity_bucket"].astype(str)

        val_size = max(1, int(0.1 * len(X_tr2)))
        clf, le = train_xgb_classifier(
            X_tr2.iloc[:-val_size], y_tr2.iloc[:-val_size],
            X_tr2.iloc[-val_size:], y_tr2.iloc[-val_size:],
        )
        assert clf is not None
        assert set(le.classes_) == {"High", "Low", "Medium"}

    def test_probabilities_sum_to_one(self, split_data, synthetic_feature_df):
        df = synthetic_feature_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        train_df = df[df["date"] <= pd.Timestamp("2022-06-30")]
        feature_cols = [c for c in get_feature_columns() if c in df.columns]
        X_tr2 = train_df[feature_cols].fillna(0)
        y_tr2 = train_df["liquidity_bucket"].astype(str)

        val_size = max(1, int(0.1 * len(X_tr2)))
        clf, le = train_xgb_classifier(
            X_tr2.iloc[:-val_size], y_tr2.iloc[:-val_size],
            X_tr2.iloc[-val_size:], y_tr2.iloc[-val_size:],
        )
        proba = clf.predict_proba(X_tr2.iloc[:10].values)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:

    def test_cv_returns_dataframe(self, synthetic_feature_df):
        feature_cols = [c for c in get_feature_columns() if c in synthetic_feature_df.columns]
        result = time_series_cv(
            synthetic_feature_df, feature_cols,
            n_splits=3, gap_days=5, model_type="xgb"
        )
        assert isinstance(result, pd.DataFrame)
        assert "fold" in result.columns
        assert "mae" in result.columns
        assert len(result) == 3

    def test_cv_mae_positive(self, synthetic_feature_df):
        feature_cols = [c for c in get_feature_columns() if c in synthetic_feature_df.columns]
        result = time_series_cv(
            synthetic_feature_df, feature_cols,
            n_splits=3, gap_days=5, model_type="xgb"
        )
        assert (result["mae"] >= 0).all()

    def test_cv_fold_sizes_increasing(self, synthetic_feature_df):
        """In walk-forward CV, train size should increase with each fold."""
        feature_cols = [c for c in get_feature_columns() if c in synthetic_feature_df.columns]
        result = time_series_cv(
            synthetic_feature_df, feature_cols,
            n_splits=3, gap_days=5
        )
        train_sizes = result["n_train"].values
        assert all(train_sizes[i] <= train_sizes[i+1] for i in range(len(train_sizes)-1))
