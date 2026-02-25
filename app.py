"""
Streamlit dashboard for the Corporate Bond Liquidity Scoring system.

Tabs:
  1. Bond Screener — Filter and rank bonds by liquidity score
  2. Score Breakdown (SHAP) — Feature attribution for a selected bond
  3. Pre-Trade Estimator — Interactive pre-trade cost estimation
  4. Impact Calibration — View and compare market impact model parameters
  5. Universe Heatmap — IG/HY universe liquidity by rating × sector

Run with:
  streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, MODEL_DIR, PROCESSED_DIR, RATINGS, SECTORS
from src.impact.cost_estimator import CostEstimator, market_impact_bps


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bond Liquidity Scoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data / model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models and data...")
def load_resources():
    import joblib

    resources = {
        "reg_model": None,
        "clf_model": None,
        "label_encoder": None,
        "feature_store": None,
        "impact_params": None,
        "shap_values": None,
        "shap_feature_names": None,
        "cost_estimator": None,
    }

    # Models
    model_key_map = {
        "xgb_regressor": "xgb_regressor",
        "xgb_classifier": "xgb_classifier",
        "label_encoder": "label_encoder",
    }
    for name, key in model_key_map.items():
        path = MODEL_DIR / f"{name}.pkl"
        if path.exists():
            try:
                resources[key] = joblib.load(path)
            except Exception:
                pass

    # Feature store
    fs_path = PROCESSED_DIR / "features.parquet"
    if fs_path.exists():
        resources["feature_store"] = pd.read_parquet(fs_path)

    # Impact params
    ip_path = PROCESSED_DIR / "impact_params.parquet"
    if ip_path.exists():
        resources["impact_params"] = pd.read_parquet(ip_path)
    elif resources["feature_store"] is not None:
        resources["impact_params"] = _derive_impact_params(resources["feature_store"])

    # SHAP importance
    shap_path = MODEL_DIR / "shap_importance.csv"
    if shap_path.exists():
        resources["shap_df"] = pd.read_csv(shap_path)

    # Build cost estimator
    if resources["impact_params"] is not None:
        from src.features.feature_pipeline import get_feature_columns
        resources["cost_estimator"] = CostEstimator(
            impact_params_df=resources["impact_params"],
            liquidity_model=resources.get("xgb_regressor"),
            classifier_model=resources.get("xgb_classifier"),
            label_encoder=resources.get("label_encoder"),
            feature_cols=get_feature_columns(),
        )

    return resources


def _derive_impact_params(fs: pd.DataFrame) -> pd.DataFrame:
    latest = fs.sort_values("date").groupby("cusip").last().reset_index()
    records = []
    for _, row in latest.iterrows():
        adv = float(row.get("volume_30d", 5.0)) or 5.0
        sigma = abs(float(row.get("daily_return", 0.005))) or 0.005
        roll = float(row.get("roll_spread_bps_21d", row.get("roll_bps", 30.0)) or 30.0)
        records.append({
            "cusip": row["cusip"],
            "alpha": 0.5, "beta": 0.6,
            "alpha_shrunk": 0.5, "beta_shrunk": 0.6,
            "adv_mm": adv, "sigma_daily": sigma,
            "roll_spread_bps_21d": roll,
            "n_trades": int(row.get("n_trades", 0)),
            "fit_quality": "default",
        })
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def get_latest_scores(_fs: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent scores for each CUSIP."""
    return _fs.sort_values("date").groupby("cusip").last().reset_index()


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main():
    resources = load_resources()
    fs = resources.get("feature_store")
    impact_params = resources.get("impact_params")
    cost_estimator = resources.get("cost_estimator")

    # ---------------------------------------------------------------------------
    # Sidebar — data status
    # ---------------------------------------------------------------------------
    st.sidebar.title("📊 Liquidity Scoring")
    st.sidebar.markdown("---")

    if fs is not None:
        n_cusips = fs["cusip"].nunique()
        date_range = f"{fs['date'].min().date()} → {fs['date'].max().date()}"
        st.sidebar.success(f"✅ Feature store loaded\n\n{n_cusips:,} CUSIPs | {date_range}")
    else:
        st.sidebar.error("❌ Feature store not found. Run `python run_pipeline.py` first.")

    if resources.get("xgb_regressor") is not None:
        st.sidebar.success("✅ ML models loaded")
    else:
        st.sidebar.warning("⚠️ ML models not found. Scores from composite formula.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Corporate Bond Liquidity Scoring v1.0")

    # ---------------------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔍 Bond Screener", "🧠 Score Breakdown", "💰 Pre-Trade Estimator", "📉 Impact Calibration", "🗺️ Universe Heatmap"]
    )

    # ==========================================================================
    # TAB 1: Bond Screener
    # ==========================================================================
    with tab1:
        st.header("Bond Screener")
        if fs is None:
            st.warning("Feature store not available.")
            return

        latest = get_latest_scores(fs)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rating_filter = st.multiselect("Rating", RATINGS, default=RATINGS)
        with col2:
            sector_filter = st.multiselect("Sector", SECTORS, default=SECTORS)
        with col3:
            score_min, score_max = st.slider("Liquidity Score Range", 0, 100, (0, 100))
        with col4:
            ig_only = st.checkbox("IG Only", value=False)

        filtered = latest.copy()
        if "rating" in filtered.columns:
            filtered = filtered[filtered["rating"].isin(rating_filter)]
        if "sector" in filtered.columns:
            filtered = filtered[filtered["sector"].isin(sector_filter)]
        if "liquidity_score" in filtered.columns:
            filtered = filtered[
                (filtered["liquidity_score"] >= score_min) &
                (filtered["liquidity_score"] <= score_max)
            ]
        if ig_only and "is_investment_grade" in filtered.columns:
            filtered = filtered[filtered["is_investment_grade"] == 1]

        st.metric("Bonds matching filter", len(filtered))

        display_cols = [c for c in [
            "cusip", "rating", "sector", "liquidity_score", "liquidity_bucket",
            "outstanding_mm", "ttm_years", "trade_count_30d", "roll_bps",
            "amihud_21d", "is_investment_grade",
        ] if c in filtered.columns]

        if display_cols:
            df_display = filtered[display_cols].sort_values("liquidity_score", ascending=False)
            df_display["liquidity_score"] = df_display["liquidity_score"].round(1)

            # Color-code by liquidity bucket
            def highlight_bucket(val):
                if val == "High": return "background-color: #d4edda"
                if val == "Low": return "background-color: #f8d7da"
                return "background-color: #fff3cd"

            if "liquidity_bucket" in df_display.columns:
                st.dataframe(
                    df_display.style.applymap(highlight_bucket, subset=["liquidity_bucket"]),
                    use_container_width=True,
                    height=420,
                )
            else:
                st.dataframe(df_display, use_container_width=True, height=420)

        # Score distribution
        if "liquidity_score" in filtered.columns:
            st.subheader("Score Distribution")
            fig = px.histogram(
                filtered, x="liquidity_score", nbins=40,
                color="liquidity_bucket" if "liquidity_bucket" in filtered.columns else None,
                color_discrete_map={"High": "#28a745", "Medium": "#ffc107", "Low": "#dc3545"},
                labels={"liquidity_score": "Liquidity Score (0-100)"},
                title="Liquidity Score Distribution",
            )
            fig.add_vline(x=settings.score_low_threshold, line_dash="dash", line_color="red", annotation_text="Low/Med")
            fig.add_vline(x=settings.score_high_threshold, line_dash="dash", line_color="green", annotation_text="Med/High")
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 2: Score Breakdown (SHAP)
    # ==========================================================================
    with tab2:
        st.header("Score Breakdown — Feature Attribution (SHAP)")

        if fs is None:
            st.warning("Feature store not available.")
        else:
            latest_s = get_latest_scores(fs)
            cusips = sorted(latest_s["cusip"].unique().tolist())
            selected_cusip = st.selectbox("Select CUSIP", cusips, key="shap_cusip")

            row = latest_s[latest_s["cusip"] == selected_cusip]
            if not row.empty:
                row = row.iloc[0]

                # Bond info card
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Liquidity Score", f"{row.get('liquidity_score', '—'):.1f}")
                col2.metric("Bucket", str(row.get("liquidity_bucket", "—")))
                col3.metric("Rating", str(row.get("rating", "—")))
                col4.metric("Sector", str(row.get("sector", "—")))

                # SHAP global importance
                shap_path = MODEL_DIR / "shap_importance.csv"
                if shap_path.exists():
                    shap_df = pd.read_csv(shap_path).head(20)
                    fig = px.bar(
                        shap_df,
                        x="mean_abs_shap",
                        y="feature",
                        orientation="h",
                        title="Global Feature Importance (Mean |SHAP|)",
                        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"},
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                # Feature values for selected bond
                feature_display = {
                    "Trade Count (30d avg)": row.get("trade_count_30d"),
                    "Roll Spread (bps)": row.get("roll_bps"),
                    "Amihud 21d": row.get("amihud_21d"),
                    "Kyle Lambda": row.get("kyle_lambda_abs"),
                    "Median ITI (hrs)": row.get("median_iti_hours"),
                    "ADV ($MM)": row.get("volume_30d"),
                    "VIX": row.get("vix"),
                    "CDX IG (bps)": row.get("cdx_ig_bps"),
                    "Outstanding ($MM)": row.get("outstanding_mm"),
                    "TTM (years)": row.get("ttm_years"),
                }
                st.subheader(f"Key Feature Values for {selected_cusip}")
                feat_df = pd.DataFrame(
                    [(k, f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in feature_display.items() if v is not None],
                    columns=["Feature", "Value"],
                )
                st.table(feat_df)
            else:
                st.warning(f"No data found for {selected_cusip}")

    # ==========================================================================
    # TAB 3: Pre-Trade Estimator
    # ==========================================================================
    with tab3:
        st.header("Pre-Trade Cost Estimator")

        if cost_estimator is None:
            st.error("Cost estimator not available. Run pipeline first.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                latest_s2 = get_latest_scores(fs) if fs is not None else pd.DataFrame()
                cusips2 = sorted(latest_s2["cusip"].unique().tolist()) if not latest_s2.empty else ["CUSIP000001"]
                pt_cusip = st.selectbox("CUSIP", cusips2, key="pt_cusip")
                pt_size = st.number_input("Trade Size ($MM par)", min_value=0.1, max_value=500.0, value=5.0, step=0.5)
                pt_side = st.radio("Side", ["Buy (B)", "Sell (S)"])
                side_code = "B" if "Buy" in pt_side else "S"
                pt_target = st.number_input(
                    "Target Cost Budget (bps, optional)", min_value=0.0, value=0.0, step=1.0
                )

                run_btn = st.button("Estimate Pre-Trade Costs", type="primary")

            if run_btn:
                row_pt = latest_s2[latest_s2["cusip"] == pt_cusip].iloc[-1] if not latest_s2.empty else None
                target = pt_target if pt_target > 0 else None

                with st.spinner("Computing..."):
                    est = cost_estimator.estimate(
                        cusip=pt_cusip,
                        trade_size_mm=pt_size,
                        side=side_code,
                        features=row_pt,
                        target_cost_bps=target,
                    )

                with col2:
                    st.subheader(f"Results for {pt_cusip}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Bid-Ask Cost", f"{est.bid_ask_cost_bps:.1f} bps")
                    m2.metric("Market Impact", f"{est.market_impact_bps:.1f} bps")
                    m3.metric("Total Cost", f"{est.total_cost_bps:.1f} bps",
                              delta=f"90% CI: [{est.ci_lower_90_bps:.1f}, {est.ci_upper_90_bps:.1f}]")

                    m4, m5, m6 = st.columns(3)
                    m4.metric("Liquidity Score", f"{est.liquidity_score:.1f}")
                    m5.metric("Liquidity Bucket", est.liquidity_bucket)
                    m6.metric("Est. Exec. Time", f"{est.est_execution_hours:.1f} hrs")

                    # Cost breakdown bar chart
                    fig = go.Figure(go.Bar(
                        x=["Bid-Ask Cost", "Market Impact"],
                        y=[est.bid_ask_cost_bps, est.market_impact_bps],
                        marker_color=["#0d6efd", "#dc3545"],
                        text=[f"{est.bid_ask_cost_bps:.1f} bps", f"{est.market_impact_bps:.1f} bps"],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title=f"Cost Breakdown: {pt_cusip} ({pt_size}$MM {pt_side})",
                        yaxis_title="bps", showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Bucket probabilities
                    if est.bucket_probabilities:
                        bucket_df = pd.DataFrame(
                            list(est.bucket_probabilities.items()),
                            columns=["Bucket", "Probability"]
                        )
                        fig2 = px.bar(
                            bucket_df, x="Bucket", y="Probability",
                            color="Bucket",
                            color_discrete_map={"High": "#28a745", "Medium": "#ffc107", "Low": "#dc3545"},
                            title="Liquidity Bucket Probabilities",
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    st.caption(
                        f"Model: α={est.alpha:.3f}, β={est.beta:.3f}, "
                        f"ADV={est.adv_mm:.1f}$MM | Fit: {est.fit_quality}"
                    )

        # Market impact curve visualization
        st.subheader("Market Impact Curve")
        if impact_params is not None and fs is not None:
            viz_cusip = st.selectbox("CUSIP for curve", sorted(impact_params["cusip"].unique()), key="mi_curve_cusip")
            row_ip = impact_params[impact_params["cusip"] == viz_cusip]
            if not row_ip.empty:
                row_ip = row_ip.iloc[0]
                a = float(row_ip.get("alpha_shrunk", row_ip.get("alpha", 0.5)))
                b = float(row_ip.get("beta_shrunk", row_ip.get("beta", 0.6)))
                adv = float(row_ip.get("adv_mm", 5.0))
                sigma = float(row_ip.get("sigma_daily", 0.005))

                sizes = np.linspace(0.1, min(adv * 2, 50), 100)
                impacts = [market_impact_bps(s, a, b, adv, sigma) for s in sizes]

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=sizes, y=impacts, mode="lines", name="Market Impact",
                                          line=dict(color="#0d6efd", width=2)))
                fig3.update_layout(
                    title=f"Market Impact Curve: {viz_cusip} (α={a:.3f}, β={b:.3f})",
                    xaxis_title="Trade Size ($MM)", yaxis_title="Market Impact (bps)",
                )
                st.plotly_chart(fig3, use_container_width=True)

    # ==========================================================================
    # TAB 4: Impact Calibration
    # ==========================================================================
    with tab4:
        st.header("Market Impact Model Calibration")

        if impact_params is None:
            st.warning("Impact parameters not available.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    impact_params, x="alpha_shrunk" if "alpha_shrunk" in impact_params.columns else "alpha",
                    nbins=40, title="Distribution of α (Impact Coefficient)",
                    labels={"alpha_shrunk": "α (impact coefficient)"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    impact_params, x="beta_shrunk" if "beta_shrunk" in impact_params.columns else "beta",
                    nbins=40, title="Distribution of β (Concavity Exponent)",
                    labels={"beta_shrunk": "β (concavity exponent)"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Scatter: n_trades vs r_squared
            if "n_trades" in impact_params.columns and "r_squared" in impact_params.columns:
                fig = px.scatter(
                    impact_params,
                    x="n_trades", y="r_squared",
                    color="fit_quality" if "fit_quality" in impact_params.columns else None,
                    title="CUSIP Fit Quality: R² vs Number of TRACE Prints",
                    labels={"n_trades": "Total TRACE Prints", "r_squared": "OLS R²"},
                    log_x=True,
                    opacity=0.6,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("CUSIP-Level Parameters")
            display_ip = impact_params.round(4)
            st.dataframe(display_ip, use_container_width=True, height=350)

    # ==========================================================================
    # TAB 5: Universe Heatmap
    # ==========================================================================
    with tab5:
        st.header("Universe Liquidity Heatmap")

        if fs is None:
            st.warning("Feature store not available.")
        else:
            latest_hm = get_latest_scores(fs)

            if "rating" in latest_hm.columns and "sector" in latest_hm.columns:
                agg = (
                    latest_hm.groupby(["rating", "sector"])
                    .agg(
                        median_score=("liquidity_score", "median"),
                        n_bonds=("cusip", "count"),
                    )
                    .reset_index()
                )

                # Pivot for heatmap
                pivot = agg.pivot(index="rating", columns="sector", values="median_score")
                pivot = pivot.reindex(RATINGS)

                fig = px.imshow(
                    pivot,
                    color_continuous_scale="RdYlGn",
                    zmin=0, zmax=100,
                    title="Median Liquidity Score by Rating × Sector",
                    labels=dict(color="Liquidity Score"),
                    aspect="auto",
                )
                fig.update_xaxes(side="top")
                st.plotly_chart(fig, use_container_width=True)

                # Bond count heatmap
                pivot_n = agg.pivot(index="rating", columns="sector", values="n_bonds")
                pivot_n = pivot_n.reindex(RATINGS)

                fig2 = px.imshow(
                    pivot_n,
                    color_continuous_scale="Blues",
                    title="Number of Bonds by Rating × Sector",
                    labels=dict(color="# Bonds"),
                    aspect="auto",
                )
                fig2.update_xaxes(side="top")
                st.plotly_chart(fig2, use_container_width=True)

                # Treemap
                fig3 = px.treemap(
                    latest_hm,
                    path=["is_investment_grade", "rating", "sector"],
                    values="outstanding_mm" if "outstanding_mm" in latest_hm.columns else None,
                    color="liquidity_score" if "liquidity_score" in latest_hm.columns else None,
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 100],
                    title="Bond Universe Treemap — Size = Outstanding, Color = Liquidity Score",
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Rating/sector columns not available in feature store.")


if __name__ == "__main__":
    main()
