"""
Price per KG/L Prediction Model for Fluidra Balancer Products (Algaecides)
==========================================================================

Predicts "Price per KG / L" from product feature buckets using linear and
log-transformed regression models.

Outputs:
- Model comparison (MAE, RMSE, R²) across 5 interpretable models
- Feature importance via elasticity analysis
- Dollar contribution per feature and per bucket
- Pricing recommendations: Underpriced / Overpriced / Aligned

GLOSSARY:
  OLS        — Ordinary Least Squares; classic linear regression, no penalty.
  Ridge      — Linear regression with L2 penalty; shrinks correlated coefficients.
  Lasso      — Linear regression with L1 penalty; can zero out weak features.
  ElasticNet — Combines L1 + L2 penalties.
  Log-Linear — Predicts log(price); back-transforms via exp(). Handles skew well.
  Log-Log    — Predicts log(price) from log(features+1). Elasticity = coefficient.
  VIF        — Variance Inflation Factor; flags multicollinear features (>5 = concern).
  MAE        — Mean Absolute Error; lower is better.
  RMSE       — Root Mean Squared Error; penalises large errors more than MAE.
  R²         — Fraction of variance explained by the model; higher is better.
  Elasticity — % change in price per 1% change in a feature, at the mean.

HIGH-LEVEL FLOW:
  1. Load and preprocess train (and optionally test) Excel data.
  2. Standardise features on train; apply same transform to test (no leakage).
  3. Train 5 models: OLS, Ridge, Lasso, ElasticNet, Bayesian Ridge,
     Log-Linear, Log-Log.
  4. Pick best model by lowest train MAE.
  5. Compute VIF, correlation matrix, elasticities, feature importance.
  6. Compute dollar contribution per feature and per bucket.
  7. Label each product: Underpriced / Overpriced / Aligned (±15% threshold).
  8. Write all outputs to CSV + model_summary.txt (timestamped).

Entry point: run_model(train_path, test_path, feature_cols, target_col, output_dir)
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Default paths and settings (CLI mode)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
TRAIN_DATA_PATH = BASE_DIR / "Modelling_Input.xlsx"
TEST_DATA_PATH  = BASE_DIR / "Test_Input.xlsx"
TARGET_COL      = "Price per KG / L"

FEATURE_COLS = [
    "Composition Bucket",
    "Brand Bucket",
    "Range Bucker",
    "Size Bucket",
    "Channel Bucket",
]


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_train_data_from_path(path: Path, target_col: str) -> pd.DataFrame:
    """
    Load training Excel data and apply standard preprocessing:
      - Drop rows where price < 2 (outliers / bad data).
      - Fill NaN in Concentration Bucket with 0 (no concentration = base tier).
    """
    df = pd.read_excel(path)
    if target_col in df.columns:
        df = df[df[target_col] >= 2].copy()
    return df


def load_test_data_from_path(
    test_path: Path, df_train: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:
    """
    Load test Excel data and align it with training features:
      - Add any missing feature columns using the train mode (most frequent value).
      - Fill NaN in existing columns with train mode.
    Using train statistics for imputation prevents data leakage.
    """
    df_test = pd.read_excel(test_path)
    for col in feature_cols:
        mode_val = (
            df_train[col].mode().iloc[0]
            if col in df_train.columns and len(df_train[col].mode()) > 0
            else 0
        )
        if col not in df_test.columns:
            df_test[col] = mode_val
        elif df_test[col].isna().any():
            df_test[col] = df_test[col].fillna(mode_val)

    return df_test


# ---------------------------------------------------------------------------
# Feature diagnostics
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature.
    VIF > 5 suggests multicollinearity — consider dropping or merging that feature.
    Returns DataFrame with columns: Feature, VIF.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return pd.DataFrame({"Feature": X.columns, "VIF": [np.nan] * len(X.columns)})

    X_with_const = pd.DataFrame(
        np.column_stack([np.ones(len(X)), X.values]),
        columns=["const"] + list(X.columns),
    )
    rows = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X_with_const.values, i + 1)
        except Exception:
            vif = np.nan
        rows.append({"Feature": col, "VIF": vif})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coefficient and elasticity helpers
# ---------------------------------------------------------------------------

def get_coefficients(model, feature_names: list) -> tuple:
    """
    Extract (coefs, intercept) from a fitted sklearn-style model.
    Pads or trims coefs array to match len(feature_names).
    """
    if hasattr(model, "coef_"):
        coefs = np.array(model.coef_).flatten()
        intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
    else:
        coefs = np.zeros(len(feature_names))
        intercept = 0.0

    if len(coefs) < len(feature_names):
        coefs = np.pad(coefs, (0, len(feature_names) - len(coefs)))
    elif len(coefs) > len(feature_names):
        coefs = coefs[: len(feature_names)]

    return coefs, intercept


def compute_elasticity_linear(coefs, X_mean, X_std, y_mean) -> np.ndarray:
    """
    Elasticity for linear model (standardised X):
      e_i = coef_i * (X_mean_i / (y_mean * X_std_i))
    Answers: "1% increase in feature i -> e_i% change in price."
    """
    elasticities = []
    for i in range(len(coefs)):
        if X_std[i] == 0 or y_mean == 0:
            elasticities.append(0.0)
        else:
            elasticities.append(coefs[i] * (X_mean[i] / (y_mean * X_std[i])))
    return np.array(elasticities)


def compute_elasticity_log_linear(coefs, X_mean) -> np.ndarray:
    """
    Elasticity for log-linear model log(y) ~ X:
      e_i = coef_i * X_mean_i
    """
    return coefs * X_mean


def compute_elasticity_log_log(coefs, X_log_raw) -> np.ndarray:
    """
    Elasticity for log-log model log(y) ~ standardised log(X+1):
      e_i = coef_i / std(log(X_i+1))
    In log-log space, the coefficient IS the elasticity after de-standardising.
    """
    stds = np.std(X_log_raw, axis=0)
    stds = np.where(stds == 0, 1e-6, stds)
    return coefs / stds


# ---------------------------------------------------------------------------
# Business logic: pricing recommendations
# ---------------------------------------------------------------------------

def recommendation(actual: float, pred: float, thresh: float = 0.15) -> str:
    """
    Classify each product's pricing status based on model deviation:
      Underpriced — model predicts higher; product could command more.
      Overpriced  — model predicts lower; product may be losing volume.
      Aligned     — within ±15% of model prediction.
    """
    if actual == 0:
        return "Aligned"
    dev = abs(actual - pred) / actual
    if dev > thresh:
        return "Underpriced" if pred > actual else "Overpriced"
    return "Aligned"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_model(
    train_path: Path,
    test_path: Path | None,
    feature_cols: list | None,
    target_col: str,
    output_dir: Path,
) -> dict:
    """
    Full modelling pipeline: load → preprocess → train → diagnose → output.

    Args:
        train_path   : Path to Modelling_Input.xlsx
        test_path    : Path to Test_Input.xlsx (optional)
        feature_cols : List of feature column names (None = use all non-target columns)
        target_col   : Column to predict, e.g. "Price per KG / L"
        output_dir   : Folder for all output CSVs and model_summary.txt

    Returns:
        dict with keys: output_dir, run_id, best_model, mae_train,
        mae_test (if test set has target), summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = (
        output_dir.name.replace("output_", "", 1)
        if output_dir.name.startswith("output_")
        else datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )

    def out(name: str) -> Path:
        base, ext = name.rsplit(".", 1) if "." in name else (name, "")
        return output_dir / (f"{base}_{ts}.{ext}" if ext else f"{base}_{ts}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading train data from {train_path}...")
    df_train = load_train_data_from_path(train_path, target_col)
    print(f"  Train rows after outlier removal: {len(df_train)}")

    if feature_cols is None:
        feature_cols = [c for c in df_train.columns if c != target_col]
    else:
        feature_cols = [c for c in feature_cols if c in df_train.columns]

    if not feature_cols:
        raise ValueError(
            "No feature columns found in train data. "
            "Check column names in Modelling_Input.xlsx."
        )
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    has_test = test_path is not None and Path(test_path).exists()
    if has_test:
        print(f"Loading test data from {test_path}...")
        df_test = load_test_data_from_path(test_path, df_train, feature_cols)
        print(f"  Test rows: {len(df_test)}")
    else:
        df_test = pd.DataFrame(columns=feature_cols)

    X_train_raw = df_train[feature_cols]
    y_train     = df_train[target_col]
    X_test_raw  = df_test[feature_cols] if len(df_test) > 0 else pd.DataFrame(columns=feature_cols)
    y_test      = df_test[target_col] if (target_col in df_test.columns and len(df_test) > 0) else None
    feature_names = list(X_train_raw.columns)

    # ------------------------------------------------------------------
    # 2. Diagnostics: VIF and correlation matrix (train only, no leakage)
    # ------------------------------------------------------------------
    print("Computing VIF and correlation matrix...")
    compute_vif(X_train_raw).to_csv(out("vif_scores.csv"), index=False)
    X_train_raw.corr().to_csv(out("correlation_matrix.csv"))

    # ------------------------------------------------------------------
    # 3. Standardise features
    #    Fit scaler on train only; apply to both train and test.
    # ------------------------------------------------------------------
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw) if len(X_test_raw) > 0 else np.empty((0, len(feature_names)))

    X_mean = X_train_raw.mean().to_numpy()
    X_std = X_train_raw.std().to_numpy().copy()
    X_std = np.where(X_std == 0, 1e-6, X_std)
    y_mean = float(y_train.mean())

    # Log-space features for log-log model (separate scaler, fit on train only)
    X_log_train_raw = np.log(X_train_raw + 1)
    X_log_test_raw  = np.log(X_test_raw + 1) if len(X_test_raw) > 0 else np.empty((0, len(feature_names)))
    scaler_log  = StandardScaler()
    X_log_train = scaler_log.fit_transform(X_log_train_raw)
    X_log_test  = scaler_log.transform(X_log_test_raw) if len(X_log_test_raw) > 0 else np.empty((0, len(feature_names)))

    # ------------------------------------------------------------------
    # 4. Train models
    #    Linear: OLS, Ridge, Lasso, ElasticNet, Bayesian Ridge
    #    Log-transformed: Log-Linear, Log-Log
    # ------------------------------------------------------------------
    print("Training models...")

    models_config = [
        ("Linear Regression (OLS)", "linear", LinearRegression()),
        ("Ridge Regression",        "linear", Ridge(alpha=1.0)),
        ("Lasso Regression",        "linear", Lasso(alpha=0.01)),
        ("Elastic Net",             "linear", ElasticNet(alpha=0.01, l1_ratio=0.5)),
        ("Bayesian Ridge",          "linear", BayesianRidge()),
    ]

    model_results = {}
    fitted_models = {}

    for name, mtype, model in models_config:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            model_results[name] = {
                "MAE_train":  mean_absolute_error(y_train, preds),
                "RMSE_train": np.sqrt(mean_squared_error(y_train, preds)),
                "R2_train":   r2_score(y_train, preds),
            }
            fitted_models[name] = (mtype, model)
        except Exception as e:
            print(f"  [Skip] {name}: {e}")

    # Log-Linear: predict log(price), back-transform via exp()
    try:
        m_log = LinearRegression()
        m_log.fit(X_train, np.log(y_train))
        preds = np.exp(m_log.predict(X_train))
        model_results["Log-Linear Regression"] = {
            "MAE_train":  mean_absolute_error(y_train, preds),
            "RMSE_train": np.sqrt(mean_squared_error(y_train, preds)),
            "R2_train":   r2_score(y_train, preds),
        }
        fitted_models["Log-Linear Regression"] = ("log", m_log)
    except Exception as e:
        print(f"  [Skip] Log-Linear Regression: {e}")

    # Log-Log: predict log(price) from standardised log(features+1)
    try:
        m_loglog = LinearRegression()
        m_loglog.fit(X_log_train, np.log(y_train))
        preds = np.exp(m_loglog.predict(X_log_train))
        model_results["Log-Log Regression"] = {
            "MAE_train":  mean_absolute_error(y_train, preds),
            "RMSE_train": np.sqrt(mean_squared_error(y_train, preds)),
            "R2_train":   r2_score(y_train, preds),
        }
        fitted_models["Log-Log Regression"] = ("log_log", m_loglog)
    except Exception as e:
        print(f"  [Skip] Log-Log Regression: {e}")

    # ------------------------------------------------------------------
    # Helper: get test predictions for a named model
    # ------------------------------------------------------------------
    def get_test_preds(name: str) -> np.ndarray:
        mtype, model = fitted_models[name]
        if mtype == "log":
            return np.exp(model.predict(X_test))
        if mtype == "log_log":
            return np.exp(model.predict(X_log_test))
        return model.predict(X_test)

    # ------------------------------------------------------------------
    # 5. Test-set metrics (if test file has target column)
    # ------------------------------------------------------------------
    for name in model_results:
        try:
            if y_test is not None and len(y_test) > 0 and len(X_test) > 0:
                preds_t = np.maximum(get_test_preds(name), 0)
                model_results[name]["MAE_test"]  = mean_absolute_error(y_test, preds_t)
                model_results[name]["RMSE_test"] = np.sqrt(mean_squared_error(y_test, preds_t))
                model_results[name]["R2_test"]   = r2_score(y_test, preds_t)
            else:
                model_results[name]["MAE_test"]  = np.nan
                model_results[name]["RMSE_test"] = np.nan
                model_results[name]["R2_test"]   = np.nan
        except Exception:
            model_results[name]["MAE_test"]  = np.nan
            model_results[name]["RMSE_test"] = np.nan
            model_results[name]["R2_test"]   = np.nan

    # ------------------------------------------------------------------
    # 6. Model comparison; pick best by train MAE
    # ------------------------------------------------------------------
    comparison_df = pd.DataFrame([
        {"Model": n, **v} for n, v in model_results.items()
    ]).sort_values("MAE_train")
    comparison_df.to_csv(out("model_comparison.csv"), index=False)

    best_name = comparison_df.iloc[0]["Model"]
    best_res  = model_results[best_name]
    print(f"Best model: {best_name}  (Train MAE={best_res['MAE_train']:.4f})")

    # ------------------------------------------------------------------
    # 7. Elasticities and feature importance
    # ------------------------------------------------------------------
    print("Computing elasticities and feature importance...")
    elasticity_rows = []

    for name, (mtype, model) in fitted_models.items():
        try:
            coefs, _ = get_coefficients(model, feature_names)
            if mtype == "linear":
                elasts = compute_elasticity_linear(coefs, X_mean, X_std, y_mean)
            elif mtype == "log":
                elasts = compute_elasticity_log_linear(coefs, X_mean)
            else:
                elasts = compute_elasticity_log_log(coefs, X_log_train_raw.values)

            for i, feat in enumerate(feature_names):
                e = elasts[i] if i < len(elasts) else 0.0
                elasticity_rows.append({
                    "Model":            name,
                    "Feature":          feat,
                    "Elasticity":       round(e, 6),
                    "Importance_Score": round(abs(e), 6),
                    "Direction":        "+" if e > 0 else ("-" if e < 0 else "0"),
                })
        except Exception as e:
            print(f"  [Skip elasticity] {name}: {e}")

    elasticities_df = pd.DataFrame(elasticity_rows)
    elasticities_df.to_csv(out("elasticities_by_model.csv"), index=False)

    best_importance = (
        elasticities_df[elasticities_df["Model"] == best_name]
        .sort_values("Importance_Score", ascending=False)
    )
    best_importance.to_csv(out("feature_importance.csv"), index=False)

    # ------------------------------------------------------------------
    # 8. Dollar contribution per feature and per bucket
    #    Always use a linear model for interpretable $ breakdown.
    #    If best model is log/log-log, fall back to best linear model.
    # ------------------------------------------------------------------
    best_linear_name = best_name
    if fitted_models[best_name][0] != "linear":
        linear_names = [n for n, (t, _) in fitted_models.items() if t == "linear"]
        if linear_names:
            best_linear_name = min(linear_names, key=lambda n: model_results[n]["MAE_train"])

    mtype_lin, contrib_model = fitted_models[best_linear_name]
    if mtype_lin == "linear":
        coefs, intercept = get_coefficients(contrib_model, feature_names)
        X_std_safe = np.where(X_std == 0, 1e-6, X_std)

        # Per-feature summary
        contributions_train  = X_train * coefs
        marginal_per_unit    = coefs / X_std_safe
        mean_abs_contrib     = np.mean(np.abs(contributions_train), axis=0)
        elasts_linear        = compute_elasticity_linear(coefs, X_mean, X_std, y_mean)

        dollar_summary = pd.DataFrame({
            "Feature":                          feature_names,
            "Marginal_Dollar_Per_Unit":         np.round(marginal_per_unit, 6),
            "Mean_Absolute_Dollar_Contribution":np.round(mean_abs_contrib, 6),
            "Elasticity":                       np.round(elasts_linear, 6),
        })
        dollar_summary.to_csv(out("dollar_contribution_summary.csv"), index=False)

        # Per-product dollar contribution (train)
        train_cols = {
            "Product_Index":        range(len(df_train)),
            "Predicted_Price":      np.round(intercept + contributions_train.sum(axis=1), 6),
            "Intercept_Contribution": round(intercept, 6),
        }
        for i, feat in enumerate(feature_names):
            safe = feat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            train_cols[f"{safe}_Dollar_Contribution"] = np.round(contributions_train[:, i], 6)
        pd.DataFrame(train_cols).to_csv(out("dollar_contribution_train.csv"), index=False)

        # Per-product dollar contribution (test)
        if len(X_test) > 0:
            contributions_test = X_test * coefs
            test_cols = {
                "Product_Index":          range(len(df_test)),
                "Predicted_Price":        np.round(intercept + contributions_test.sum(axis=1), 6),
                "Intercept_Contribution": round(intercept, 6),
            }
            for i, feat in enumerate(feature_names):
                safe = feat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
                test_cols[f"{safe}_Dollar_Contribution"] = np.round(contributions_test[:, i], 6)
            pd.DataFrame(test_cols).to_csv(out("dollar_contribution_test.csv"), index=False)

        # Dollar contribution per bucket value
        # Answers: "since Brand Premium = 1, it adds $X to price;
        #           if it were 0, it would contribute $Y instead."
        bucket_rows   = []
        bucket_lookup = {}
        for j, feat in enumerate(feature_names):
            for b in np.unique(X_train_raw.iloc[:, j].values):
                scaled_b   = (float(b) - X_mean[j]) / X_std_safe[j]
                contrib_b  = coefs[j] * scaled_b
                bucket_rows.append({
                    "Feature":            feat,
                    "Bucket_Value":       b,
                    "Dollar_Contribution":round(contrib_b, 6),
                })
                bucket_lookup[(j, b)] = contrib_b
        pd.DataFrame(bucket_rows).to_csv(out("dollar_contribution_by_bucket.csv"), index=False)

        # Per-product: actual bucket + contribution + "what if other bucket" columns
        per_product_rows = []
        for i in range(len(df_train)):
            row_data = {"Product_Index": i}
            for j, feat in enumerate(feature_names):
                safe = feat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
                bval = X_train_raw.iloc[i, j]
                row_data[f"{safe}_Bucket"]              = bval
                row_data[f"{safe}_Dollar_Contribution"] = round(contributions_train[i, j], 6)
                for b in np.unique(X_train_raw.iloc[:, j].values):
                    if b != bval:
                        row_data[f"{safe}_If_Bucket_{b}"] = round(
                            bucket_lookup.get((j, b), np.nan), 6
                        )
            per_product_rows.append(row_data)
        pd.DataFrame(per_product_rows).to_csv(
            out("dollar_contribution_by_bucket_per_product.csv"), index=False
        )
        print("  Dollar contribution outputs written.")

    # ------------------------------------------------------------------
    # 9. Predictions + pricing recommendations
    # ------------------------------------------------------------------
    mtype_best, best_model = fitted_models[best_name]
    if mtype_best == "log":
        preds_train = np.exp(best_model.predict(X_train))
        preds_test  = np.exp(best_model.predict(X_test)) if len(X_test) > 0 else np.array([])
    elif mtype_best == "log_log":
        preds_train = np.exp(best_model.predict(X_log_train))
        preds_test  = np.exp(best_model.predict(X_log_test)) if len(X_log_test) > 0 else np.array([])
    else:
        preds_train = best_model.predict(X_train)
        preds_test  = best_model.predict(X_test) if len(X_test) > 0 else np.array([])

    # Train: actual vs predicted + recommendation
    pd.DataFrame({
        "Product_Index":   range(len(df_train)),
        "Actual_Price":    y_train.values,
        "Predicted_Price": preds_train,
        "Recommendation":  [recommendation(a, p) for a, p in zip(y_train, preds_train)],
    }).to_csv(out("recommended_prices_train.csv"), index=False)

    # Test: predicted price (+ actual and recommendation if test has target)
    if len(preds_test) > 0:
        preds_test = np.maximum(preds_test, 0)
        test_out = {
            "Product_Index":   range(len(df_test)),
            "Predicted_Price": preds_test,
        }
        if y_test is not None and len(y_test) > 0:
            test_out["Actual_Price"]    = y_test.values
            test_out["Recommendation"]  = [recommendation(a, p) for a, p in zip(y_test, preds_test)]
            mae_test_best = mean_absolute_error(y_test, preds_test)
            print(f"  Test MAE (best model): {mae_test_best:.4f}")
        pd.DataFrame(test_out).to_csv(out("test_predictions.csv"), index=False)

        # All models' test predictions in one file
        all_preds = {"Product_Index": range(len(df_test))}
        if y_test is not None and len(y_test) > 0:
            all_preds["Actual_Price"] = y_test.values
        for name in fitted_models:
            try:
                p = np.maximum(get_test_preds(name), 0)
                safe_col = name.replace("/", "_").replace("(", "").replace(")", "").strip()
                all_preds[safe_col] = np.round(p, 6)
            except Exception:
                pass
        pd.DataFrame(all_preds).to_csv(out("test_predictions_by_model.csv"), index=False)

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    test_line = ""
    if y_test is not None and len(y_test) > 0 and not np.isnan(best_res.get("MAE_test", np.nan)):
        test_line = (
            f"\nTest  MAE: {best_res['MAE_test']:.4f} | "
            f"RMSE: {best_res['RMSE_test']:.4f} | "
            f"R²: {best_res['R2_test']:.4f}"
        )

    summary = f"""Price per KG/L Prediction Model — Summary
==========================================
Train : {train_path.name} ({len(df_train)} rows)
Test  : {test_path.name if test_path else 'None'} ({len(df_test)} rows)

Best Model : {best_name}
Train MAE  : {best_res['MAE_train']:.4f} | RMSE: {best_res['RMSE_train']:.4f} | R²: {best_res['R2_train']:.4f}{test_line}

Preprocessing:
  - Outliers removed (price < 2)
  - Concentration Bucket NaN → 0
  - Test missing columns filled with train mode

Output files (all timestamped):
  model_comparison.csv                       — MAE / RMSE / R² for all models
  feature_importance.csv                     — Elasticity-ranked features (best model)
  elasticities_by_model.csv                  — Elasticities across all models
  dollar_contribution_summary.csv            — Marginal $/unit and mean |$| per feature
  dollar_contribution_train.csv              — Per-product $ contribution (train)
  dollar_contribution_test.csv               — Per-product $ contribution (test)
  dollar_contribution_by_bucket.csv          — $ contribution per feature bucket value
  dollar_contribution_by_bucket_per_product  — Per-product bucket + "what if" counterfactuals
  recommended_prices_train.csv               — Actual vs predicted + Underpriced/Overpriced/Aligned
  test_predictions.csv                       — Test predictions (best model)
  test_predictions_by_model.csv              — Test predictions from every model
  vif_scores.csv                             — VIF per feature
  correlation_matrix.csv                     — Feature correlation matrix
  model_summary.txt                          — This summary
"""
    with open(out("model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    print(f"Done. Outputs in: {output_dir}/")

    result = {
        "output_dir": str(output_dir),
        "run_id":     output_dir.name,
        "best_model": best_name,
        "mae_train":  float(best_res["MAE_train"]),
        "summary":    summary,
    }
    if y_test is not None and len(y_test) > 0 and len(preds_test) > 0:
        result["mae_test"] = float(mean_absolute_error(y_test, preds_test))
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = BASE_DIR / f"output_{timestamp}"
    run_model(TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURE_COLS, TARGET_COL, output_dir)


if __name__ == "__main__":
    main()