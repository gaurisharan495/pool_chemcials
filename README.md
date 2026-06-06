# Pool Chemicals Pricing Analytics — Fluidra Balancer Products

A pricing analytics model to predict Price per KG/L for pool 
chemical products (Algaecides segment) and generate business 
recommendations for pricing alignment.

Built during internship at Finzarc. Client data is confidential 
and not included — use sample_input.xlsx to run the pipeline.

---

## What It Does

- Predicts price per KG/L from product attributes (size, 
  concentration, formulation, brand, function buckets)
- Runs 7 regression models and selects best by train MAE
- Computes dollar contribution per feature and per bucket — 
  answers "how much does Brand Premium add to price vs no premium?"
- Labels each product as Underpriced / Overpriced / Aligned 
  based on deviation from model prediction (±15% threshold)
- Outputs VIF scores to flag multicollinear features

---

## Models Used

| Model | Type |
|---|---|
| OLS Linear Regression | Linear |
| Ridge Regression | Linear + L2 |
| Lasso Regression | Linear + L1 |
| Elastic Net | Linear + L1/L2 |
| Bayesian Ridge | Linear + Bayesian prior |
| Log-Linear Regression | Log-transformed target |
| Log-Log Regression | Log-transformed features + target |

---

## Feature Engineering

Products are encoded into interpretable buckets:
- **Size Bucket** — pack size tier
- **Concentration Bucket** — active ingredient concentration tier
- **Solid/Liquid Bucket** — formulation type
- **Brand Premium** — brand tier flag
- **Function Bucket** — product function category
- **Corrective/Preventive Bucket** — usage type

Header zone keywords scored 3x for title-zone sensitivity.
Missing Concentration Bucket values filled with 0 (no 
concentration data = base tier assumption).

---

## Business Outputs

- `recommended_prices_train.csv` — Actual vs predicted price 
  with Underpriced/Overpriced/Aligned label per product
- `dollar_contribution_by_bucket.csv` — Dollar impact of each 
  feature bucket on price
- `dollar_contribution_by_bucket_per_product.csv` — Per product: 
  actual bucket contribution + "what if" counterfactuals
- `feature_importance.csv` — Elasticity-based importance ranking
- `model_comparison.csv` — MAE, RMSE, R² for all models

---

## How To Run

```bash
pip install -r requirements.txt

# Place your input file as Modelling_Input.xlsx (see sample_input.xlsx)
python price_prediction_model.py
```

Outputs written to `output_YYYY-MM-DD_HHMMSS/`

---

*Client pricing data confidential — dummy sample provided for 
reproducibility.*
