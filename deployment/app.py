# =====================================================
# app.py — Fraud Detection API (EFL-based)
# XGBoost + Native TreeSHAP + Dual Dashboard Mode
# =====================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

import joblib
import json
import io
import pandas as pd
import numpy as np
import xgboost as xgb


# =====================================================
# LOAD MODEL & DATA
# =====================================================

model           = joblib.load("fraud_model.pkl")
features        = joblib.load("features.pkl")
X_test          = joblib.load("X_test_imp.pkl")
X_test_original = joblib.load("X_test_original.pkl")
y_test          = joblib.load("y_test.pkl")

# Economic manual review cost threshold (C_MR)
C_MR = 10.0  # US dollars ($) per manual review, same currency as the dataset and thesis

# Fix XGBoost 3.x TreeSHAP compatibility
booster = model.get_booster()
cfg     = json.loads(booster.save_config())
cfg['learner']['learner_model_param']['base_score'] = '0.5'
booster.load_config(json.dumps(cfg))

# =====================================================
# BATCH STORAGE (for dashboard)
# =====================================================
LAST_BATCH = None


# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title       = "Fraud Detection API",
    description = "Fraud Detection & Explainability using XGBoost + TreeSHAP (EFL-based)",
    version     = "1.0"
)


# =====================================================
# HELPERS
# =====================================================

def get_risk_level(probability: float):
    if probability >= 0.90:
        return "CRITICAL", "#c62828"
    elif probability >= 0.50:
        return "HIGH", "#ef6c00"
    elif probability >= 0.10:
        return "MEDIUM", "#f9a825"
    else:
        return "LOW", "#2e7d32"


def compute_shap(row_df: pd.DataFrame):
    dmatrix  = xgb.DMatrix(row_df, feature_names=features)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    shap_vals  = contribs[0][:-1]
    base_value = float(contribs[0][-1])
    return shap_vals, base_value


def top_shap_drivers(row_df: pd.DataFrame, shap_vals, n=10):
    shap_df = pd.DataFrame({
        "feature": features,
        "value": row_df.iloc[0].values,
        "shap": shap_vals,
    })
    shap_df["abs"] = np.abs(shap_df["shap"])
    shap_df = shap_df.sort_values("abs", ascending=False).head(n)

    drivers = []
    for _, r in shap_df.iterrows():
        value = float(r["value"]) if isinstance(r["value"], (np.integer, np.floating)) else r["value"]
        drivers.append({
            "feature": r["feature"],
            "value": round(float(r["value"]), 0),
            "shap": round(float(r["shap"]), 4),
            "direction": "↑ fraud" if r["shap"] > 0 else "↓ legit",
        })
    return drivers


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "XGBoost Fraud Detector",
        "decisioning": "EFL-based (p(fraud) * TransactionAmt)",
        "C_MR": C_MR,
        "transactions_loaded": len(X_test),
        "features": len(features),
        "endpoints": {
            "/predict_sample/{idx}": "JSON prediction + SHAP for one transaction (test set)",
            "/batch_predict": "JSON predictions for multiple indices (test set)",
            "/score_batch_csv": "Upload CSV → EFL-ranked list for manual review",
            "/dashboard/{transaction_id}": "HTML dashboard (test set or CSV)",
        }
    }


# =====================================================
# SINGLE PREDICTION — JSON (test set only, EFL-based)
# =====================================================

@app.get("/predict_sample/{idx}")
def predict_sample(idx: int):
    if idx < 0 or idx >= len(X_test):
        return {"error": f"Index must be between 0 and {len(X_test)-1}"}

    row = pd.DataFrame([X_test[idx]], columns=features)
    probability = float(model.predict_proba(row)[0, 1])

    amount = float(X_test_original.iloc[idx]["TransactionAmt"])
    efl = probability * amount

    prediction = int(efl >= C_MR)
    decision = "MANUAL REVIEW" if prediction else "APPROVE"
    risk_level, _ = get_risk_level(probability)

    shap_vals, base_value = compute_shap(row)
    base_probability = float(1 / (1 + np.exp(-base_value)))

    return {
        "sample": idx,
        "actual_label": int(y_test[idx]),
        "TransactionAmt": amount,
        "fraud_probability": round(probability, 4),
        "base_probability": round(base_probability, 4),
        "C_MR": C_MR,
        "EFL": round(efl, 2),
        "prediction": prediction,
        "decision": decision,
        "risk_level": risk_level,
        "top_drivers": top_shap_drivers(row, shap_vals),
    }


# =====================================================
# BATCH PREDICTION — JSON (test set only, EFL-based)
# =====================================================

@app.post("/batch_predict")
def batch_predict(indices: list[int]):
    results = []
    for idx in indices:
        if idx < 0 or idx >= len(X_test):
            results.append({
                "sample": idx,
                "error": f"Index out of range (0–{len(X_test)-1})"
            })
            continue

        row = pd.DataFrame([X_test[idx]], columns=features)
        probability = float(model.predict_proba(row)[0, 1])
        amount = float(X_test_original.iloc[idx]["TransactionAmt"])
        efl = probability * amount

        prediction = int(efl >= C_MR)
        risk_level, _ = get_risk_level(probability)

        results.append({
            "sample": idx,
            "actual_label": int(y_test[idx]),
            "TransactionAmt": amount,
            "fraud_probability": round(probability, 4),
            "EFL": round(efl, 2),
            "C_MR": C_MR,
            "prediction": prediction,
            "decision": "MANUAL REVIEW" if prediction else "APPROVE",
            "risk_level": risk_level,
        })

    flagged = sum(1 for r in results if r.get("prediction") == 1)
    return {
        "total": len(results),
        "flagged_for_review": flagged,
        "auto_approved": len(results) - flagged,
        "flag_rate_pct": round(flagged / len(results) * 100, 2) if results else 0.0,
        "C_MR": C_MR,
        "predictions": results,
    }


# =====================================================
# CSV BATCH SCORING — realistic operational workflow (EFL-based)
# =====================================================

@app.post("/score_batch_csv")
async def score_batch_csv(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df_new = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        return {"error": f"Could not parse CSV: {str(e)}"}

    if df_new.empty:
        return {"error": "Uploaded CSV is empty"}

    if "TransactionAmt" not in df_new.columns:
        return {"error": "CSV must contain a 'TransactionAmt' column"}

    if "transaction_id" not in df_new.columns:
        return {"error": "CSV must contain a 'transaction_id' column"}

    n_received = len(df_new)

    # Align features for model input
    df_scored = df_new.reindex(columns=features, fill_value=0)

    probas = model.predict_proba(df_scored.values)[:, 1]
    df_new['fraud_score'] = probas

    # Compute EFL
    df_new['EFL'] = df_new['fraud_score'] * df_new['TransactionAmt']

    # EFL-based decision
    df_new['decision'] = np.where(df_new['EFL'] >= C_MR, 'MANUAL REVIEW', 'APPROVE')
    df_new['risk_level'] = [get_risk_level(p)[0] for p in probas]

    # Rank by EFL (descending)
    df_new['rank'] = df_new['EFL'].rank(ascending=False, method='first').astype(int)

    flagged = df_new[df_new['decision'] == 'MANUAL REVIEW'].sort_values('EFL', ascending=False)
    approved = df_new[df_new['decision'] == 'APPROVE']

    top20 = flagged.head(20)[['transaction_id', 'rank', 'fraud_score', 'EFL', 'decision', 'risk_level', 'TransactionAmt']].copy()
    top20['fraud_score'] = top20['fraud_score'].round(4)
    top20['EFL'] = top20['EFL'].round(2)

    global LAST_BATCH
    LAST_BATCH = df_new.copy()

    return {
        "summary": {
            "total_received": n_received,
            "flagged_for_review": len(flagged),
            "auto_approved": len(approved),
            "flag_rate_pct": round(len(flagged) / n_received * 100, 2),
            "C_MR": C_MR,
        },
        "top_20_priority_review": top20.to_dict('records'),
        "all_results": df_new[['transaction_id', 'rank', 'fraud_score', 'EFL', 'decision', 'risk_level', 'TransactionAmt']]
            .sort_values('rank')
            .to_dict('records'),
    }


# =====================================================
# DASHBOARD — HTML (dual-mode: CSV batch or test set, EFL-based)
# =====================================================

@app.get("/dashboard/{transaction_id}", response_class=HTMLResponse)
def dashboard(transaction_id: str):

    # =====================================================
    # 1. PRODUCTION MODE — CSV batch
    # =====================================================
    if LAST_BATCH is not None:
        row_index = LAST_BATCH.index[LAST_BATCH['transaction_id'] == transaction_id]

        if len(row_index) > 0:
            # --- CSV MODE ---
            mode = "Production (CSV batch)"
            idx = row_index[0]
            source_df = LAST_BATCH
            actual_label = "N/A (production data)"
            amount = float(source_df.loc[idx]["TransactionAmt"])

            row = pd.DataFrame([source_df.loc[idx, features]], columns=features)
            probability = float(model.predict_proba(row)[0, 1])
            efl = probability * amount

            prediction  = int(efl >= C_MR)
            decision    = "MANUAL REVIEW" if prediction else "APPROVE"
            risk_level, colour = get_risk_level(probability)
            percentage  = probability * 100

            shap_vals, base_value = compute_shap(row)
            base_probability      = float(1 / (1 + np.exp(-base_value)))

            shap_df = pd.DataFrame({
                "Feature": features,
                "Value"  : row.iloc[0].values,
                "SHAP"   : shap_vals,
            })
            shap_df["ABS"] = np.abs(shap_df["SHAP"])
            shap_df = shap_df.sort_values("ABS", ascending=False).head(10)

            table_rows = ""
            for _, r in shap_df.iterrows():
                direction        = "↑ Fraud" if r["SHAP"] > 0 else "↓ Legit"
                direction_colour = "#c62828" if r["SHAP"] > 0 else "#2e7d32"
                table_rows += f"""
                <tr>
                    <td>{r["Feature"]}</td>
                    <td>{r["Value"]:.4f}</td>
                    <td>{r["SHAP"]:.4f}</td>
                    <td style="font-weight:bold;color:{direction_colour};">
                        {direction}
                    </td>
                </tr>"""

            html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Fraud Detection Dashboard — Transaction {transaction_id}</title>
<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f6f9;
    margin: 40px;
}}
.container {{ max-width: 1100px; margin: auto; }}
h1 {{ color: #003366; margin-bottom: 4px; }}
.subtitle {{ color: #666; margin-bottom: 30px; font-size: 14px; }}
.cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin-bottom: 25px;
}}
.card {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 3px 10px rgba(0,0,0,.10);
}}
.card-title {{ font-size: 13px; color: #666; }}
.card-value {{
    margin-top: 10px;
    font-size: 26px;
    font-weight: bold;
    color: {colour};
}}
.panel {{
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 3px 10px rgba(0,0,0,.10);
}}
.bar {{
    width: 100%;
    height: 26px;
    background: #e0e0e0;
    border-radius: 20px;
    overflow: hidden;
    margin: 12px 0;
}}
.fill {{
    width: {percentage:.1f}%;
    height: 100%;
    background: {colour};
    border-radius: 20px;
}}
.meta {{ color: #444; font-size: 14px; line-height: 2; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{
    background: #003366;
    color: white;
    padding: 11px 14px;
    text-align: left;
}}
td {{ padding: 9px 14px; border-bottom: 1px solid #ececec; }}
tr:hover {{ background: #f9f9f9; }}
.footer {{
    margin-top: 30px;
    font-size: 12px;
    color: #999;
    text-align: center;
}}
.badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    background: {colour};
    color: white;
}}
</style>
</head>
<body>
<div class="container">

<h1>Fraud Detection Dashboard</h1>
<div class="subtitle">
    Transaction ID: {transaction_id} &nbsp;|&nbsp;
    Mode: {mode}
</div>

<div class="cards">
    <div class="card">
        <div class="card-title">Fraud Probability</div>
        <div class="card-value">{probability:.2%}</div>
    </div>
    <div class="card">
        <div class="card-title">Decision</div>
        <div class="card-value">{decision}</div>
    </div>
    <div class="card">
        <div class="card-title">Actual Label</div>
        <div class="card-value">{actual_label}</div>
    </div>
    <div class="card">
        <div class="card-title">Risk Level</div>
        <div class="card-value">{risk_level}</div>
    </div>
</div>

<div class="panel">
    <h2 style="margin-top:0;">Fraud Risk & Economic Impact</h2>
    <div class="bar"><div class="fill"></div></div>
    <div class="meta">
        <b>Transaction Amount:</b> ${amount:,.2f} &nbsp;|&nbsp;
        <b>EFL (expected loss):</b> €{efl:,.2f} &nbsp;|&nbsp;
        <b>C_MR (review cost):</b> €{C_MR:,.2f} &nbsp;|&nbsp;
        <b>Base probability:</b> {base_probability:.2%} &nbsp;|&nbsp;
        <span class="badge">{risk_level}</span>
    </div>
</div>

<div class="panel">
    <h2 style="margin-top:0;">Top 10 SHAP Drivers</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Value</th>
            <th>SHAP</th>
            <th>Effect</th>
        </tr>
        {table_rows}
    </table>
</div>

<div class="footer">
    Model: XGBoost Tuned &nbsp;|&nbsp;
    Decisioning: EFL-based (p(fraud) × amount) &nbsp;|&nbsp;
    C_MR: €{C_MR:,.2f}
</div>

</div>
</body>
</html>"""

            return HTMLResponse(content=html)

        # --- CSV loaded but ID NOT FOUND → go to test mode ---
        else:
            pass  # continue to test mode below

    # =====================================================
    # 2. VALIDATION MODE — Test set
    # =====================================================
    if transaction_id.isdigit() and int(transaction_id) < len(X_test):

        mode = "Validation (Test set)"
        idx = int(transaction_id)
        source_df = X_test
        actual_label = "FRAUD" if y_test[idx] == 1 else "LEGIT"
        amount = float(X_test_original.iloc[idx]["TransactionAmt"])

        row = pd.DataFrame([source_df[idx]], columns=features)
        probability = float(model.predict_proba(row)[0, 1])
        efl = probability * amount

        prediction  = int(efl >= C_MR)
        decision    = "MANUAL REVIEW" if prediction else "APPROVE"
        risk_level, colour = get_risk_level(probability)
        percentage  = probability * 100

        shap_vals, base_value = compute_shap(row)
        base_probability      = float(1 / (1 + np.exp(-base_value)))

        shap_df = pd.DataFrame({
            "Feature": features,
            "Value"  : row.iloc[0].values,
            "SHAP"   : shap_vals,
        })
        shap_df["ABS"] = np.abs(shap_df["SHAP"])
        shap_df = shap_df.sort_values("ABS", ascending=False).head(10)

        table_rows = ""
        for _, r in shap_df.iterrows():
            direction        = "↑ Fraud" if r["SHAP"] > 0 else "↓ Legit"
            direction_colour = "#c62828" if r["SHAP"] > 0 else "#2e7d32"
            table_rows += f"""
            <tr>
                <td>{r["Feature"]}</td>
                <td>{r["Value"]:.4f}</td>
                <td>{r["SHAP"]:.4f}</td>
                <td style="font-weight:bold;color:{direction_colour};">
                    {direction}
                </td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Fraud Detection Dashboard — Transaction {transaction_id}</title>
<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f6f9;
    margin: 40px;
}}
.container {{ max-width: 1100px; margin: auto; }}
h1 {{ color: #003366; margin-bottom: 4px; }}
.subtitle {{ color: #666; margin-bottom: 30px; font-size: 14px; }}
.cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin-bottom: 25px;
}}
.card {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 3px 10px rgba(0,0,0,.10);
}}
.card-title {{ font-size: 13px; color: #666; }}
.card-value {{
    margin-top: 10px;
    font-size: 26px;
    font-weight: bold;
    color: {colour};
}}
.panel {{
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 3px 10px rgba(0,0,0,.10);
}}
.bar {{
    width: 100%;
    height: 26px;
    background: #e0e0e0;
    border-radius: 20px;
    overflow: hidden;
    margin: 12px 0;
}}
.fill {{
    width: {percentage:.1f}%;
    height: 100%;
    background: {colour};
    border-radius: 20px;
}}
.meta {{ color: #444; font-size: 14px; line-height: 2; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{
    background: #003366;
    color: white;
    padding: 11px 14px;
    text-align: left;
}}
td {{ padding: 9px 14px; border-bottom: 1px solid #ececec; }}
tr:hover {{ background: #f9f9f9; }}
.footer {{
    margin-top: 30px;
    font-size: 12px;
    color: #999;
    text-align: center;
}}
.badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    background: {colour};
    color: white;
}}
</style>
</head>
<body>
<div class="container">

<h1>Fraud Detection Dashboard</h1>
<div class="subtitle">
    Transaction ID: {transaction_id} &nbsp;|&nbsp;
    Mode: {mode}
</div>

<div class="cards">
    <div class="card">
        <div class="card-title">Fraud Probability</div>
        <div class="card-value">{probability:.2%}</div>
    </div>
    <div class="card">
        <div class="card-title">Decision</div>
        <div class="card-value">{decision}</div>
    </div>
    <div class="card">
        <div class="card-title">Actual Label</div>
        <div class="card-value">{actual_label}</div>
    </div>
    <div class="card">
        <div class="card-title">Risk Level</div>
        <div class="card-value">{risk_level}</div>
    </div>
</div>

<div class="panel">
    <h2 style="margin-top:0;">Fraud Risk & Economic Impact</h2>
    <div class="bar"><div class="fill"></div></div>
    <div class="meta">
        <b>Transaction Amount:</b> ${amount:,.2f} &nbsp;|&nbsp;
        <b>EFL (expected loss):</b> €{efl:,.2f} &nbsp;|&nbsp;
        <b>C_MR (review cost):</b> €{C_MR:,.2f} &nbsp;|&nbsp;
        <b>Base probability:</b> {base_probability:.2%} &nbsp;|&nbsp;
        <span class="badge">{risk_level}</span>
    </div>
</div>

<div class="panel">
    <h2 style="margin-top:0;">Top 10 SHAP Drivers</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Value</th>
            <th>SHAP</th>
            <th>Effect</th>
        </tr>
        {table_rows}
    </table>
</div>

<div class="footer">
    Model: XGBoost Tuned &nbsp;|&nbsp;
    Decisioning: EFL-based (p(fraud) × amount) &nbsp;|&nbsp;
    C_MR: €{C_MR:,.2f}
</div>

</div>
</body>
</html>"""

        return HTMLResponse(content=html)

    # =====================================================
    # 3. NOT FOUND ANYWHERE
    # =====================================================
    return HTMLResponse(
        f"<h2>Transaction ID '{transaction_id}' not found in CSV nor test set.</h2>",
        status_code=404
    )
