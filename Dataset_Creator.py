from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

risk_segments = np.random.choice(
    ["Prime", "NearPrime", "SubPrime", "HighRisk"],
    size=n,
    p=[0.40, 0.30, 0.20, 0.10]
)

marital_status = np.random.choice(
    ["Single", "Married", "Divorced"], size=n, p=[0.45, 0.45, 0.10]
)

employment_type = np.random.choice(
    ["Salaried", "SelfEmployed", "Business", "Student"],
    size=n,
    p=[0.65, 0.15, 0.15, 0.05]
)

risk_factor = pd.Series(risk_segments).map(
    {"Prime": 0, "NearPrime": 1, "SubPrime": 2, "HighRisk": 3}
).values

age = np.clip(np.random.normal(40 - risk_factor * 2, 10, n), 21, 75).round()
customer_tenure_months = np.clip(np.random.gamma(5, 18, n), 3, 300).round()

annual_income = np.clip(
    np.random.lognormal(mean=13.2 - risk_factor * 0.2, sigma=0.5, size=n),
    150000,
    5000000,
).round()

bureau_score = np.clip(
    np.random.normal(780 - risk_factor * 70, 40, n),
    300,
    850
).round()

total_credit_limit = np.clip(
    annual_income * np.random.uniform(0.15, 0.8, n),
    20000,
    2000000
).round()

utilization_avg_12m = np.clip(
    np.random.beta(2 + risk_factor, 5, n),
    0,
    1.5
)

utilization_avg_6m = np.clip(
    utilization_avg_12m + np.random.normal(0.03 * risk_factor, 0.08, n),
    0,
    1.5
)

utilization_avg_3m = np.clip(
    utilization_avg_6m + np.random.normal(0.04 * risk_factor, 0.08, n),
    0,
    1.5
)

utilization_max_12m = np.clip(
    utilization_avg_12m + np.random.uniform(0.05, 0.5, n),
    0,
    1.5
)

total_outstanding_balance = (
    total_credit_limit * utilization_avg_3m
).round()

num_open_trades = np.clip(
    np.random.poisson(4 + risk_factor, n),
    1,
    25
)

num_delinquent_accounts = np.clip(
    np.random.poisson(risk_factor * 0.8, n),
    0,
    10
)

payment_ratio_avg_12m = np.clip(
    np.random.normal(0.95 - risk_factor * 0.15, 0.15, n),
    0.05,
    1.5
)

payment_ratio_avg_6m = np.clip(
    payment_ratio_avg_12m - np.random.normal(0.02 * risk_factor, 0.08, n),
    0.05,
    1.5
)

payment_ratio_avg_3m = np.clip(
    payment_ratio_avg_6m - np.random.normal(0.03 * risk_factor, 0.08, n),
    0.05,
    1.5
)

dpd_avg_12m = np.clip(
    np.random.gamma(1 + risk_factor, 3, n),
    0,
    90
)

dpd_avg_6m = np.clip(
    dpd_avg_12m + np.random.normal(risk_factor * 2, 4, n),
    0,
    120
)

dpd_avg_3m = np.clip(
    dpd_avg_6m + np.random.normal(risk_factor * 3, 5, n),
    0,
    180
)

max_dpd_12m = np.clip(
    dpd_avg_12m + np.random.gamma(2, 8, n),
    0,
    180
)

txn_count_avg_12m = np.clip(
    np.random.poisson(20 - risk_factor * 2, n),
    1,
    100
)

txn_count_avg_6m = np.clip(
    txn_count_avg_12m + np.random.normal(0, 3, n),
    1,
    100
)

txn_count_avg_3m = np.clip(
    txn_count_avg_6m + np.random.normal(0, 3, n),
    1,
    100
)

spend_avg_12m = np.clip(
    annual_income * np.random.uniform(0.01, 0.05, n),
    1000,
    200000
)

spend_avg_6m = np.clip(
    spend_avg_12m * np.random.uniform(0.9, 1.1, n),
    1000,
    250000
)

spend_avg_3m = np.clip(
    spend_avg_6m * np.random.uniform(0.9, 1.1, n),
    1000,
    250000
)

cash_advance_amt_3m = np.clip(
    np.random.gamma(1 + risk_factor, 3000, n),
    0,
    100000
)

missed_payments_last_6m = np.clip(
    np.random.poisson(risk_factor * 0.8, n),
    0,
    12
)

logit = (
    -5.0
    + 2.5 * (risk_segments == "HighRisk")
    + 1.5 * (risk_segments == "SubPrime")
    + 0.8 * (utilization_avg_3m > 0.85)
    + 1.2 * (payment_ratio_avg_3m < 0.30)
    + 1.5 * (max_dpd_12m > 30)
    + 0.7 * (missed_payments_last_6m > 1)
    + 0.5 * (bureau_score < 620)
)

pd_default = 1 / (1 + np.exp(-logit))
default_flag = np.random.binomial(1, pd_default)

df = pd.DataFrame({
    "age": age,
    "customer_tenure_months": customer_tenure_months,
    "annual_income": annual_income,
    "marital_status": marital_status,
    "employment_type": employment_type,
    "risk_segment": risk_segments,
    "bureau_score": bureau_score,
    "total_credit_limit": total_credit_limit,
    "total_outstanding_balance": total_outstanding_balance,
    "num_open_trades": num_open_trades,
    "num_delinquent_accounts": num_delinquent_accounts,
    "utilization_avg_3m": utilization_avg_3m,
    "utilization_avg_6m": utilization_avg_6m,
    "utilization_avg_12m": utilization_avg_12m,
    "utilization_max_12m": utilization_max_12m,
    "payment_ratio_avg_3m": payment_ratio_avg_3m,
    "payment_ratio_avg_6m": payment_ratio_avg_6m,
    "payment_ratio_avg_12m": payment_ratio_avg_12m,
    "dpd_avg_3m": dpd_avg_3m,
    "dpd_avg_6m": dpd_avg_6m,
    "dpd_avg_12m": dpd_avg_12m,
    "max_dpd_12m": max_dpd_12m,
    "txn_count_avg_3m": txn_count_avg_3m,
    "txn_count_avg_6m": txn_count_avg_6m,
    "txn_count_avg_12m": txn_count_avg_12m,
    "spend_avg_3m": spend_avg_3m.round(),
    "spend_avg_6m": spend_avg_6m.round(),
    "spend_avg_12m": spend_avg_12m.round(),
    "cash_advance_amt_3m": cash_advance_amt_3m.round(),
    "missed_payments_last_6m": missed_payments_last_6m,
    "default_flag": default_flag
})

csv_path = "credit_card_default_ssb_dataset_50000.csv"
df.to_csv(csv_path, index=False)

print({
    "rows": len(df),
    "columns": len(df.columns),
    "default_rate": round(df["default_flag"].mean(), 4),
    "file": csv_path
})
