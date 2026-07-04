import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# 1. Define dataset size
n_rows = 3*10**6  # 3 million rows

print("Generating features...")

# 2. Generate Continuous Variables
# Age: Uniform distribution between 21 and 65
age = np.random.randint(21, 66, size=n_rows).astype(np.int32)

# Credit Score: Bounded between 300 and 850, peaking around 680
credit_score = np.random.normal(680, 70, size=n_rows)
credit_score = np.clip(credit_score, 300, 850).astype(np.int32)

# Annual Income (INR/USD): Log-normal distribution to mimic real-world wealth distribution
annual_income = np.random.lognormal(mean=13.2, sigma=0.4, size=n_rows)
annual_income = np.clip(annual_income, 200000, 3500000).astype(np.float32)

# Average Monthly Spend: Strong positive correlation with income + random variance
avg_monthly_spend = (annual_income * np.random.uniform(0.15, 0.35, size=n_rows) / 12)
avg_monthly_spend = np.round(avg_monthly_spend, 2).astype(np.float32)

# 3. Generate Categorical Variables
emp_options = ["Salaried", "Self-Employed", "Student", "Unemployed"]
emp_type = np.random.choice(emp_options, size=n_rows, p=[0.65, 0.20, 0.10, 0.05])

housing_options = ["Own", "Rent", "Mortgage"]
housing_status = np.random.choice(housing_options, size=n_rows, p=[0.45, 0.35, 0.20])

existing_loan = np.random.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])

# 4. Generate Target Variable (Non-Zero Inflated)
# We build a latent linear equation + noise to make the target realistic for ML models
norm_income = (annual_income - 800000) / 300000
norm_score = (credit_score - 680) / 70
norm_spend = (avg_monthly_spend - 20000) / 10000

# Logistic-like propensity score
latent_score = (
    0.6 * norm_score
    + 0.5 * norm_spend
    + 0.3 * norm_income
    - 0.4 * (age > 50)
    + np.random.normal(0, 1.2, size=n_rows)
)

# Threshold at the 55th percentile to ensure EXACTLY 45% of rows are 1s (Not zero-inflated)
threshold = np.percentile(latent_score, 55)
target_cc_take = (latent_score >= threshold).astype(np.int8)

# 5. Assemble and Optimize DataFrame
print("Assembling DataFrame...")
df = pd.DataFrame(
    {
        "customer_id": np.arange(1000001, 1000001 + n_rows, dtype=np.int32),
        "age": age,
        "credit_score": credit_score,
        "annual_income": annual_income,
        "avg_monthly_spend": avg_monthly_spend,
        "employment_type": emp_type,
        "housing_status": housing_status,
        "existing_loan": existing_loan,
        "target_cc_take": target_cc_take,
    }
)

# Convert strings to category dtype for a massive memory reduction (~75% lighter)
categorical_cols = ["employment_type", "housing_status", "existing_loan"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

print("\nDataset Generation Complete!")
print(f"Final Shape: {df.shape}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

df.to_csv("synthetic_dataset_cc_takeup.csv", index=False)