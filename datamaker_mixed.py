import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def generate_loan_dataset(output_path="loan_applications_500k.parquet", num_rows=500_000):
    print(f"Generating {num_rows:,} simulated loan applications...")
    np.random.seed(42)  # For reproducible testing

    # 1. Core Target: Default Indicator (1 = Defaulted, 0 = Paid off)
    # Simulating a typical ~15% baseline default rate in high-risk lending
    target = np.random.choice([0, 1], size=num_rows, p=[0.85, 0.15])

    # 2. Continuous Numeric Features
    credit_score = np.clip(np.random.normal(680, 50, size=num_rows) + (target * -60), 300, 850)
    income = np.clip(np.random.exponential(60000, size=num_rows) + (target * -15000), 10000, 500000)
    debt_to_income = np.clip(np.random.beta(2, 5, size=num_rows) * 100 + (target * 12), 0, 100)

    # 3. Categorical Features (Strings)
    home_ownership_pool = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    home_ownership = np.random.choice(home_ownership_pool, size=num_rows, p=[0.4, 0.45, 0.14, 0.01])
    
    purpose_pool = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical"]
    loan_purpose = np.random.choice(purpose_pool, size=num_rows, p=[0.5, 0.2, 0.15, 0.1, 0.05])

    # 4. Explicit Binary Numeric Features (0 or 1 flags)
    has_prior_bankruptcy = np.random.choice([0, 1], size=num_rows, p=[0.92, 0.08])
    # Correlate a flag slightly with default
    is_auto_pay = np.where(target == 1, 
                           np.random.choice([0, 1], size=num_rows, p=[0.8, 0.2]), 
                           np.random.choice([0, 1], size=num_rows, p=[0.3, 0.7]))

    # =========================================================================
    # THE DISGUISED LEAK: "recovery_collection_fee"
    # =========================================================================
    # In practice, bank risk databases write off non-paying loans to agencies.
    # If the customer defaulted (target=1), they generated a collection fee 95% of the time.
    # If they paid fine (target=0), their fee is strictly 0. 
    # This feature will have a massive Information Value (IV) and represents complete leakage.
    recovery_collection_fee = np.where(
        target == 1,
        np.clip(np.random.normal(150, 50, size=num_rows), 0, None) * np.random.choice([0, 1], size=num_rows, p=[0.05, 0.95]),
        0.0
    )
    # =========================================================================

    # Build the PyArrow Table schema directly for downstream pipeline consumption
    data_dict = {
        "application_id": [f"APP-{i:06d}" for i in range(num_rows)],
        "credit_score": credit_score,
        "annual_income": income,
        "debt_to_income_ratio": debt_to_income,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "has_prior_bankruptcy": has_prior_bankruptcy,
        "is_auto_pay": is_auto_pay,
        "recovery_collection_fee": recovery_collection_fee,  # <-- The Trojan Horse Leak!
        "is_defaulted": target  # Dependent Variable
    }

    table = pa.Table.from_pydict(data_dict)
    pq.write_table(table, output_path)
    print(f"Successfully exported clean test dataset to: {output_path}")

if __name__ == "__main__":
    generate_loan_dataset()