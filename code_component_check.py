import numpy as np
import pandas as pd
import logging
from rapidsegment import StrategicSegmentBuilder

# Configure local logging to observe execution flow
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("TestRunner")

def generate_stress_test_data(n_rows: int = 500000) -> pd.DataFrame:
    """Generates a 500k row dataset with structured segments, categorical features,
    and highly volatile low-volume trap segments to stress-test the engine constraints.
    """
    logger.info(f"Generating {n_rows:,} rows of synthetic test data...")
    np.random.seed(42)
    
    # 1. Generate clean continuous numerical variables
    utilization = np.random.uniform(0.0, 1.0, n_rows)
    age = np.random.randint(18, 75, n_rows)
    debt_to_income = np.random.uniform(0.1, 0.6, n_rows)
    
    # 2. Generate categorical variables (with varying string lengths)
    risk_grades = np.random.choice(["Grade_A", "Grade_B", "Grade_C", "Grade_D", "Grade_E"], size=n_rows, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    employment_types = np.random.choice(["Salaried", "Self_Employed", "Unemployed"], size=n_rows, p=[0.7, 0.2, 0.1])
    
    # 3. Establish a baseline uninformative background target rate (~2% default rate)
    target = np.random.choice([0, 1], size=n_rows, p=[0.98, 0.02])
    
    df = pd.DataFrame({
        "utilization": utilization,
        "age": age,
        "debt_to_income": debt_to_income,
        "risk_grade": risk_grades,
        "employment_type": employment_types,
        "target": target
    })
    
    # --------------------------------------------------------------------------
    # INJECTING VALID HIGH-VOLUME SEGMENTS (What the engine SHOULD find)
    # --------------------------------------------------------------------------
    # Segment 1: High utilization + Poor Risk Grade (~15k rows, ~25% event rate)
    mask_seg1 = (df["utilization"] > 0.85) & (df["risk_grade"] == "Grade_E")
    df.loc[mask_seg1, "target"] = np.random.choice([0, 1], size=mask_seg1.sum(), p=[0.75, 0.25])
    
    # Segment 2: Young age + Unemployed (~8k rows, ~20% event rate)
    mask_seg2 = (df["age"] < 25) & (df["employment_type"] == "Unemployed")
    df.loc[mask_seg2, "target"] = np.random.choice([0, 1], size=mask_seg2.sum(), p=[0.80, 0.20])

    # --------------------------------------------------------------------------
    # INJECTING THE "TRAP" SEGMENT (Tests the Grid Search & Volatility Fixes)
    # --------------------------------------------------------------------------
    # We isolate a tiny group of exactly 12 rows. We turn all 12 into positive events.
    # This creates a pseudo-segment with 100% event rate (massive Lift), but it 
    # strictly violates your robust minimum sample size floor (e.g., min_sample_size=1000).
    mask_trap = (df["utilization"] < 0.05) & (df["age"] == 33) & (df["risk_grade"] == "Grade_A")
    trap_indices = df[mask_trap].index[:12]  # Force precisely 12 rows
    
    if len(trap_indices) >= 12:
        df.loc[trap_indices, "target"] = 1
        # Alter the feature values slightly to create a clean isolated boundary string
        df.loc[trap_indices, "employment_type"] = "Self_Employed"
        logger.info(f"Successfully injected a high-lift volatile trap segment of size: {len(trap_indices)}")
    
    logger.info("Dataset generation complete.")
    return df

if __name__ == "__main__":
    # Generate the dataset (Contains zero NULL values)
    data = generate_stress_test_data(500000)
    
    # Initialize the builder with a dynamic grid search configuration
    builder = StrategicSegmentBuilder(
        target="target",
        n_jobs=-1,              # Enforce maximum multi-core processing
        min_sample_size=1000,    # Top-level global floor requirement
        min_lift=2.0,
        min_events=20,
        top_n_vars=5,
        max_segments=5,
        param_grid={
            "min_sample_size": [10, 500, 2000],  # Includes the '10' size trap setting
            "min_lift": [1.5, 3.0]
        }
    )
    
    # Run execution pipeline
    logger.info("Starting segment extraction loop...")
    discovered_segments = builder.extract_segments(data)
    
    print("\n" + "="*80)
    print("EXTRACTED SEGMENTS PERFORMANCE SUMMARY")
    print("="*80)
    for seg in discovered_segments:
        print(f"Segment {seg['segment_id']}:")
        print(f"  • SQL Filter   : {seg['sql_filter']}")
        print(f"  • Row Count    : {seg['count']:,}")
        print(f"  • Event Rate   : {seg['rate']:.2f}%")
        print(f"  • Segment Lift : {seg['lift']:.2f}x")
        print(f"  • Grid Ceilings: Min Sample Size={seg['meta_applied_sample_size']} | Min Lift={seg['meta_applied_min_lift']}")
        print("-" * 80)
        
    # Evaluate final coverage cross-check metrics
    logger.info("Running final coverage database audit...")
    final_report = builder.evaluate_final_coverage(data)
    
    print("\n" + "="*80)
    print("FINAL MUTUALLY EXCLUSIVE COVERAGE AUDIT REPORT")
    print("="*80)
    print(f"{'SEGMENT_ID':<12}{'TOTAL_ROWS':<14}{'EVENTS':<10}{'RESP_RATE':<12}{'LIFT':<10}")
    print("-" * 80)
    for row in final_report:
        print(f"{row['segment']:<12}{row['total_count']:<14,}{int(row['target_events']):<10}{row['response_rate']:<12.2f}%{row['lift']:<10.2f}x")
    print("="*80)