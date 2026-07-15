
> [!IMPORTANT]
> **Legal Disclaimer**
> This open-source software library (`RapidSegment`) is an independent, community-driven predictive analytics framework. It is **completely unaffiliated, unrelated, and not associated** with any commercial products, software-as-a-service (SaaS) platforms, trademarks, or enterprise solutions of the same or similar name found on the internet or operated by other corporations. Any overlap in nomenclature is purely coincidental.

<p align="center">
<img width="500" height="300" alt="ChatGPT Image Jul 15, 2026, 08_33_50 PM" src="https://github.com/user-attachments/assets/0d06b429-aa61-4730-9f40-90ee4f0805c5" />
</p>

# RapidSegment - Strategic Segmentation & Scorecard Engine

A high-performance, industrial-grade combinatorial heuristic engine and scoring framework for extracting highly predictive, mutually exclusive segments from tabular data and compiling them into optimized financial scorecards.

---

## 1. Executive Summary

### WHAT is it?

The **Strategic Segmentation & Scorecard Engine (RapidSegment)** is an automated framework designed to solve core challenges in enterprise predictive analytics: discovering high-performing sub-populations (segments) within large feature spaces, executing high-throughput cloud feature screening, and compiling those segments into transparent, production-ready linear scoring models.

It contains four decoupled, synergistic core components:

1. **`StrategicSegmentBuilder`**: Searches feature combinations using an Apriori-inspired pruning technique and an exhaustive multi-threshold grid search to extract precise, non-overlapping rule conditions.

2. **`StrategicSegmentScore`**: Converts those rules into a vectorized scoring model by computing harmonic feature weights, executing ultra-fast linear algebra transformations, calibrating decile distributions, and exporting the results into a lightweight JSON artifact.

3. **`BigQueryFeatureSelector`**: Scales massive feature screening loops directly within Google BigQuery, evaluating Information Value (IV) and variation markers across billions of records before downloading data.

4. **`UniversalDataLoader`**: Standardizes multi-format inputs (CSV, Parquet, Arrow, Excel, and BigQuery) into optimized in-memory structures compatible with vectorized down-stream compute engines.

### WHY do we need it?

In risk management, fraud detection, and marketing analytics, engineering transparent models usually forces teams into a difficult trade-off:
* **The Machine Learning Pitfall**: Black-box models (e.g., XGBoost, LightGBM) surface multi-way feature interactions automatically but cannot be natively translated into transparent, hard-coded SQL logic required by legacy transaction rules engines.
* **The Manual Profiling Pitfall**: Manual segmentation misses non-linear intersections of three or more variables, risks severe over-fitting on small sample sizes, and frequently creates overlapping conditions where a single record qualifies for multiple rules.
* **The Solution**: This framework bridges the gap by combining the automated discovery power of multi-threaded algorithmic search with the execution speeds of **DuckDB** and **Num BLAS operations**, outputting pure ANSI SQL text filters and robust linear scorecards.

### HOW does it work?

The engine runs recursively through an analytical lifecycle:
1. **Rank Feature Significance**: Uses Information Value (IV) calculated via optimal discretization to isolate and surface the top predictive columns.
2. **Combinatorial Grid Search**: Pairs and triplets features using an Apriori heuristic across a hyperparameter grid of sample sizes and lift limits to find highly predictive intersections.
3. **Isolate and Deduplicate**: Selects the single strongest multi-way rule, translates it into production-ready SQL, and extracts the matching records using an in-memory SQL layer (`duckdb`) to guarantee absolute mutual exclusivity for subsequent iterations.
4. **Vectorized Weight Calculation**: Performs an O(1) table scan across the final rule cohort matrix to compute segment weights using actual lift scaled by an optimized harmonic mean formulation.
5. **BLAS Execution & Model Export**: Multiplies the resulting sparse input arrays against the weight vector at raw C speeds using a matrix dot-product, calibrates score deciles, and flushes the parameters to a standardized JSON schema.

---

## 2. Statistical & Weighting Foundations

### 1. Feature Profiling: WOE & IV

Before evaluating multi-way intersections, the engine filters individual continuous and categorical features using **Weight of Evidence (WOE)** and **Information Value (IV)**.

* **WOE (Weight of Evidence)**: Measures the predictive power of an individual bin relative to the overall baseline population. It establishes how much a specific value band shifts the log-odds of an event occurring:
  $$WOE = \ln \left( \frac{\text{Percent of Non-Events}}{\text{Percent of Events}} \right)$$
* **IV (Information Value)**: Summarizes the overall predictive power of the entire variable across all its discrete bins:
  $$IV = \sum \left( \text{Percent of Non-Events} - \text{Percent of Events} \right) \times WOE$$

Variables that yield a total $IV \times 100 > 30.0$ are flagged as strong individual predictors and are automatically prioritized during rule induction loops.

---

### 2. Scorecard Weight Formulation

Once rules are finalized, they are passed to the scorecard module as a matrix of binary indicator flags ($1$ if the customer satisfies the rule, $0$ otherwise). The engine calculates the mathematical weight for each segment using a combination of **Lift** and the **Harmonic Mean of Response and Capture Rates**.

For a given segment $s$:
* **Response Rate ($RR_s$)**: $\frac{\text{Events}_s}{\text{Total Count}_s}$
* **Capture Rate ($CR_s$)**: $\frac{\text{Events}_s}{\text{Total Population Events}}$
* **Lift ($L_s$)**: $\frac{RR_s}{\text{Baseline Population Event Rate}}$

The engine derives the balance between structural segment density (capture rate) and vertical risk concentration (response rate) by computing their harmonic mean:
$$\text{Harmonic Mean}_s = 2 \times \left( \frac{RR_s \times CR_s}{RR_s + CR_s} \right)$$

This metric is multiplied by the segment's lift and scaled to produce a robust, integer-rounded operational weight:
$$\text{Raw Weight}_s = L_s \times \text{Harmonic Mean}_s \times 100.0$$
$$\text{Weight}_s = \lfloor \text{round}(\text{Raw Weight}_s) \rfloor$$

#### Decile Calculation

* Calculate and Sort Final Scores: For each customer, sum up the weights of all the segments they triggered to calculate their total score. Line up all customers and sort them by this final score in descending order (highest score to lowest score).
* Divide the Population Equally: Split the sorted list of customers into 10 perfectly equal groups (buckets). For example, if you have 10,000 customers, each decile bucket will contain exactly 1,000 customers.. 
* Identify the Boundary Row: For each decile band (1 through 10), look at the very last customer sitting at the bottom of that specific bucket.
* Assign the Minimum Threshold: Capture that customer's score and assign it as the decile minimum threshold. This score represents the lowest value required to qualify for that specific decile tier.

---

### 3. Zero-Inflation & Active Population Calibration

Real-world behavioral datasets frequently exhibit high zero-inflation, where the vast majority of records fail to trigger any specialized rules.

To prevent highly skewed zero-bins from distorting score calibrations, the engine calculates the dataset's **Zero-Inflation Rate**:
$$\text{Zero-Inflation Rate} = 1.0 - \text{Baseline Population Event Rate}$$

* **Normal Distribution (< 80% Zero-Inflation)**: If the zero-inflation rate is low, the engine computes decile minimum score thresholds across the entire population.
* **High Zero-Inflation ($\ge$ 80% Zero-Inflation)**: If the zero-inflation rate meets or exceeds the 80% threshold, the engine automatically runs a Num boolean slicing mask to isolate the **Active Population** (`train_scores > 0`). Decile step-boundaries are then calibrated exclusively over this active subgroup, preventing a large block of unsegmented records from flattening the model's risk stratification tiers.

## 3. Algorithmic Design: Apriori Pruning & Grid Search

Evaluating multidimensional feature intersections across wide schemas risks severe processing delays and combinatorial explosion. This framework addresses these bottlenecks by pairing a multi-core **Apriori pruning heuristic** with a **Champion-Challenger Grid Search** architecture.

### 1. Apriori-Style Dimensional Pruning

If you evaluate combinations up to 3 dimensions deep across the top 20 predictive variables, a standard brute-force grid search must calculate every mathematical combination:
* **1-Way Checks**: 20 aggregations
* **2-Way Checks**: $\binom{20}{2} = 190$ aggregations
* **3-Way Checks**: $\binom{20}{3} = 1,140$ aggregations
* **Total Evaluated Configurations**: **1,350 aggregations** *per iteration*.

The engine cuts down this search space by leveraging the Apriori property: *If a single feature path fails to meet a performance threshold, any higher-order combination containing that path is guaranteed to fail.*

```text
[Top 20 Ranked Features] ──► [Level 1: 1-Way Check] ──► (Only 6 Features Pass Floors)
                                      │
                                      ▼
                             [Level 2: 2-Way Check] ──► (Only pairs formed from those 6 evaluated: 15 pairs)
                                      │
                                      ▼
                             [Level 3: 3-Way Check] ──► (Only triplets where ALL internal pairs passed evaluated)
```
This pruning mechanism reduces multi-key grouping evaluations by up to 90%, speeding up execution while ensuring higher-order 3-way rules represent stable, statistically sound relationships rather than random noise in tiny data subsets.

### 2. Multi-Threshold Grid Search

When extracting rules, the engine natively accepts an execution hyperparameter matrix (param_grid) tracking lists of min_sample_size and min_lift constraints.
```thon
param_grid = {
    "min_sample_size": [1000, 5000, 10000], 
    "min_lift": [1.5, 2.0, 2.5]
}
```
Instead of short-circuiting early or requiring inputs to be sorted from highest to lowest threshold, the engine uses an exhaustive execution lifecycle:  
1. **Permutation Generation**: It maps every parameter permutation into isolated, independent experiments via itertools.product.
2. **Exhaustive Evaluation**: For each iteration, it runs the entire multi-level Apriori combination loop for every hyperparameter combination, collecting the top rule that cleared that specific experiment's floor into a master grid_candidates array.
3. **Global Champion Resolution**: Once all experiments finish, the complete candidate table is sorted globally across all dimensions:
     ```thon
    grid_candidates.sort(key=lambda x: (x["lift"], x["count"], x["rate"]), reverse=True)
     ```
4. **Winning Extraction**: The record at index zero (iloc[0]) is crowned the absolute champion for that loop. The engine locks in its specific rule, updates feature usage metrics, tracks the applied parameters, and isolates the target cohort.
---
* ## 4. System Architecture & Process Flow

```text
                        [Raw Tabular Dataset Input]
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Compute Feature IV Rankings   │
                     └───────────────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │  Optimal Monotonic Binning    │
                     └───────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         [Apriori Pruning Layers]          [Multi-Threshold Grid Search]
         • Level 1: Filter 1-Way Base      • Generates all size/lift combinations
         • Level 2: Construct 2-Way Pairs  • Runs experiments via parallel loops
         • Level 3: Form valid triplets    • Gathers candidate rule sets
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Global Champion Selection     │ ◄── Sorts by actual Lift & Volume
                     └───────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
       ┌─────────────────────────┐       ┌─────────────────────────┐
       │ Parse Rule to Pure SQL  │       │ Residual Filter Block   │
       │ (IN, AND, Range Clauses)│       │ (DuckDB In-Memory Scan) │
       └─────────────────────────┘       └─────────────────────────┘
                                                      │
                                                      ▼
                                        [Loop to Next Segment Iteration]
                                                      │
                                           (Once Segment Pool completes)
                                                      │
                                                      ▼
                                     ┌───────────────────────────────┐
                                     │ Scorecard Engine Initialization│
                                     └───────────────────────────────┐
                                                      │
                                                      ▼
                                     ┌───────────────────────────────┐
                                     │ Single-Pass DuckDB Aggregation│ ◄── Extracts population/event counts
                                     └───────────────────────────────┐
                                                      │
                                                      ▼
                                     ┌───────────────────────────────┐
                                     │ Weight Compilation Matrix     │ ◄── Scaled via Harmonic Mean equations
                                     └───────────────────────────────┐
                                                      │
                                                      ▼
                                     ┌───────────────────────────────┐
                                     │ BLAS Matrix Dot-Product Opt   │ ◄── Array @ Vector at raw C speeds
                                     └───────────────────────────────┐
                                                      │
                                                      ▼
                                     ┌───────────────────────────────┐
                                     │ Decile Threshold Calibration  │ ◄── Auto-isolates active pop if >= 80%
                                     └───────────────────────────────┐
                                                      │
                                                      ▼
                                      [Final JSON Model Model Export]

```

### Batch 3: Parameter & Attribute Reference


## 5. Class Attributes & Parameter Reference

### 1. `UniversalDataLoader`

Standardizes multiple ingestion pathways into a uniform, memory-optimized PyArrow data frame abstraction.

#### Initialization Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `project_id` | `str` | `None` | Google Cloud project location (for BigQuery streaming). |
| `dataset_id` | `str` | `None` | Target BigQuery Dataset identifier. |
| `table_id` | `str` | `None` | Target BigQuery Table identifier. |
| `file_path` | `str` | `None` | Local folder system pointer path (`.csv`, `.parquet`, `.xlsx`, `.feather`). |

### 2. `StrategicSegmentBuilder`

#### Initialization Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target` | `str` | *Required* | Dependent binary target column name (`1` = Event, `0` = Non-Event). |
| `n_jobs` | `int` | `-1` | CPU threads for parallel processing. `-1` uses `available_cores - 1`. |
| `min_sample_size` | `int` | `1000` | Absolute minimum record volume floor required to validate a rule. |
| `min_lift` | `float` | `2.0` | Minimum lift cutoff value ($\text{Segment Rate} / \text{Base Rate}$). |
| `min_events` | `float` | `5.0` | Absolute minimum event record volume floor required to validate a rule. |
| `top_n_vars` | `int` | `20` | Total highest-IV features passed into the combinatorial engine. |
| `max_segments` | `int` | `10` | Hard stopping ceiling for total extracted mutually exclusive segments. |
| `max_feature_reuse` | `int` | `1` | Max times an individual feature can appear across all final rules. |
| `enable_diversity` | `bool` | `False` | If `True`, blocks combinations pairing features within the same business group. |
| `enable_1way` | `bool` | `True` | Allows or blocks 1-dimensional rules in the final candidate pool. |
| `enable_2way` | `bool` | `True` | Allows or blocks 2-dimensional intersection rules in the final pool. |
| `enable_3way` | `bool` | `True` | Allows or blocks 3-dimensional intersection rules in the final pool. |
| `feature_groups` | `Dict` | `None` | Maps descriptive business category keys to groups of column strings. |
| `ignore_features` | `List` | `None` | Explicit list of metadata columns to drop before running processing steps. |

#### Generated Output Fields (Segment Dataframe)

* **`segment_id`**: Sequential iteration integer index.
* **`rule_string`**: Raw rule syntax returned by the OptBinning transformation layer.
* **`sql_filter`**: Standardized production-ready ANSI SQL WHERE clause condition.
* **`count`**: Actual volume of records matching the rule criteria.
* **`rate`**: Internal event frequency percentage observed inside the segment.
* **`lift`**: Calculated performance lift multiplier relative to the global population rate.
* **`meta_applied_sample_size`**: The specific `min_sample_size` parameter that captured the winning rule.
* **`meta_applied_min_lift`**: The specific `min_lift` parameter that captured the winning rule.

#### Diagnostic & Audit Trail Methods

**`explain_feature_journey(feature_name: str)`**: Prints an execution audit trail tracking the targeted feature across all execution loops. Details its dynamic IV, previous usage count, structural status flags (e.g., Excluded, Eligible), and whether it was adopted by a winning segment.

---

### 3. `StrategicSegmentScore`

#### Initialization Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `target_col` | `str` | Name of the dependent binary target column (`1` = Event, `0` = Non-Event). |
| `primary_key` | `str` | Row-level tracking key or transaction sequence identifier string. |
| `segment_cols` | `List[str]` | List of compiled binary segment indicator columns ($1$ or $0$) to build into the scorecard. |

#### Exported JSON Model Artifact Schema

* **`model_metadata`**: Holds population execution records (`total_training_population`, `active_scored_population`, `active_population_pct`, `baseline_event_rate`).
* **`segment_weights`**: Nested dictionary mapping each column to its `weight`, `lift`, `response_rate`, and `capture_rate`.
* **`decile_min_thresholds`**: Dictionary mapping decile levels (`"1"` to `"10"`) to their corresponding integer score cutoffs.

### 4. `BigQueryFeatureSelector`

Designed for enterprise-scale pre-screening loops. It calculates naive IV distributions and filters variance boundaries natively inside Google BigQuery to prevent local memory bottlenecks.

#### Initialization Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `project_id` | `str` | `Required` | Google Cloud Platform Project identifier. |
| `dataset_id` | `str` | `Required` | Target BigQuery Dataset namespace. |
| `table_id` | `str` | `Required` | Target BigQuery Data Table identity string. |
| `target_column` | `str` | `Required` | Dependent binary tracking variable (1 or 0). |
| `iv_threshold` | `float` | `0.02` | Minimum Information Value needed to keep a feature. |
| `stddev_threshold` | `float` | `1e-5` | Minimum standard deviation needed to prevent zero-variance processing. |
| `min_bin_n_event` | `int` | `1` | Defensive baseline floor for positive items inside an individual bin. |
| `bins` | `int` | `10` | Quantile groupings (NTILE) used to build numeric range limits. |
| `batch_size` | `int` | `15` | Maximum number of column structures processed per BigQuery execution hit. |



## 6. Quick Start Guide

### 1. Installation
You can install RapidSegment directly from PyPI using either standard pip or the high-performance uv virtual environment package manager:
```bash
pip install rapidsegment
```
or
```bash
uv pip install rapidsegment
```
### 2. Get Started
This guide demonstrates an end-to-end analytical pipeline: extracting rules across a hyperparameter grid, evaluating cascading database coverage, generating binary indicator flags, and building an optimized scorecard.

```Python
import numpy as np
import pandas as pd
import duckdb
from rapidsegment import StrategicSegmentBuilder, StrategicSegmentScore, UniversalDataLoader

# 1. Generate Synthetic Tabular Transaction Pool for Verification
np.random.seed(42)
n_records = 50000

data = pd.DataFrame({
    "cust_id": [f"CUST_{i:05d}" for i in range(n_records)],
    "max_dpd_12m": np.random.choice([0, 15, 30, 60, 90], size=n_records, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
    "utilization_avg_3m": np.random.uniform(0.0, 1.2, size=n_records),
    "spend_avg_6m": np.random.exponential(scale=3000, size=n_records),
    "payment_ratio_3m": np.random.uniform(0.0, 1.0, size=n_records),
    "risk_segment": np.random.choice(["Low", "Medium", "High"], size=n_records, p=[0.6, 0.3, 0.1]),
    "default_flag": np.random.choice([0, 1], size=n_records, p=[0.95, 0.05]) # 5% baseline rate
})

# Inject structured high-risk rules to verify engine extraction
high_risk_mask = (data["max_dpd_12m"] >= 60) & (data["utilization_avg_3m"] >= 0.85)
data.loc[high_risk_mask, "default_flag"] = np.random.choice([0, 1], size=high_risk_mask.sum(), p=[0.2, 0.8])

# 2. Configure Domain Knowledge Feature Groups
business_groups = {
    "delinquency_metrics": ["max_dpd_12m", "risk_segment"],
    "utilization_metrics": ["utilization_avg_3m"],
    "transaction_metrics": ["spend_avg_6m", "payment_ratio_3m"]
}

# 3. Define Multi-Threshold Hyperparameter Grid
grid_config = {
    "min_sample_size": [1000, 2500, 5000],
    "min_lift": [2.0, 3.5, 5.0]
}

# 4. Initialize and Run the Segment Extraction Engine
builder = StrategicSegmentBuilder(
    target="default_flag",
    top_n_vars=15,
    max_segments=5,
    max_feature_reuse=1,
    param_grid=grid_config,
    enable_diversity=True,
    feature_groups=business_groups,
    ignore_features=["cust_id"]
)

print("Executing recursive multi-threshold segment search...")
segments_summary = builder.extract_segments(data)

# Convert list of dicts to DataFrame for clean terminal output profiling
segments_df = pd.DataFrame(segments_summary)
print("\nExtracted Segment Profiles:")
print(segments_df[["segment_id", "count", "lift", "meta_applied_sample_size", "sql_filter"]])

# [Diagnostic Step] Print audit trail for a key feature
builder.explain_feature_journey("max_dpd_12m")

# 5. Review Cascading Portfolio Coverage Analysis Report
coverage_report = builder.evaluate_final_coverage(data)
print("\nCascading Portfolio Coverage Analysis:")
print(pd.DataFrame(coverage_report))

# 6. Prepare Binary Array Representation for Scorecard Tuning
scoring_df = data[["cust_id", "default_flag"]].copy()

# Map SQL filters to binary columns (1 = matches rule, 0 = otherwise)
segment_columns = []
for segment in segments_summary:
    seg_id = segment["segment_id"]
    sql_cond = segment["sql_filter"]
    col_name = f"SEGMENT_{seg_id}"
    
    # Query via DuckDB to apply pure SQL strings directly to the dataframe
    matched_ids = duckdb.query(f"SELECT cust_id FROM data WHERE {sql_cond}").df()["cust_id"]
    scoring_df[col_name] = scoring_df["cust_id"].isin(matched_ids).astype(int)
    segment_columns.append(col_name)

# 7. Execute High-Throughput Scorecard Matrix Engine
scorecard_engine = StrategicSegmentScore(
    target_col="default_flag",
    primary_key="cust_id",
    segment_cols=segment_columns
)

print("\nCompiling scorecard weights and decile thresholds...")
model_parameters = scorecard_engine.calculate_and_export_weights(
    data=scoring_df, 
    export_path="production_scorecard_model.json"
)

print("\nModel Metadata Summary:")
print(model_parameters["model_metadata"])

print("\nCalibrated Score Decile Thresholds:")
for decile, min_score in model_parameters["decile_min_thresholds"].items():
    print(f"Decile {decile:2s} -> Minimum Passing Score: {min_score}")
```
## 7. Notes

### 1. Why can't we produce OR-based rules?

Allowing OR operations within the search layer causes a massive combinatorial explosion that makes algorithmic pruning impossible. The engine relies on the Apriori property (an AND intersection), where a higher-order rule can be safely skipped if its lower-order components fail the baseline performance thresholds. If OR logic is introduced, a higher-order combination could still clear the threshold even if its individual parts fail, completely breaking the pruning heuristic and forcing an exhaustive, computationally prohibitive search.

Additionally, while single multi-way rules can handle internal OR states for categorical fields (implemented via SQL IN clauses), introducing cross-variable OR conditions natively within the same step complicates the sequential deletion process, making it much harder to cleanly extract distinct, high-risk populations. 

The OR Mutually Exclusive Clarification: Technically, an OR rule could be forced to be mutually exclusive if you deleted anyone who met any part of the OR condition. The real problem with OR is that it breaks the Apriori pruning math. If Rule A fails and Rule B fails, Rule A AND Rule B is guaranteed to fail (Apriori works). But Rule A OR Rule B might succeed, meaning you can no longer drop failed features from your search space.

### 2. Why is Segment $n+k$ (e.g., Segment 3) sometimes better than Segment $n$ (e.g., Segment 2) in terms of lift or other KPIs when evaluated on the full dataset?

The segment extraction process is entirely sequential and operates on a shrinking residual population. Once a champion rule is discovered, its matching records are deleted from the working environment before the next iteration begins.
Because of this cascading extraction:  
    **`Local Optimization`**: The engine optimizes parameters and evaluates candidates based purely on the residual portfolio left behind by previous segments. A rule that yields massive lift on a specific, purified subset of data might look less dominant if it had been evaluated against the noisy baseline of the entire original population.  
    **`Changing Base Rates`**: As high-risk or high-performing records are stripped away in early rounds, the baseline event rate of the remaining pool shifts dynamically. This shifting baseline changes the mathematical benchmark for what constitutes a "high-lift" rule during that specific loop.  Consequently, when evaluate_final_coverage maps all rules simultaneously back over the original, unfiltered dataset, the global KPIs can naturally surface instances where a later segment outperforms an earlier one.  

### 3. My dataset is not zero inflated, still my deciles 3 onwards the floor is zero?

This happens when your extraction criteria are so restrictive that your final segments capture only a tiny fraction of the total population. Even though your input data is healthy, the final scored data becomes artificially zero-inflated because the vast majority of your rows fail to qualify for any segment rules and receive a baseline score of exactly 0.  
When the engine sorts the entire population from highest to lowest score, the small group of customers who actually triggered rules get pushed into Deciles 1 and 2. Because the remaining 80%+ of the population all have a score of 0, Decile 3 onwards fills up entirely with these unsegmented, lowest-risk customers—collapsing their minimum thresholds to 0.  
This indicates that your segment rules are too strict and lack generalizability. To fix this and distribute your scores more evenly across deciles, you can give the engine more breathing room by applying these adjustments:  
**`Increase max_feature_reuse (e.g., set to 2 or 3)`**: This allows highly predictive features to be reused across different segment combinations instead of being locked out after their first use.

**`Increase top_n_vars (e.g., set to 25 or 30)`**: This expands the pool of candidate features the engine can look at in later iterations.  

**`Relax the param_grid thresholds`**: Lower your minimum min_lift or min_sample_size constraints so that smaller or slightly less concentrated segments can still be captured in later rounds.  

**`Disable diversity constraints (enable_diversity = False)`**: This allows features within the same business category to pair up, unlocking more valid rule combinations.  

### 4. My dataset may contain target leaked feature (100% correaltion with Target). Will it be taken as important feature?

No. The feature will be dropped by Optbinning during segment creation steps.
Furthermore, if you are using `BigQueryFeatureSelector` the feature IV will be marked as 0 and not considered. 
