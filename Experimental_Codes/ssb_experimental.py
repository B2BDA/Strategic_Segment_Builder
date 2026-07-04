"""
Strategic Segmentation & Scorecard Engine
=========================================
Combinatorial heuristic segmentation using Optimal Binning, Apriori pruning, 
and vectorized DuckDB scorecard deciling.

Author: Bishwarup Biswas + Gemini
Python Version: 3.9+
"""

import json
import logging
import multiprocessing
import re
import itertools
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
from joblib import Parallel, delayed
from optbinning import OptimalBinning

# Configure Production Module Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
)
logger = logging.getLogger("StrategicEngine")

# Pre-compile regex at module load for O(1) lookup inside loops
_BRACKET_REGEX = re.compile(r"\[(.*)\]")


class StrategicSegmentBuilder:
    """Extracts mutually exclusive, predictive segments from tabular data.

    Utilizes Optimal Binning to discretize continuous features into monotonic
    Information Value (IV) bins, applying an Apriori-style combinatorial prune
    to surface multi-way rules meeting defined Lift and Volume thresholds.

    Attributes:
        target: Dependent binary target column name (1 = Event, 0 = Non-Event).
        n_jobs: Number of CPU cores allocated to parallelized search jobs.
        min_sample_size: Absolute minimum row count required for a valid rule (fallback default).
        min_lift: Minimum lift cutoff (Segment Rate / Population Base Rate) (fallback default).
        top_n_vars: Number of highest-IV features passed into the Apriori engine.
        max_segments: Hard stopping ceiling for extracted mutually exclusive segments.
        max_feature_reuse: Structural limit for tracking and restricting feature dominance.
        enable_diversity: If True, blocks rules combining variables from the same business group.
        enable_1way: Allow 1-dimensional rules in final pool.
        enable_2way: Allow 2-dimensional intersection rules in final pool.
        enable_3way: Allow 3-dimensional intersection rules in final pool.
        feature_groups: Mapping of business categories to columns (e.g. {'risk': ['scr', 'bal']}).
        ignore_features: Explicit list of columns to drop prior to IV calculation.
    """

    def __init__(
        self,
        target: str,
        n_jobs: int = -1,
        min_sample_size: int = 1000,
        min_lift: float = 2.0,
        top_n_vars: int = 20,
        max_segments: int = 10,
        max_feature_reuse: int = 1,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        enable_diversity: bool = False,
        enable_1way: bool = True,
        enable_2way: bool = True,
        enable_3way: bool = True,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        ignore_features: Optional[List[str]] = None,
    ) -> None:
        self.target = target
        self.n_jobs = (
            n_jobs if n_jobs != -1 else max(1, multiprocessing.cpu_count() - 1)
        )
        self.min_sample_size = min_sample_size
        self.min_lift = min_lift
        self.top_n_vars = top_n_vars
        self.max_segments = max_segments
        self.max_feature_reuse = max_feature_reuse
        self.segments: List[Dict[str, Any]] = []
        self.param_grid = param_grid or {}
        self.enable_diversity = enable_diversity
        self.enable_1way = enable_1way
        self.enable_2way = enable_2way
        self.enable_3way = enable_3way
        self.feature_groups = feature_groups or {}
        self.ignore_features = ignore_features or []
        self.feature_usage_counts: Dict[str, int] = {}

    @staticmethod
    def _resolve_optb_dtype(duckdb_type: str) -> str:
        """Determines the correct OptBinning data type flag from a DuckDB type string."""
        dtype_upper = duckdb_type.upper()
        if any(t in dtype_upper for t in ["VARCHAR", "CHAR", "STRING", "TEXT", "UUID"]):
            return "categorical"
        return "numerical"

    @staticmethod
    def _is_numeric_string(val: str) -> bool:
        """Safely evaluates if a raw string represents a float/int (handles scientific notation)."""
        try:
            float(val)
            return True
        except ValueError:
            return False

    def _validate_feature_groups(self, columns: List[str]) -> None:
        """Validates that all declared feature group variables exist in the target dataset."""
        if not self.feature_groups:
            return

        active_cols = set(columns) - {self.target} - set(self.ignore_features)
        validated_count = 0

        for group, vars_list in self.feature_groups.items():
            for var in vars_list:
                if var not in active_cols:
                    raise ValueError(
                        f"Schema Mismatch: Feature '{var}' declared in group '{group}' "
                        "was not found in the provided DataFrame/Table."
                    )
                validated_count += 1

        logger.info(
            f"Feature group validation passed. ({validated_count} features mapped)"
        )

    def get_group(self, var: str) -> str:
        """Returns the assigned business category for a feature, or the feature name itself."""
        for group, vars_list in self.feature_groups.items():
            if var in vars_list:
                return group
        return var

    def is_diverse(self, combo: Tuple[str, ...]) -> bool:
        """Ensures a tuple of features spans strictly distinct analytical groups."""
        if not self.enable_diversity:
            return True
        groups = [self.get_group(v) for v in combo]
        return len(groups) == len(set(groups))

    def compute_iv_ranking(self, con: duckdb.DuckDBPyConnection) -> List[Dict[str, Union[str, float]]]:
        """Calculates Information Value (IV) for all eligible features using natively fetched numpy arrays."""
        
        cols_info = con.execute("DESCRIBE current_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}
        eligible_cols = [c for c in columns_types.keys() if c != self.target and c not in self.ignore_features]

        # Fetch dictionary of flat NumPy arrays for fast, zero-copy serialization across joblib workers
        data_dict = con.execute("SELECT * FROM current_df").fetchnumpy()

        def _worker(col: str) -> Dict[str, Union[str, float]]:
            try:
                col_arr = data_dict[col]
                target_arr = data_dict[self.target]
                dtype = self._resolve_optb_dtype(columns_types[col])
                
                optb = OptimalBinning(name=col, dtype=dtype)
                optb.fit(col_arr, target_arr)
                
                # Extract IV value directly without needing a pandas import in this file
                iv_val = optb.binning_table.build()["IV"].values[-1]
                return {"variable": col, "iv": float(iv_val) * 100}
            except Exception as e:
                logger.debug(f"IV computation failed for {col}: {e}")
                return {"variable": col, "iv": 0.0}

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_worker)(col) for col in eligible_cols
        )
        
        return sorted(results, key=lambda x: x["iv"], reverse=True)

    def create_binned_table(self, con: duckdb.DuckDBPyConnection, variables: List[str]) -> None:
        """Transforms continuous data into discrete optimal binned strings natively mapped in DuckDB."""
        data_dict = con.execute("SELECT * FROM current_df").fetchnumpy()
        binned_data = {self.target: data_dict[self.target]}
        
        cols_info = con.execute("DESCRIBE current_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}

        for col in variables:
            col_arr = data_dict[col]
            target_arr = data_dict[self.target]
            dtype = self._resolve_optb_dtype(columns_types[col])
            
            optb = OptimalBinning(name=col, dtype=dtype)
            optb.fit(col_arr, target_arr)

            transformed_bins = optb.transform(col_arr, metric="bins")
            binned_data[col] = transformed_bins.astype(str)

        # DuckDB resolves local dictionary variables automatically
        con.execute("DROP TABLE IF EXISTS binned_df")
        con.execute("CREATE TABLE binned_df AS SELECT * FROM binned_data")

    def _agg_combinations(
        self,
        con: duckdb.DuckDBPyConnection,
        combo_list: List[Tuple[str, ...]],
        base_rate: float,
    ) -> List[Dict[str, Any]]:
        """Batch-executes SQL combinatorics via DuckDB GROUP BY to bypass slow Pandas operations."""
        if not combo_list:
            return []

        queries = []
        for combo in combo_list:
            cols_str = ", ".join([f'"{c}"' for c in combo])
            rule_concat = " || ' & ' || ".join([f"'{c}=' || CAST(\"{c}\" AS VARCHAR)" for c in combo])
            combo_str = ",".join(combo)
            
            query = f"""
                SELECT 
                    {rule_concat} AS rule,
                    COUNT("{self.target}")::BIGINT AS count,
                    SUM(CAST("{self.target}" AS DOUBLE)) AS events,
                    '{combo_str}' AS combo_vars_str
                FROM binned_df
                GROUP BY {cols_str}
                HAVING COUNT("{self.target}") >= {self.min_sample_size}
            """
            queries.append(query)

        valid_results = []
        chunk_size = 50
        
        # Batch execute independent group bys natively in C++ via UNION ALL
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i+chunk_size]
            union_query = " UNION ALL ".join(chunk)
            
            res = con.execute(union_query).fetchall()
            for rule, count, events, combo_vars_str in res:
                rate = (events / count) * 100.0 if count > 0 else 0
                lift = rate / (base_rate * 100.0) if base_rate > 0 else 0
                
                if lift >= self.min_lift:
                    valid_results.append({
                        "rule": rule,
                        "count": count,
                        "rate": rate,
                        "lift": lift,
                        "combo_vars": tuple(combo_vars_str.split(","))
                    })

        return valid_results

    def parse_rule_to_sql(self, rule_str: str) -> str:
        """Translates OptBinning string syntax into a production SQL WHERE clause."""
        parts = [p.strip() for p in rule_str.split("&")]
        sql_conditions: List[str] = []

        for part in parts:
            if "=" not in part:
                continue

            col, interval = [x.strip() for x in part.split("=", 1)]
            bracket_match = _BRACKET_REGEX.search(interval)

            is_categorical = False
            if bracket_match:
                content = bracket_match.group(1)
                if any(
                    k in interval for k in ("'", '"', "Array", "Categorical")
                ) or not interval.startswith(("[", "(")):
                    is_categorical = True
                elif len(content.split(",")) > 2:
                    is_categorical = True

            # 1. Categorical Set Handling
            if is_categorical and bracket_match:
                raw_items = [
                    i.strip().strip("'").strip('"')
                    for i in bracket_match.group(1).split(",")
                    if i.strip()
                ]
                formatted_items = ", ".join(
                    [
                        item if self._is_numeric_string(item) else f"'{item}'"
                        for item in raw_items
                    ]
                )
                sql_conditions.append(f"{col} IN ({formatted_items})")
                continue

            # 2. Null/Special State Handling
            if interval in ["Special", "Missing"]:
                sql_conditions.append(f"{col} IS NULL")
                continue

            # 3. Continuous Numeric Range Handling
            if interval.startswith(("[", "(")):
                left_char, right_char = interval[0], interval[-1]
                lower_str, upper_str = [x.strip() for x in interval[1:-1].split(",", 1)]

                range_conds = []
                if lower_str.lower() != "-inf":
                    op = ">=" if left_char == "[" else ">"
                    range_conds.append(f"{col} {op} {lower_str}")

                if upper_str.lower() != "inf":
                    op = "<=" if right_char == "]" else "<"
                    range_conds.append(f"{col} {op} {upper_str}")

                if range_conds:
                    sql_conditions.append(" AND ".join(range_conds))

        return " AND ".join(
            f"({cond})" if "AND" in cond else cond for cond in sql_conditions
        )

    def extract_segments(self, data: Any) -> List[Dict[str, Any]]:
        """Sequentially extracts high-lift segments using an iterative Multi-Threshold Grid Search 
        while applying feature usage constraints to eliminate structural feature dominance.
        """
        con = duckdb.connect()
        # DuckDB resolves local Python variables automatically (handles Pandas, Arrow, Dictionaries)
        con.execute("CREATE TABLE current_df AS SELECT * FROM data")

        cols_info = con.execute("DESCRIBE current_df").fetchall()
        all_cols = [row[0] for row in cols_info]

        if self.enable_diversity:
            self._validate_feature_groups(all_cols)

        # Initialize global tracking map for tracking structural dominance
        eligible_cols = [c for c in all_cols if c != self.target and c not in self.ignore_features]
        self.feature_usage_counts = {col: 0 for col in eligible_cols}
        
        # Build dynamic grid search boundaries
        if self.param_grid:
            logger.info(
                f"Dynamic Grid Search Enabled: {len(self.param_grid.get('min_sample_size', [self.min_sample_size])) * len(self.param_grid.get('min_lift', [self.min_lift]))} total configurations."
            )
            sizes = self.param_grid.get("min_sample_size", [self.min_sample_size])
            lifts = self.param_grid.get("min_lift", [self.min_lift])
            experiments = [
                {"min_sample_size": s, "min_lift": l}
                for s, l in itertools.product(sizes, lifts)
            ]
        else:
            experiments = [{"min_sample_size": self.min_sample_size, "min_lift": self.min_lift}]

        for i in range(1, self.max_segments + 1):
            res = con.execute(f'SELECT AVG("{self.target}"), COUNT(*) FROM current_df').fetchone()
            base_rate, current_volume = res[0] or 0.0, res[1] or 0

            min_floor_volume = min(exp["min_sample_size"] for exp in experiments)
            
            if base_rate == 0 or current_volume < min_floor_volume:
                break

            logger.info(
                f"Iteration {i} | Remaining Volume: {current_volume:,} | Base Rate: {base_rate*100:.2f}%"
            )

            # 1. Recalculate Dynamic IV Ranking on residual portfolio population
            iv_ranking = self.compute_iv_ranking(con)
            
            # 2. Apply Dominance Constraints: Filter out exhausted features
            allowed_vars = [
                row["variable"] for row in iv_ranking
                if self.feature_usage_counts.get(row["variable"], 0) < self.max_feature_reuse
            ]
            
            top_vars = allowed_vars[:self.top_n_vars]
            if not top_vars:
                logger.warning("All eligible features have been exhausted via max_feature_reuse filters. Aborting.")
                break

            self.create_binned_table(con, top_vars)
            
            valid_vars = []
            for v in top_vars:
                distinct_count = con.execute(f'SELECT COUNT(DISTINCT "{v}") FROM binned_df').fetchone()[0]
                if distinct_count > 1:
                    valid_vars.append(v)
            
            grid_candidates: List[Dict[str, Any]] = []

            # 3. Parameter Matrix Grid Sweep
            for config in experiments:
                self.min_sample_size = config["min_sample_size"]
                self.min_lift = config["min_lift"]

                all_rules: List[Dict[str, Any]] = []

                # Apriori Level 1 (Singles)
                res_1 = self._agg_combinations(
                    con, [(c,) for c in valid_vars], base_rate
                )
                valid_1way_vars = set()

                if res_1:
                    valid_1way_vars = {c["combo_vars"][0] for c in res_1}
                    if self.enable_1way:
                        all_rules.extend(res_1)

                if not valid_1way_vars:
                    continue

                # Apriori Level 2 (Pairs)
                valid_2way_sets = set()
                if len(valid_1way_vars) >= 2 and (self.enable_2way or self.enable_3way):
                    combos_2 = [
                        c for c in combinations(valid_1way_vars, 2) if self.is_diverse(c)
                    ]
                    if combos_2:
                        res_2 = self._agg_combinations(con, combos_2, base_rate)
                        if res_2:
                            valid_2way_sets = {frozenset(c["combo_vars"]) for c in res_2}
                            if self.enable_2way:
                                all_rules.extend(res_2)

                # Apriori Level 3 (Triplets)
                if self.enable_3way and len(valid_1way_vars) >= 3 and valid_2way_sets:
                    combos_3 = [
                        c
                        for c in combinations(valid_1way_vars, 3)
                        if self.is_diverse(c)
                        and all(
                            frozenset(p) in valid_2way_sets for p in combinations(c, 2)
                        )
                    ]
                    if combos_3:
                        res_3 = self._agg_combinations(con, combos_3, base_rate)
                        if res_3:
                            all_rules.extend(res_3)

                if all_rules:
                    # Sort candidates down natively in python to avoid DataFrame serialization overhead
                    all_rules.sort(key=lambda x: (x["lift"], x["rate"], x["count"]), reverse=True)
                    top_match = all_rules[0].copy()
                    top_match["grid_min_sample_size"] = config["min_sample_size"]
                    top_match["grid_min_lift"] = config["min_lift"]
                    grid_candidates.append(top_match)

            if not grid_candidates:
                logger.info("No active candidates cleared criteria pool across grid variations. Stopping.")
                break

            # 4. Resolve Championship Rule Across Parameter Configuration Grid Result Sets
            grid_candidates.sort(key=lambda x: (x["lift"], x["count"], x["rate"]), reverse=True)
            best_match = grid_candidates[0]
            
            best_rule = best_match["rule"]
            best_sql = self.parse_rule_to_sql(best_rule)
            winning_combo = best_match["combo_vars"]

            # 5. Commit Feature Adoption and Increment Dominance Counter State
            for var in winning_combo:
                self.feature_usage_counts[var] = self.feature_usage_counts.get(var, 0) + 1
                logger.info(f"Feature Usage Tracker Update -> '{var}' used count = {self.feature_usage_counts[var]}")

            self.segments.append(
                {
                    "segment_id": i,
                    "rule_string": best_rule,
                    "sql_filter": best_sql,
                    "count": int(best_match["count"]),
                    "rate": float(best_match["rate"]),
                    "lift": float(best_match["lift"]),
                    "meta_applied_sample_size": int(best_match["grid_min_sample_size"]),
                    "meta_applied_min_lift": float(best_match["grid_min_lift"])
                }
            )

            logger.info(f"Segment {i} Captured (Size Floor: {best_match['grid_min_sample_size']} | Lift Floor: {best_match['grid_min_lift']}): {best_sql}")
            
            # Execute deletion directly on the view instead of copying dataframe slices back and forth
            con.execute(f"DELETE FROM current_df WHERE ({best_sql})")

        return self.segments

    def evaluate_final_coverage(self, original_data: Any) -> List[Dict[str, Any]]:
        """Executes a full CASE WHEN query over the source dataset to map mutually exclusive coverage."""
        if not self.segments:
            return []
            
        con = duckdb.connect()
        con.execute("CREATE TABLE original_df AS SELECT * FROM original_data")

        case_statements = [
            f"WHEN {seg['sql_filter']} THEN {seg['segment_id']}"
            for seg in self.segments
        ]
        case_sql = "\n                ".join(case_statements)

        final_query = f"""
        WITH PER_SEG_KPIS AS (
            SELECT 
                CASE {case_sql} ELSE 0 END AS segment, 
                COUNT(*) AS total_count,
                SUM(CAST("{self.target}" AS DOUBLE)) AS target_events,
                (SUM(CAST("{self.target}" AS DOUBLE)) * 100.0 / COUNT(*)) AS response_rate
            FROM original_df
            GROUP BY 1
        ),
        BASE_KPIS AS (
            SELECT *,
                SUM(total_count) OVER() AS total_population,
                (SUM(target_events) OVER() * 1.0 / SUM(total_count) OVER()) * 100 AS base_response_rate 
            FROM PER_SEG_KPIS
        )
        SELECT 
            PER_SEG_KPIS.*, 
            BASE_KPIS.base_response_rate,
            (PER_SEG_KPIS.total_count * 1.0 / BASE_KPIS.total_population) * 100 AS capture_rate,
            (PER_SEG_KPIS.response_rate / BASE_KPIS.base_response_rate) AS lift
        FROM PER_SEG_KPIS
        LEFT JOIN BASE_KPIS ON PER_SEG_KPIS.segment = BASE_KPIS.segment
        ORDER BY segment
        """
        
        # Native return as a List of Dictionaries
        res = con.execute(final_query)
        columns = [desc[0] for desc in res.description]
        return [dict(zip(columns, row)) for row in res.fetchall()]


class StrategicSegmentScore:
    """High-Throughput Vectorized Scorecard Engine.

    Computes segment weights via Harmonic Mean and applies dot-product deciling
    over large datasets using optimized DuckDB aggregations and NumPy BLAS operations.
    """

    def __init__(
        self, target_col: str, primary_key: str, segment_cols: List[str]
    ) -> None:
        self.target_col = target_col
        self.primary_key = primary_key
        self.segment_cols = segment_cols
        self.model_artifact: Dict[str, Any] = {}

    def calculate_and_export_weights(
        self, data: Any, export_path: str = "scorecard_model.json"
    ) -> Dict[str, Any]:
        """Calculates harmonic weights and derives decile boundaries via vectorized execution."""
        logger.info(f"Initializing DuckDB scorecard engine...")

        ctx = duckdb.connect()
        ctx.execute("CREATE TABLE df AS SELECT * FROM data")

        # Step 1: Baseline metrics + Vectorized multi-segment aggregation (O(1) Scan)
        agg_expressions = [
            f'COUNT(CASE WHEN "{col}" = 1 THEN 1 END) AS "{col}_cnt", '
            f'SUM(CASE WHEN "{col}" = 1 THEN "{self.target_col}" ELSE 0 END) AS "{col}_ev"'
            for col in self.segment_cols
        ]

        master_sql = f"""
            SELECT 
                COUNT(*) AS total_pop, 
                SUM(CAST("{self.target_col}" AS DOUBLE)) AS total_ev,
                {', '.join(agg_expressions)}
            FROM df
        """

        master_res = ctx.execute(master_sql).fetchone()
        if not master_res:
            raise RuntimeError("Database engine failed to return aggregations.")

        total_population, total_events = master_res[0], master_res[1]

        if total_population == 0 or total_events == 0:
            raise ValueError(
                "Invalid Dataset: Population and total events must be greater than zero."
            )

        baseline_rate = total_events / total_population
        zero_inflation_rate = 1.0 - baseline_rate

        # Step 2: Unpack vectorized SQL aggregations into weight lookup
        logger.info("Computing harmonic scorecard weights...")
        weights_lookup: Dict[str, Dict[str, Union[int, float]]] = {}

        for idx, seg_col in enumerate(self.segment_cols):
            # Unpack the specific column offsets from the single master tuple
            seg_count = master_res[2 + (idx * 2)] or 0
            seg_events = master_res[2 + (idx * 2) + 1] or 0

            if seg_count == 0 or seg_events == 0:
                logger.warning(
                    f"Segment '{seg_col}' has zero volume or events. Setting weight=0."
                )
                weights_lookup[seg_col] = {
                    "weight": 0,
                    "lift": 0.0,
                    "response_rate": 0.0,
                    "capture_rate": 0.0,
                }
                continue

            response_rate = seg_events / seg_count
            capture_rate = seg_events / total_events
            lift = response_rate / baseline_rate

            harmonic_mean = 2 * (
                (response_rate * capture_rate) / (response_rate + capture_rate)
            )
            raw_weight = lift * harmonic_mean * 100.0

            weights_lookup[seg_col] = {
                "weight": int(np.round(raw_weight)),
                "lift": round(lift, 4),
                "response_rate": round(response_rate, 4),
                "capture_rate": round(capture_rate, 4),
            }

        # Step 3: BLAS Matrix Dot-Product Scoring
        logger.info("Scoring training dataset via NumPy Linear Algebra engine...")
        scored_cols = list(weights_lookup.keys())
        weights_vector = np.array(
            [weights_lookup[c]["weight"] for c in scored_cols], dtype=np.float64
        )

        query_cols = ", ".join([f'"{c}"' for c in scored_cols])
        
        # Native conversion directly from DuckDB engine to NumPy arrays speeds this up exponentially
        features_dict = ctx.execute(f"SELECT {query_cols} FROM df").fetchnumpy()
        
        # Construct optimized C contiguous 2D Matrix Array bypassing python loop logic
        features_matrix = np.column_stack([features_dict[c] for c in scored_cols])

        # Matrix DOT Operation performs at raw C speed
        train_scores = features_matrix @ weights_vector

        logger.info(f"Dataset Zero-Inflation Rate: {zero_inflation_rate:.2%}")

        if zero_inflation_rate >= 0.80:
            logger.info("High Zero-Inflation (>=80%). Isolating Active Population...")
            active_scores = train_scores[train_scores > 0]
        else:
            logger.info("Normal Distribution (<80%). Deciling across full dataset...")
            active_scores = train_scores

        if len(active_scores) == 0:
            raise ValueError(
                "Scorecard Failure: 0 customers triggered any segment rules."
            )

        # Step 4: High-speed NumPy sorting
        logger.info(
            f"Calibrating deciles across {len(active_scores):,} target customers..."
        )
        sorted_scores = np.sort(active_scores)[::-1]
        active_pop_size = len(sorted_scores)

        decile_thresholds: Dict[str, int] = {}
        for d in range(1, 11):
            row_idx = int((d / 10.0) * active_pop_size) - 1
            row_idx = max(0, min(active_pop_size - 1, row_idx))
            decile_thresholds[str(d)] = int(sorted_scores[row_idx])

        self.model_artifact = {
            "model_metadata": {
                "total_training_population": int(total_population),
                "active_scored_population": int(active_pop_size),
                "active_population_pct": round(
                    (active_pop_size / total_population) * 100.0, 2
                ),
                "baseline_event_rate": round(baseline_rate, 4),
            },
            "segment_weights": weights_lookup,
            "decile_min_thresholds": decile_thresholds,
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(self.model_artifact, f, indent=4)

        return self.model_artifact