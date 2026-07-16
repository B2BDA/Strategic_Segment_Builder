"""
Strategic Segmentation Engine
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
import os
import psutil
import duckdb
import numpy as np
from joblib import Parallel, delayed
from optbinning import OptimalBinning
from datetime import datetime

# 1. Get current date and time
now = datetime.now()

# Example output: "2026_07_11_22_24_30"
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

# Configure Production Module Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
)
logger = logging.getLogger("StrategicEngine")

# Pre-compile regex at module load for O(1) lookup inside loops
_BRACKET_REGEX = re.compile(r"\[(.*?)\]", flags = re.DOTALL)


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
        min_events: Minimum number of positive events required for a valid rule (fallback default).
        top_n_vars: Number of highest-IV features passed into the Apriori engine.
        max_segments: Hard stopping ceiling for extracted mutually exclusive segments.
        max_feature_reuse: Structural limit for tracking and restricting feature dominance.
        param_grid: Optional dictionary of parameter grid search values for min_sample_size and min_lift.
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
        min_events: int = 5,
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
        self.min_events = min_events
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
        # New: Diagnostic repository
        self.diagnostics_: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_optb_dtype(duckdb_type: str) -> str:
        """Determines the correct OptBinning data type flag from a DuckDB type string."""
        dtype_upper = duckdb_type.upper()
        if any(t in dtype_upper for t in ["VARCHAR", "CHAR", "STRING", "TEXT", "UUID"]):
            return "categorical"
        return "numerical"

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

    def compute_iv_ranking_and_bin(
        self, con: duckdb.DuckDBPyConnection, eligible_cols: List[str], columns_types: Dict[str, str]
    ) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
        """Calculates Information Value (IV) and returns pre-computed bins in a single pass."""
        
        def _worker(col: str) -> Tuple[str, float, Optional[np.ndarray]]:
            try:
                # FIX: Thread-safe cursor implementation to prevent connection corruption
                thread_con = con.cursor()
                data_dict = thread_con.execute(f'SELECT "{col}", "{self.target}" FROM current_df').fetchnumpy()
                col_arr = data_dict[col]
                target_arr = data_dict[self.target]
                dtype = self._resolve_optb_dtype(columns_types[col])
                
                optb = OptimalBinning(name=col, dtype=dtype)
                optb.fit(col_arr, target_arr)
                
                iv_val = optb.binning_table.build()["IV"].values[-1]
                transformed_bins = optb.transform(col_arr, metric="bins").astype(str)
                
                thread_con.close()
                return col, float(iv_val) * 100, transformed_bins
            except Exception as e:
                logger.debug(f"IV computation failed for {col}: {e}")
                return col, 0.0, None

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_worker)(col) for col in eligible_cols
        )
        
        ranking = []
        precomputed_bins = {}
        for col, iv, bins in results:
            ranking.append({"variable": col, "iv": iv})
            if bins is not None:
                precomputed_bins[col] = bins
                
        ranking.sort(key=lambda x: x["iv"], reverse=True)
        return ranking, precomputed_bins

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
                    AND SUM(CAST("{self.target}" AS DOUBLE)) >= {self.min_events}
            """
            queries.append(query)

        valid_results = []
        chunk_size = 100  # Increased step size to leverage broader hardware parallelism
        
        # Batch execute independent group bys natively in C++ via UNION ALL
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i+chunk_size]
            union_query = " UNION ALL ".join(chunk)
            
            res = con.execute(union_query).fetchall()
            for rule, count, events, combo_vars_str in res:
                rate = (events / count) * 100.0 if count > 0 else 0
                lift = rate / (base_rate * 100.0) if base_rate > 0 else 0
                
                # Require at least self.min_events events and lift > 1.0

                if lift >= self.min_lift and events >= self.min_events and lift > 1.0:
                    valid_results.append({
                        "rule": rule,
                        "count": count,
                        "rate": rate,
                        "lift": lift,
                        "events": events,
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

            # Robust Categorical Set Handling using AST Literal Parsing
            if is_categorical and bracket_match:
                import ast
                try:
                    # Safely convert the string representation of a list into an actual Python list
                    raw_items = ast.literal_eval(bracket_match.group(0))
                except Exception:
                    # Handle string-truncated categorical buckets (e.g. missing commas ['A' 'B'])
                    raw_content = bracket_match.group(1)
                    if "," not in raw_content:
                        raw_content = re.sub(r"'\s+'", "','", raw_content)
                        raw_content = re.sub(r'"\s+"', '","', raw_content)
                        raw_content = re.sub(r'\s+', ',', raw_content) # Aggressive fallback
                        
                    raw_items = [
                        i.strip().strip("'").strip('"')
                        for i in raw_content.split(",")
                        if i.strip()
                    ]
                
                # If the item is a true string instance, wrap it in SQL quotes. Otherwise, treat it as a raw literal number.
                formatted_items = ", ".join(
                    [
                        f"'{item}'" if isinstance(item, str) else str(item)
                        for item in raw_items
                    ]
                )
                
                if formatted_items:
                    # Clean output: No double quotes around column names
                    sql_conditions.append(f"{col} IN ({formatted_items})")
                continue

            # 2. Null/Special State Handling
            if interval in ["Special", "Missing"]:
                # Clean output: No double quotes around column names
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
        # Cache true global hard floors before the grid sweeps alter them
        abs_min_sample_size = self.min_sample_size
        abs_min_events = self.min_events

        # Memory Optimization: Use file-backed storage
        if os.path.exists(f"experiment_{timestamp}.db"):
            os.remove(f"experiment_{timestamp}.db")
        con = duckdb.connect(f"experiment_{timestamp}.db")
        
        # 1. Dynamically calculate available CPU cores
        total_cores = os.cpu_count() or 1
        target_threads = max(1, total_cores - 2) if total_cores > 4 else total_cores
        
        # 2. Dynamically calculate memory limit (in bytes)
        total_memory_bytes = psutil.virtual_memory().total
        target_memory_gb = int((total_memory_bytes * 0.5) / (1024 ** 3))
        
        # 3. Apply the settings dynamically to DuckDB
        con.execute(f"SET threads = {target_threads};")
        con.execute(f"SET memory_limit = '{target_memory_gb}GB';")
    
        logger.info(f"DuckDB Configured: Threads={target_threads}/{total_cores}, MemoryLimit={target_memory_gb}GB")

        con.execute("CREATE OR REPLACE TABLE current_df AS SELECT * FROM data")

        cols_info = con.execute("DESCRIBE current_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}
        all_cols = list(columns_types.keys())

        if self.enable_diversity:
            self._validate_feature_groups(all_cols)

        eligible_cols = [c for c in all_cols if c != self.target and c not in self.ignore_features]
        self.feature_usage_counts = {col: 0 for col in eligible_cols}
        
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

            # Speed Tweak: Single-pass pipeline computes IV and structures transformation bins at the same time
            iv_ranking, precomputed_bins = self.compute_iv_ranking_and_bin(con, eligible_cols, columns_types)

            # --- START DIAGNOSTIC CAPTURE ---
            current_iv_map = {row["variable"]: row["iv"] for row in iv_ranking}
            top_n_variable_names = [r["variable"] for r in iv_ranking[:self.top_n_vars]]
            
            iteration_snapshot = {}
            for col in eligible_cols:
                used_count = self.feature_usage_counts.get(col, 0)
                current_iv = current_iv_map.get(col, 0.0)
                
                if used_count >= self.max_feature_reuse:
                    status = "Excluded (Max Feature Reuse Exceeded)"
                elif current_iv <= 0.0:
                    status = "Excluded (Information Value is Zero/Invalid)"
                elif col not in top_n_variable_names:
                    status = "Excluded (Outside Top N Features by IV)"
                else:
                    status = "Eligible for Combination Search"
                    
                iteration_snapshot[col] = {
                    "iv": current_iv,
                    "times_used_previously": used_count,
                    "status": status
                }
            
            self.diagnostics_.append({
                "iteration": i,
                "residual_volume": current_volume,
                "base_rate": base_rate,
                "features_state": iteration_snapshot,
                "winning_segment": None 
            })
            # --- END DIAGNOSTIC CAPTURE ---
            
            allowed_vars = [
                row["variable"] for row in iv_ranking
                if self.feature_usage_counts.get(row["variable"], 0) < self.max_feature_reuse
            ]
            
            top_vars = allowed_vars[:self.top_n_vars]
            if not top_vars:
                logger.warning("All eligible features have been exhausted via max_feature_reuse filters. Aborting.")
                break

            # Speed Tweak: Load the arrays from memory without another fit cycle
            binned_data = {self.target: con.execute(f'SELECT "{self.target}" FROM current_df').fetchnumpy()[self.target]}
            valid_vars = []
            for v in top_vars:
                if v in precomputed_bins:
                    if len(np.unique(precomputed_bins[v])) > 1:
                        binned_data[v] = precomputed_bins[v]
                        valid_vars.append(v)

            con.execute("DROP TABLE IF EXISTS binned_df")
            con.execute("CREATE TABLE binned_df AS SELECT * FROM binned_data")
            
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
                    # Sort candidates natively in python to avoid DataFrame serialization overhead
                    all_rules.sort(key=lambda x: (x["lift"], x["rate"], x["count"]), reverse=True)
                    top_match = all_rules[0].copy()
                    top_match["grid_min_sample_size"] = config["min_sample_size"]
                    top_match["grid_min_lift"] = config["min_lift"]
                    grid_candidates.append(top_match)

            if not grid_candidates:
                logger.info("No active candidates cleared criteria pool across grid variations. Stopping.")
                break

            # 4. Resolve Championship Rule Across Parameter Configuration Grid Result Sets
            # FIX: Sorted primarily by volume to ensure high-lift, ultra-low-sample subsets don't override robust sets.
            grid_candidates.sort(key=lambda x: (x["count"], x["lift"], x["rate"]), reverse=True)
            best_match = grid_candidates[0]
            # ==================================================================
            # 🛑 ABSOLUTE SAFETY GATE
            # ==================================================================
            # If a micro-config (like size=10) games the system and wins the grid selection,
            # this check forcefully catches it and drops the execution line here.
            if (best_match["count"] < abs_min_sample_size or best_match["events"] < abs_min_events):
                logger.warning(
                    f"Iteration {i} | Winning candidate rejected by absolute safety gate. "
                    f"Required >= {abs_min_sample_size} rows and >= {abs_min_events} events. "
                    f"Got {best_match['count']:,} rows and {best_match['events']} events instead. Halting."
                )
                break
            # ==================================================================
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
            
            self.diagnostics_[-1]["winning_segment"] = {
                "rule": best_rule,
                "sql_filter": best_sql,
                "variables_used": list(winning_combo),
                "lift": float(best_match["lift"]),
                "count": int(best_match["count"])
            }

            # Memory Optimization: Copy and Swap prevents dead-tuple fragmentation
            con.execute(f"""
                CREATE TABLE temp_residual AS 
                SELECT * FROM current_df 
                WHERE NOT ({best_sql})
            """)
            con.execute("DROP TABLE current_df")
            con.execute("ALTER TABLE temp_residual RENAME TO current_df")

        # 🟢 CLEANUP: Restore the original configuration values back to the instance
        self.min_sample_size = abs_min_sample_size
        self.min_events = abs_min_events
        
        con.close()
        return self.segments    

    def evaluate_final_coverage(self, original_data: Any) -> List[Dict[str, Any]]:
        """Executes a full CASE WHEN query over the source dataset to map mutually exclusive coverage."""
        if not self.segments:
            return []
            
        con = duckdb.connect(f"experiment_{timestamp}.db")
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
        # Make sure to close connection before returning
        res_list = [dict(zip(columns, row)) for row in res.fetchall()]
        con.close()
        return res_list
    
    def explain_feature_journey(self, feature_name: str) -> None:
        """Prints a detailed audit trail of a specific feature across all iterations."""
        if not self.diagnostics_:
            print("No diagnostic records found. Run extract_segments() first.")
            return
            
        print("=" * 80)
        print(f"AUDIT TRAIL FOR FEATURE: '{feature_name}'")
        print("=" * 80)
        
        for record in self.diagnostics_:
            iter_num = record["iteration"]
            state = record["features_state"].get(feature_name)
            winner = record["winning_segment"]
            
            if not state:
                print(f"Iteration {iter_num}: Variable not present or was ignored.")
                continue
                
            print(f"\n[Iteration {iter_num}]")
            print(f"  • Current dynamic IV   : {state['iv']:.4f}")
            print(f"  • Previous times used  : {state['times_used_previously']}")
            print(f"  • Selection Status     : {state['status']}")
            
            if winner and feature_name in winner["variables_used"]:
                print(f"  🎉 SELECTED as part of winning rule!")
                print(f"     Rule: {winner['rule']}")
            elif winner:
                print(f"  • Winner this round    : {winner['rule']} (Variables: {winner['variables_used']})")
        print("=" * 80)


