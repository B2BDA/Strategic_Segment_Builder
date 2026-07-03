"""
Strategic Segmentation & Scorecard Engine (Pure Columnar Architecture)
======================================================================
Combinatorial heuristic segmentation using Optimal Binning, Apriori pruning, 
and vectorized DuckDB scorecard deciling. Completely free of Pandas dependencies.

Author: Bishwarup Biswas + Gemini Pro
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
import pyarrow as pa
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
    """Extracts mutually exclusive, predictive segments from tabular PyArrow Tables.

    Utilizes Optimal Binning to discretize continuous features into monotonic
    Information Value (IV) bins, applying an Apriori-style combinatorial prune
    to surface multi-way rules meeting defined Lift and Volume thresholds.
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

        self.enable_diversity = enable_diversity
        self.enable_1way = enable_1way
        self.enable_2way = enable_2way
        self.enable_3way = enable_3way
        self.feature_groups = feature_groups or {}
        self.ignore_features = ignore_features or []
        self.feature_usage_counts: Dict[str, int] = {}

    @staticmethod
    def _resolve_optb_dtype(table: pa.Table, col: str) -> str:
        """Determines the correct OptBinning data type flag for a PyArrow DataType field."""
        t = table.schema.field(col).type
        if pa.types.is_string(t) or pa.types.is_binary(t) or pa.types.is_dictionary(t):
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

    def _validate_feature_groups(self, table: pa.Table) -> None:
        """Validates that all declared feature group variables exist in the target Table."""
        if not self.feature_groups:
            return

        active_cols = set(table.column_names) - {self.target} - set(self.ignore_features)
        validated_count = 0

        for group, vars_list in self.feature_groups.items():
            for var in vars_list:
                if var not in active_cols:
                    raise ValueError(
                        f"Schema Mismatch: Feature '{var}' declared in group '{group}' "
                        "was not found in the provided PyArrow Table."
                    )
                validated_count += 1

        logger.info(f"Feature group validation passed. ({validated_count} features mapped)")

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

    def compute_iv_ranking(self, table: pa.Table) -> List[Dict[str, Any]]:
        """Calculates Information Value (IV) for all eligible features using Arrow zero-copy views."""

        def _worker(col: str) -> Dict[str, Any]:
            try:
                dtype = self._resolve_optb_dtype(table, col)
                optb = OptimalBinning(name=col, dtype=dtype)
                # Extracted as raw contiguous NumPy view of the underlying Arrow column
                x = table[col].to_numpy()
                y = table[self.target].to_numpy()
                optb.fit(x, y)
                iv_val = optb.binning_table.build().IV.iloc[-1]
                return {"variable": col, "iv": float(iv_val) * 100}
            except Exception as e:
                logger.debug(f"IV computation failed for {col}: {e}")
                return {"variable": col, "iv": 0.0}

        eligible_cols = [
            c for c in table.column_names if c != self.target and c not in self.ignore_features
        ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_worker)(col) for col in eligible_cols
        )

        results.sort(key=lambda val: val["iv"], reverse=True)
        return results

    def create_binned_dict(self, table: pa.Table, variables: List[str]) -> Dict[str, np.ndarray]:
        """Transforms continuous data into discrete optimal binned strings stored in a NumPy Dictionary."""
        binned_dict = {}
        y = table[self.target].to_numpy()

        for col in variables:
            dtype = self._resolve_optb_dtype(table, col)
            optb = OptimalBinning(name=col, dtype=dtype)
            x = table[col].to_numpy()
            optb.fit(x, y)
            binned_dict[col] = optb.transform(x, metric="bins")

        binned_dict[self.target] = y
        return binned_dict

    def _agg_combinations(
        self,
        binned_dict: Dict[str, np.ndarray],
        combo_list: List[Tuple[str, ...]],
        base_rate: float,
    ) -> List[Dict[str, Any]]:
        """Vectorized DuckDB multi-column grouping engine over NumPy buffers."""

        def _process_combo(combo: Tuple[str, ...]) -> List[Dict[str, Any]]:
            cols_str = ", ".join([f'"{c}"' for c in combo])
            query = f"""
                SELECT {cols_str}, COUNT(*) AS cnt, SUM("{self.target}") AS ev
                FROM binned_dict
                GROUP BY {cols_str}
                HAVING COUNT(*) >= {self.min_sample_size}
            """
            try:
                # DuckDB directly reads the out-of-process dictionary zero-copy
                res = duckdb.query(query).fetchall()
            except Exception as e:
                logger.debug(f"DuckDB combination processing failed for {combo}: {e}")
                return []

            valid_matches = []
            for row in res:
                count = row[-2]
                events = row[-1]
                rate = (events / count) * 100.0
                lift = rate / (base_rate * 100.0)

                if lift >= self.min_lift:
                    rule_parts = [f"{combo[idx]}={row[idx]}" for idx in range(len(combo))]
                    rule_str = " & ".join(rule_parts)
                    
                    valid_matches.append({
                        "rule": rule_str,
                        "count": count,
                        "rate": rate,
                        "lift": lift,
                        "combo_vars": combo
                    })
            return valid_matches

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_process_combo)(c) for c in combo_list
        )
        
        return [item for sublist in results if sublist for item in sublist]

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

    def extract_segments(self, table: pa.Table, param_grid: Optional[Dict[str, List[Any]]] = None) -> List[Dict[str, Any]]:
        """Sequentially extracts high-lift segments using an iterative Multi-Threshold Grid Search 
        while executing fully on PyArrow memory states to eliminate structural feature dominance.
        """
        if self.enable_diversity:
            self._validate_feature_groups(table)

        current_table = table
        eligible_cols = [c for c in table.column_names if c != self.target and c not in self.ignore_features]
        self.feature_usage_counts = {col: 0 for col in eligible_cols}

        if param_grid:
            sizes = param_grid.get("min_sample_size", [self.min_sample_size])
            lifts = param_grid.get("min_lift", [self.min_lift])
            experiments = [
                {"min_sample_size": s, "min_lift": l}
                for s, l in itertools.product(sizes, lifts)
            ]
        else:
            experiments = [{"min_sample_size": self.min_sample_size, "min_lift": self.min_lift}]

        for i in range(1, self.max_segments + 1):
            base_res = duckdb.query(f'SELECT AVG("{self.target}"), COUNT(*) FROM current_table').fetchone()
            if not base_res or base_res[1] == 0:
                break
            
            base_rate = base_res[0]
            current_volume = base_res[1]
            
            min_floor_volume = min(exp["min_sample_size"] for exp in experiments)
            if base_rate == 0 or current_volume < min_floor_volume:
                break

            logger.info(
                f"Iteration {i} | Remaining Volume: {current_volume:,} | Base Rate: {base_rate*100:.2f}%"
            )

            # 1. Recalculate dynamic IV ranking using PyArrow zero-copy structures
            iv_ranking = self.compute_iv_ranking(current_table)
            
            # 2. Filter out exhausted features
            allowed_vars = [
                row["variable"] for row in iv_ranking
                if self.feature_usage_counts.get(row["variable"], 0) < self.max_feature_reuse
            ]
            
            top_vars = allowed_vars[:self.top_n_vars]
            if not top_vars:
                logger.warning("All eligible features have been exhausted via max_feature_reuse filters. Aborting.")
                break

            binned_dict = self.create_binned_dict(current_table, top_vars)
            
            valid_vars = []
            for v in top_vars:
                uniq_cnt = duckdb.query(f'SELECT COUNT(DISTINCT "{v}") FROM binned_dict').fetchone()[0]
                if uniq_cnt > 1:
                    valid_vars.append(v)
            
            grid_candidates = []

            # 3. Parameter Grid Matrix sweep via DuckDB Aggregations
            for config in experiments:
                self.min_sample_size = config["min_sample_size"]
                self.min_lift = config["min_lift"]

                all_rules: List[Dict[str, Any]] = []

                # Apriori Level 1 (Singles)
                res_1 = self._agg_combinations(binned_dict, [(c,) for c in valid_vars], base_rate)
                valid_1way_vars = set()
                if res_1:
                    valid_1way_vars = {r["combo_vars"][0] for r in res_1}
                    if self.enable_1way:
                        all_rules.extend(res_1)

                if not valid_1way_vars:
                    continue

                # Apriori Level 2 (Pairs)
                valid_2way_sets = set()
                if len(valid_1way_vars) >= 2 and (self.enable_2way or self.enable_3way):
                    combos_2 = [c for c in combinations(valid_1way_vars, 2) if self.is_diverse(c)]
                    if combos_2:
                        res_2 = self._agg_combinations(binned_dict, combos_2, base_rate)
                        if res_2:
                            valid_2way_sets = {frozenset(r["combo_vars"]) for r in res_2}
                            if self.enable_2way:
                                all_rules.extend(res_2)

                # Apriori Level 3 (Triplets)
                if self.enable_3way and len(valid_1way_vars) >= 3 and valid_2way_sets:
                    combos_3 = [
                        c for c in combinations(valid_1way_vars, 3)
                        if self.is_diverse(c) and all(frozenset(p) in valid_2way_sets for p in combinations(c, 2))
                    ]
                    if combos_3:
                        res_3 = self._agg_combinations(binned_dict, combos_3, base_rate)
                        if res_3:
                            all_rules.extend(res_3)

                if all_rules:
                    all_rules.sort(key=lambda val: (val["lift"], val["count"], val["rate"]), reverse=True)
                    top_match = all_rules[0].copy()
                    top_match["grid_min_sample_size"] = config["min_sample_size"]
                    top_match["grid_min_lift"] = config["min_lift"]
                    grid_candidates.append(top_match)

            if not grid_candidates:
                logger.info("No active candidates cleared criteria pool across grid variations. Stopping.")
                break

            # 4. Extract Champion Rule Across Configuration Grid Sets
            grid_candidates.sort(key=lambda val: (val["lift"], val["count"], val["rate"]), reverse=True)
            best_match = grid_candidates[0]
            
            best_rule = best_match["rule"]
            best_sql = self.parse_rule_to_sql(best_rule)
            winning_combo = best_match["combo_vars"]

            # 5. Commit Feature Adoption Metrics
            for var in winning_combo:
                self.feature_usage_counts[var] += 1
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
            
            # Slice down population using zero-copy dynamic PyArrow allocations
            current_table = duckdb.query(
                f"SELECT * FROM current_table WHERE NOT ({best_sql})"
            ).arrow()

        return self.segments

    def evaluate_final_coverage(self, original_table: pa.Table) -> List[Dict[str, Any]]:
        """Executes a full CASE WHEN query over the PyArrow Table source to map mutually exclusive coverage."""
        if not self.segments:
            return []

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
                SUM("{self.target}") AS target_events,
                (SUM("{self.target}") * 100.0 / COUNT(*)) AS response_rate
            FROM original_table
            GROUP BY 1
        ),
        BASE_KPIS AS (
            SELECT *,
                SUM(total_count) OVER() AS total_population,
                (SUM(target_events) OVER() * 1.0 / SUM(total_count) OVER()) * 100 AS base_response_rate 
            FROM PER_SEG_KPIS
        )
        SELECT 
            PER_SEG_KPIS.segment,
            PER_SEG_KPIS.total_count,
            PER_SEG_KPIS.target_events,
            PER_SEG_KPIS.response_rate,
            BASE_KPIS.base_response_rate,
            (PER_SEG_KPIS.total_count * 1.0 / BASE_KPIS.total_population) * 100 AS capture_rate,
            (PER_SEG_KPIS.response_rate / BASE_KPIS.base_response_rate) AS lift
        FROM PER_SEG_KPIS
        LEFT JOIN BASE_KPIS ON PER_SEG_KPIS.segment = BASE_KPIS.segment
        ORDER BY segment
        """
        rel = duckdb.query(final_query)
        cols = rel.columns
        return [dict(zip(cols, row)) for row in rel.fetchall()]


class StrategicSegmentScore:
    """High-Throughput Vectorized Scorecard Engine.

    Computes segment weights via Harmonic Mean and applies dot-product deciling
    over large PyArrow datasets using optimized DuckDB aggregations and NumPy BLAS operations.
    """

    def __init__(
        self, target_col: str, primary_key: str, segment_cols: List[str]
    ) -> None:
        self.target_col = target_col
        self.primary_key = primary_key
        self.segment_cols = segment_cols
        self.model_artifact: Dict[str, Any] = {}

    def calculate_and_export_weights(
        self, table: pa.Table, export_path: str = "scorecard_model.json"
    ) -> Dict[str, Any]:
        """Calculates harmonic weights and derives decile boundaries via vectorized execution."""
        logger.info(f"Initializing DuckDB scorecard engine for {len(table):,} records...")

        ctx = duckdb.connect()

        # Vectorized multi-segment aggregation via zero-copy scan on Arrow
        agg_expressions = [
            f'COUNT(CASE WHEN "{col}" = 1 THEN 1 END) AS "{col}_cnt", '
            f'SUM(CASE WHEN "{col}" = 1 THEN "{self.target_col}" ELSE 0 END) AS "{col}_ev"'
            for col in self.segment_cols
        ]

        master_sql = f"""
            SELECT 
                COUNT(*) AS total_pop, 
                SUM("{self.target_col}") AS total_ev,
                {', '.join(agg_expressions)}
            FROM table
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

        logger.info("Computing harmonic scorecard weights...")
        weights_lookup: Dict[str, Dict[str, Union[int, float]]] = {}

        for idx, seg_col in enumerate(self.segment_cols):
            seg_count = master_res[2 + (idx * 2)] or 0
            seg_events = master_res[2 + (idx * 2) + 1] or 0

            if seg_count == 0 or seg_events == 0:
                logger.warning(f"Segment '{seg_col}' has zero volume or events. Setting weight=0.")
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

        logger.info("Scoring training dataset via NumPy Linear Algebra engine...")
        scored_cols = list(weights_lookup.keys())
        weights_vector = np.array(
            [weights_lookup[c]["weight"] for c in scored_cols], dtype=np.float64
        )

        # Efficient horizontal stack directly from Arrow memory to feed raw C-BLAS
        X = np.column_stack([table[c].to_numpy() for c in scored_cols])
        train_scores = X @ weights_vector

        logger.info(f"Dataset Zero-Inflation Rate: {zero_inflation_rate:.2%}")

        if zero_inflation_rate >= 0.80:
            logger.info("High Zero-Inflation (>=80%). Isolating Active Population...")
            active_scores = train_scores[train_scores > 0]
        else:
            logger.info("Normal Distribution (<80%). Deciling across full dataset...")
            active_scores = train_scores

        if len(active_scores) == 0:
            raise ValueError("Scorecard Failure: 0 customers triggered any segment rules.")

        logger.info(f"Calibrating deciles across {len(active_scores):,} target customers...")
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
                "active_population_pct": round((active_pop_size / total_population) * 100.0, 2),
                "baseline_event_rate": round(baseline_rate, 4),
            },
            "segment_weights": weights_lookup,
            "decile_min_thresholds": decile_thresholds,
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(self.model_artifact, f, indent=4)

        return self.model_artifact