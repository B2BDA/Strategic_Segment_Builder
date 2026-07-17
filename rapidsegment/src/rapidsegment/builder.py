"""
Strategic Segmentation Engine
==============================
Combinatorial heuristic segmentation using Optimal Binning, Apriori pruning,
and vectorized DuckDB scorecard deciling.

Author: Bishwarup Biswas + Gemini + DeepSeek
Python Version: 3.9+
"""

import logging
import os
import re
from datetime import datetime
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import psutil
from joblib import Parallel, delayed
from optbinning import OptimalBinning

# -----------------------------------------------------------------------------
# Module-level configuration
# -----------------------------------------------------------------------------
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
)
logger = logging.getLogger("StrategicEngine")

# Pre-compiled regex for fast parsing inside loops
_BRACKET_REGEX = re.compile(r"\[(.*?)\]", flags=re.DOTALL)


class StrategicSegmentBuilder:
    """
    Extracts hierarchical, predictive segments from tabular data.

    The extraction is sequential:
        - At each step, the best rule (by lift and volume) is found on the current residual dataset.
        - The rule is applied to remove those rows before the next iteration.
        - The final segmentation is hierarchical: the first rule has highest priority,
          the second rule applies to the remainder, and so on.

    The `extract_segments()` method returns segments whose counts reflect the
    final hierarchical assignment on the original dataset, exactly matching the
    output of `evaluate_final_coverage()`.
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
        """
        Args:
            target: Name of the binary target column (1 = Event, 0 = Non-Event).
            n_jobs: Number of parallel jobs for IV computation. -1 uses all but one core.
            min_sample_size: Absolute minimum row count for a valid rule (fallback).
            min_lift: Minimum lift threshold (fallback).
            min_events: Minimum number of positive events for a valid rule (fallback).
            top_n_vars: Number of highest‑IV features passed into the Apriori engine.
            max_segments: Maximum number of segments to extract.
            max_feature_reuse: Max times a feature can appear across segments.
            param_grid: Optional grid of {min_sample_size, min_lift} to sweep.
            enable_diversity: If True, blocks rules combining variables from same group.
            enable_1way: Allow 1‑dimensional rules.
            enable_2way: Allow 2‑dimensional intersection rules.
            enable_3way: Allow 3‑dimensional intersection rules.
            feature_groups: Mapping of business categories to columns (e.g. {'risk': ['scr', 'bal']}).
            ignore_features: Explicit list of columns to drop prior to IV calculation.
        """
        self.target = target
        self.n_jobs = n_jobs if n_jobs != -1 else max(1, os.cpu_count() - 1)
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

        # Diagnostic repository (feature journey tracking)
        self.diagnostics_: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_optb_dtype(duckdb_type: str) -> str:
        """
        Determines the correct OptBinning data type flag from a DuckDB type string.

        Args:
            duckdb_type: DuckDB column type as string.

        Returns:
            'categorical' or 'numerical'.
        """
        dtype_upper = duckdb_type.upper()
        if any(t in dtype_upper for t in ["VARCHAR", "CHAR", "STRING", "TEXT", "UUID"]):
            return "categorical"
        return "numerical"

    def _validate_feature_groups(self, columns: List[str]) -> None:
        """
        Validates that all declared feature group variables exist in the target dataset.

        Args:
            columns: List of all column names from the table.

        Raises:
            ValueError: If any feature in a group is not found.
        """
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

        logger.info(f"✅ Feature group validation passed. ({validated_count} features mapped)")

    def get_group(self, var: str) -> str:
        """
        Returns the assigned business category for a feature, or the feature name itself.

        Args:
            var: Feature name.

        Returns:
            Group name or the feature name.
        """
        for group, vars_list in self.feature_groups.items():
            if var in vars_list:
                return group
        return var

    def is_diverse(self, combo: Tuple[str, ...]) -> bool:
        """
        Ensures a tuple of features spans strictly distinct analytical groups.

        Args:
            combo: Tuple of feature names.

        Returns:
            True if all features belong to different groups (or diversity is disabled).
        """
        if not self.enable_diversity:
            return True
        groups = [self.get_group(v) for v in combo]
        return len(groups) == len(set(groups))

    def compute_iv_ranking_and_bin(
        self,
        con: duckdb.DuckDBPyConnection,
        eligible_cols: List[str],
        columns_types: Dict[str, str],
    ) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
        """
        Computes Information Value (IV) and pre‑computed bins in a single parallel pass.

        Args:
            con: DuckDB connection.
            eligible_cols: List of feature columns to evaluate.
            columns_types: Mapping of column name to its DuckDB type.

        Returns:
            - Ranking: List of dicts with 'variable' and 'iv' (IV * 100).
            - Precomputed bins: Dict mapping feature name to transformed bin array.
        """
        logger.info(f"🔍 Computing IV and bins for {len(eligible_cols)} features...")

        def _worker(col: str) -> Tuple[str, float, Optional[np.ndarray]]:
            try:
                thread_con = con.cursor()
                data_dict = thread_con.execute(
                    f'SELECT "{col}", "{self.target}" FROM current_df'
                ).fetchnumpy()
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
        """
        Batches SQL GROUP BY queries for a list of feature combinations.

        Args:
            con: DuckDB connection.
            combo_list: List of tuples, each tuple is a combination of features.
            base_rate: Global event rate (used to compute lift).

        Returns:
            List of rule dictionaries with keys: rule, count, events, rate, lift, combo_vars.
        """
        if not combo_list:
            return []

        queries = []
        for combo in combo_list:
            cols_str = ", ".join([f'"{c}"' for c in combo])
            rule_concat = " || ' & ' || ".join(
                [f"'{c}=' || CAST(\"{c}\" AS VARCHAR)" for c in combo]
            )
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
        chunk_size = 100

        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i + chunk_size]
            union_query = " UNION ALL ".join(chunk)

            res = con.execute(union_query).fetchall()
            for rule, count, events, combo_vars_str in res:
                rate = (events / count) * 100.0 if count > 0 else 0
                lift = rate / (base_rate * 100.0) if base_rate > 0 else 0

                if lift >= self.min_lift and events >= self.min_events and lift > 1.0:
                    valid_results.append(
                        {
                            "rule": rule,
                            "count": count,
                            "rate": rate,
                            "lift": lift,
                            "events": events,
                            "combo_vars": tuple(combo_vars_str.split(",")),
                        }
                    )

        return valid_results

    def parse_rule_to_sql(self, rule_str: str) -> str:
        """
        Translates OptBinning string syntax into a production SQL WHERE clause.

        Args:
            rule_str: Rule string like "col1='A' & col2=[1,2,3]".

        Returns:
            SQL condition string (e.g., `col1 IN ('A') AND col2 IN (1,2,3)`).
        """
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

            # --- Categorical set handling ---
            if is_categorical and bracket_match:
                import ast

                try:
                    raw_items = ast.literal_eval(bracket_match.group(0))
                except Exception:
                    raw_content = bracket_match.group(1)
                    if "," not in raw_content:
                        raw_content = re.sub(r"'\s+'", "','", raw_content)
                        raw_content = re.sub(r'"\s+"', '","', raw_content)
                        raw_content = re.sub(r"\s+", ",", raw_content)

                    raw_items = [
                        i.strip().strip("'").strip('"')
                        for i in raw_content.split(",")
                        if i.strip()
                    ]

                formatted_items = ", ".join(
                    [f"'{item}'" if isinstance(item, str) else str(item) for item in raw_items]
                )

                if formatted_items:
                    sql_conditions.append(f"{col} IN ({formatted_items})")
                continue

            # --- Special / Missing ---
            if interval in ["Special", "Missing"]:
                sql_conditions.append(f"{col} IS NULL")
                continue

            # --- Continuous range ---
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

        return " AND ".join(f"({cond})" if "AND" in cond else cond for cond in sql_conditions)

    def extract_segments(self, data: Any) -> List[Dict[str, Any]]:
        """
        Sequentially extracts high‑lift rules on the residual dataset.

        After extraction, the stored counts reflect the final hierarchical segmentation
        on the original dataset, ensuring consistency with `evaluate_final_coverage()`.

        Args:
            data: Input data (will be loaded into DuckDB).

        Returns:
            List of segment dictionaries with keys: segment_id, rule_string, sql_filter,
            count, rate, lift, meta_applied_sample_size, meta_applied_min_lift.
        """
        logger.info("🚀 Starting hierarchical segment extraction...")

        # Cache absolute hard constraints
        abs_min_sample_size = self.min_sample_size
        abs_min_events = self.min_events

        # Use file‑backed storage for performance
        if os.path.exists(f"experiment_{timestamp}.db"):
            os.remove(f"experiment_{timestamp}.db")
        con = duckdb.connect(f"experiment_{timestamp}.db")

        total_cores = os.cpu_count() or 1
        target_threads = max(1, total_cores - 2) if total_cores > 4 else total_cores
        total_memory_bytes = psutil.virtual_memory().total
        target_memory_gb = int((total_memory_bytes * 0.5) / (1024 ** 3))

        con.execute(f"SET threads = {target_threads};")
        con.execute(f"SET memory_limit = '{target_memory_gb}GB';")
        logger.info(
            f"⚙️ DuckDB Configured: Threads={target_threads}/{total_cores}, "
            f"MemoryLimit={target_memory_gb}GB"
        )

        con.execute("CREATE OR REPLACE TABLE current_df AS SELECT * FROM data")

        cols_info = con.execute("DESCRIBE current_df").fetchall()
        columns_types = {row[0]: row[1] for row in cols_info}
        all_cols = list(columns_types.keys())

        if self.enable_diversity:
            self._validate_feature_groups(all_cols)

        eligible_cols = [
            c for c in all_cols if c != self.target and c not in self.ignore_features
        ]
        self.feature_usage_counts = {col: 0 for col in eligible_cols}

        # Build parameter grid experiments
        if self.param_grid:
            sizes = self.param_grid.get("min_sample_size", [self.min_sample_size])
            lifts = self.param_grid.get("min_lift", [self.min_lift])
            experiments = [
                {"min_sample_size": s, "min_lift": l}
                for s, l in product(sizes, lifts)
            ]
            logger.info(
                f"📊 Dynamic Grid Search Enabled: {len(experiments)} configurations."
            )
        else:
            experiments = [{"min_sample_size": self.min_sample_size, "min_lift": self.min_lift}]

        for i in range(1, self.max_segments + 1):
            res = con.execute(
                f'SELECT AVG("{self.target}"), COUNT(*) FROM current_df'
            ).fetchone()
            base_rate, current_volume = res[0] or 0.0, res[1] or 0

            min_floor_volume = min(exp["min_sample_size"] for exp in experiments)

            if base_rate == 0 or current_volume < min_floor_volume:
                logger.info(
                    f"⏹️ Stopping: base_rate={base_rate}, volume={current_volume} < "
                    f"min_floor={min_floor_volume}"
                )
                break

            logger.info(
                f"🔄 Iteration {i} | Remaining Volume: {current_volume:,} | "
                f"Base Rate: {base_rate*100:.2f}%"
            )

            iv_ranking, precomputed_bins = self.compute_iv_ranking_and_bin(
                con, eligible_cols, columns_types
            )

            # --- Diagnostic snapshot ---
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
                    "status": status,
                }
            self.diagnostics_.append(
                {
                    "iteration": i,
                    "residual_volume": current_volume,
                    "base_rate": base_rate,
                    "features_state": iteration_snapshot,
                    "winning_segment": None,
                }
            )

            allowed_vars = [
                row["variable"]
                for row in iv_ranking
                if self.feature_usage_counts.get(row["variable"], 0) < self.max_feature_reuse
            ]
            top_vars = allowed_vars[:self.top_n_vars]
            if not top_vars:
                logger.warning("⚠️ All eligible features exhausted. Aborting.")
                break

            # Build binned table
            binned_data = {
                self.target: con.execute(
                    f'SELECT "{self.target}" FROM current_df'
                ).fetchnumpy()[self.target]
            }
            valid_vars = []
            for v in top_vars:
                if v in precomputed_bins and len(np.unique(precomputed_bins[v])) > 1:
                    binned_data[v] = precomputed_bins[v]
                    valid_vars.append(v)
            if not valid_vars:
                logger.warning("⚠️ No valid binned variables found. Stopping.")
                break

            con.execute("DROP TABLE IF EXISTS binned_df")
            con.execute("CREATE TABLE binned_df AS SELECT * FROM binned_data")

            # --- Grid search over parameter configurations ---
            grid_candidates: List[Dict[str, Any]] = []
            for config in experiments:
                self.min_sample_size = config["min_sample_size"]
                self.min_lift = config["min_lift"]

                all_rules: List[Dict[str, Any]] = []

                # Level 1 (Singles)
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

                # Level 2 (Pairs)
                valid_2way_sets = set()
                if len(valid_1way_vars) >= 2 and (self.enable_2way or self.enable_3way):
                    combos_2 = [
                        c
                        for c in combinations(valid_1way_vars, 2)
                        if self.is_diverse(c)
                    ]
                    if combos_2:
                        res_2 = self._agg_combinations(con, combos_2, base_rate)
                        if res_2:
                            valid_2way_sets = {
                                frozenset(c["combo_vars"]) for c in res_2
                            }
                            if self.enable_2way:
                                all_rules.extend(res_2)

                # Level 3 (Triplets)
                if (
                    self.enable_3way
                    and len(valid_1way_vars) >= 3
                    and valid_2way_sets
                ):
                    combos_3 = [
                        c
                        for c in combinations(valid_1way_vars, 3)
                        if self.is_diverse(c)
                        and all(
                            frozenset(p) in valid_2way_sets
                            for p in combinations(c, 2)
                        )
                    ]
                    if combos_3:
                        res_3 = self._agg_combinations(con, combos_3, base_rate)
                        if res_3:
                            all_rules.extend(res_3)

                if all_rules:
                    # Sort by lift first, then count, then rate (prioritise lift)
                    all_rules.sort(
                        key=lambda x: (x["lift"], x["count"], x["rate"]),
                        reverse=True,
                    )
                    top_match = all_rules[0].copy()
                    top_match["grid_min_sample_size"] = config["min_sample_size"]
                    top_match["grid_min_lift"] = config["min_lift"]
                    grid_candidates.append(top_match)

            if not grid_candidates:
                logger.info("⏹️ No candidates cleared the grid. Stopping.")
                break

            # Rank candidates by (lift, count, rate)
            grid_candidates.sort(
                key=lambda x: (x["lift"], x["count"], x["rate"]), reverse=True
            )

            selected_candidate = None
            for candidate in grid_candidates:
                rule_str = candidate["rule"]
                sql_filter = self.parse_rule_to_sql(rule_str)
                # Validate on raw current_df (residual)
                actual = con.execute(
                    f'SELECT COUNT(*) AS cnt, SUM(CAST("{self.target}" AS DOUBLE)) AS evt '
                    f'FROM current_df WHERE ({sql_filter})'
                ).fetchone()
                actual_cnt, actual_evt = actual[0], actual[1] or 0

                if actual_cnt >= abs_min_sample_size and actual_evt >= abs_min_events:
                    selected_candidate = {
                        **candidate,
                        "sql_filter": sql_filter,
                        "actual_count": actual_cnt,
                        "actual_events": actual_evt,
                    }
                    break
                else:
                    logger.debug(
                        f"Candidate rejected by raw validation: {rule_str} -> "
                        f"rows={actual_cnt}, events={actual_evt}"
                    )

            if selected_candidate is None:
                logger.warning(
                    f"⚠️ Iteration {i}: No candidate passed hard constraints. Stopping."
                )
                break

            best_rule = selected_candidate["rule"]
            best_raw_sql = selected_candidate["sql_filter"]
            winning_combo = selected_candidate["combo_vars"]

            # Compute metrics from the residual counts
            actual_rate = (
                selected_candidate["actual_events"] / selected_candidate["actual_count"]
            ) * 100.0
            actual_lift = (
                actual_rate / (base_rate * 100.0) if base_rate > 0 else 0.0
            )

            for var in winning_combo:
                self.feature_usage_counts[var] = (
                    self.feature_usage_counts.get(var, 0) + 1
                )
                logger.info(
                    f"📌 Feature Usage Tracker Update -> '{var}' used count = "
                    f"{self.feature_usage_counts[var]}"
                )

            # Store the raw rule and raw SQL (no exclusions)
            self.segments.append(
                {
                    "segment_id": i,
                    "rule_string": best_rule,
                    "sql_filter": best_raw_sql,
                    "count": int(selected_candidate["actual_count"]),
                    "rate": float(actual_rate),
                    "lift": float(actual_lift),
                    "meta_applied_sample_size": int(
                        selected_candidate["grid_min_sample_size"]
                    ),
                    "meta_applied_min_lift": float(
                        selected_candidate["grid_min_lift"]
                    ),
                }
            )

            logger.info(
                f"✅ Segment {i} Captured (Size Floor: "
                f"{selected_candidate['grid_min_sample_size']} | "
                f"Lift Floor: {selected_candidate['grid_min_lift']}): "
                f"rows={selected_candidate['actual_count']}, "
                f"events={selected_candidate['actual_events']}, "
                f"lift={actual_lift:.2f}\n"
                f"  Rule: {best_rule}\n"
                f"  SQL: {best_raw_sql}"
            )

            self.diagnostics_[-1]["winning_segment"] = {
                "rule": best_rule,
                "sql_filter": best_raw_sql,
                "variables_used": list(winning_combo),
                "lift": actual_lift,
                "count": int(selected_candidate["actual_count"]),
            }

            # Remove rows matching the raw rule from the residual
            # NULL‑safe: keep rows where the rule evaluates to NULL
            con.execute(
                f"""
                CREATE TABLE temp_residual AS
                SELECT * FROM current_df
                WHERE NOT ({best_raw_sql}) OR ({best_raw_sql}) IS NULL
            """
            )
            con.execute("DROP TABLE current_df")
            con.execute("ALTER TABLE temp_residual RENAME TO current_df")

        # Restore original config
        self.min_sample_size = abs_min_sample_size
        self.min_events = abs_min_events

        con.close()
        logger.info("🏁 Extraction complete.")
        return self.segments

    def evaluate_final_coverage(self, original_data: Any) -> List[Dict[str, Any]]:
        """
        Evaluates the hierarchical segmentation on the original dataset.

        The rules are applied in the order they were extracted (first rule gets
        highest priority). This yields the true hierarchical segmentation.

        Args:
            original_data: The original (unfiltered) dataset.

        Returns:
            List of dictionaries with segment KPIs: segment, total_count,
            target_events, response_rate, base_response_rate, capture_rate, lift.
        """
        if not self.segments:
            return []

        logger.info("📊 Evaluating final hierarchical coverage on original data...")
        con = duckdb.connect(f"experiment_{timestamp}.db")
        con.execute("CREATE OR REPLACE TABLE original_df AS SELECT * FROM original_data")

        # Build CASE statement with raw SQL filters in order
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

        res = con.execute(final_query)
        columns = [desc[0] for desc in res.description]
        res_list = [dict(zip(columns, row)) for row in res.fetchall()]
        con.close()
        return res_list

    def explain_feature_journey(self, feature_name: str) -> None:
        """
        Prints a detailed audit trail of a specific feature across all iterations.

        Args:
            feature_name: Name of the feature to trace.
        """
        if not self.diagnostics_:
            print("⚠️ No diagnostic records found. Run extract_segments() first.")
            return

        print("=" * 80)
        print(f"📌 AUDIT TRAIL FOR FEATURE: '{feature_name}'")
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
                print(
                    f"  • Winner this round    : {winner['rule']} "
                    f"(Variables: {winner['variables_used']})"
                )
        print("=" * 80)