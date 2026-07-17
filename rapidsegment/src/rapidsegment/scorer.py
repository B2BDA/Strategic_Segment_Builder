"""
Strategic Segmentation Scorecard Engine
=======================================
Creating scorecards for multi‑segmented populations using vectorised DuckDB aggregations.

Author: Bishwarup Biswas + Gemini + DeepSeek
Python Version: 3.9+
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import duckdb

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


class StrategicSegmentScore:
    """
    High‑Throughput Vectorised Scorecard Engine.

    Computes segment weights via Harmonic Mean and applies deciling
    over large datasets natively inside DuckDB's out‑of‑core engine.

    Args:
        target_col: Name of the binary target column (0/1).
        primary_key: Unique identifier column.
        segment_cols: List of binary (0/1) segment flag columns.
    """

    def __init__(
        self, target_col: str, primary_key: str, segment_cols: List[str]
    ) -> None:
        self.target_col = target_col
        self.primary_key = primary_key
        self.segment_cols = segment_cols
        self.model_artifact: Dict[str, Any] = {}

    def calculate_and_export_weights(
        self,
        data: Any,
        export_path: str = f"scored_experiment_{timestamp}.json",
    ) -> Dict[str, Any]:
        """
        Calculates harmonic weights and derives decile boundaries via vectorised execution.

        Args:
            data: Input data (will be loaded into DuckDB).
            export_path: File path to save the model artifact JSON.

        Returns:
            Dictionary containing model metadata, segment weights, and decile thresholds.
        """
        logger.info("🚀 Initialising out‑of‑core DuckDB scorecard engine...")

        # Use file‑backed storage for large datasets
        if os.path.exists(f"score_experiment_{timestamp}.db"):
            os.remove(f"score_experiment_{timestamp}.db")
        ctx = duckdb.connect(f"score_experiment_{timestamp}.db")
        ctx.execute("CREATE OR REPLACE TABLE df AS SELECT * FROM data")

        # ---------------------------------------------------------------------
        # Step 1: Baseline metrics + vectorised multi‑segment aggregation
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Step 2: Unpack aggregated results into weight lookup
        # ---------------------------------------------------------------------
        logger.info("📊 Computing harmonic scorecard weights...")
        weights_lookup: Dict[str, Dict[str, Union[int, float]]] = {}

        for idx, seg_col in enumerate(self.segment_cols):
            seg_count = master_res[2 + (idx * 2)] or 0
            seg_events = master_res[2 + (idx * 2) + 1] or 0

            if seg_count == 0 or seg_events == 0:
                logger.warning(
                    f"⚠️ Segment '{seg_col}' has zero volume or events. Setting weight=0."
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
                "weight": int(round(raw_weight)),
                "lift": round(lift, 4),
                "response_rate": round(response_rate, 4),
                "capture_rate": round(capture_rate, 4),
            }

        # ---------------------------------------------------------------------
        # Step 3: Score the entire population using C++ SQL math
        # ---------------------------------------------------------------------
        logger.info("⚡ Scoring population natively via SQL engine...")
        scored_cols = list(weights_lookup.keys())
        if not scored_cols:
            raise ValueError("Scorecard Failure: No valid segments found to score.")

        score_terms = [
            f'(CAST("{col}" AS DOUBLE) * {weights_lookup[col]["weight"]})'
            for col in scored_cols
        ]
        score_math_expr = " + ".join(score_terms)

        ctx.execute(f"""
            CREATE OR REPLACE TABLE scored_population AS
            SELECT "{self.primary_key}", ({score_math_expr}) AS total_score
            FROM df
        """)

        logger.info(f"📉 Dataset Zero‑Inflation Rate: {zero_inflation_rate:.2%}")

        # ---------------------------------------------------------------------
        # Step 4: Out‑of‑Core Decile Boundary Profiling (via SQL quantiles)
        # ---------------------------------------------------------------------
        logger.info("📈 Calibrating deciles across active populations...")

        # Handle zero‑inflation masking entirely on the database side
        filter_clause = "WHERE total_score > 0" if zero_inflation_rate >= 0.80 else ""

        # Compute all 10 deciles in a single table scan using quantile_disc
        quantile_query = f"""
            SELECT QUANTILE_DISC(total_score, [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            ])
            FROM scored_population
            {filter_clause}
        """
        quantiles_res = ctx.execute(quantile_query).fetchone()

        if not quantiles_res or not quantiles_res[0]:
            raise ValueError(
                "Scorecard Failure: 0 customers triggered any segment rules."
            )

        quantiles = quantiles_res[0]
        quantiles = quantiles[::-1]  # Reverse to descending order
        decile_thresholds = {str(i + 1): int(quantiles[i]) for i in range(10)}

        # Active population size (only those with positive scores or all, depending on filter)
        active_pop_size = ctx.execute(
            f"SELECT COUNT(*) FROM scored_population {filter_clause}"
        ).fetchone()[0]

        # ---------------------------------------------------------------------
        # Step 5: Build and export model artifact
        # ---------------------------------------------------------------------
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

        logger.info(f"✅ Scorecard exported to: {export_path}")
        ctx.close()
        return self.model_artifact