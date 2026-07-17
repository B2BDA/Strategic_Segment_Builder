"""
BigQuery Feature Selector
=========================
High‑performance Information Value (IV) and standard deviation screening
natively inside Google BigQuery. Designed for large‑scale enterprise pipelines.

WARNING: This utility may incur significant BigQuery costs. Use with caution.
"""

import logging
from typing import List, Optional, Tuple

import duckdb
from google.cloud import bigquery

# -----------------------------------------------------------------------------
# Module-level configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BigQueryFeatureSelector:
    """
    Highly scalable feature selection utility for enterprise analytics pipelines.

    Optimizations:
        - Single‑hit schema extraction to reduce API latency.
        - Dynamic binary profiling: automatically detects numeric 0/1 flags and
          treats them as categorical.
        - Batched SQL execution to safeguard BigQuery slot availability.
        - DuckDB integration: streams data via PyArrow to bypass Pandas entirely.

    Args:
        project_id: Google Cloud project ID.
        dataset_id: BigQuery dataset ID.
        table_id: BigQuery table ID.
        target_column: Name of the binary target variable (must be 0 or 1).
        iv_threshold: Minimum Information Value required to retain a feature.
        stddev_threshold: Minimum standard deviation required to retain a feature.
        bins: Number of quantiles (NTILEs) for the naive IV calculation.
        batch_size: Number of columns to process per BigQuery job to prevent timeouts.
        binary_columns: Explicit list of binary columns. If None, auto‑detected.
        bq_client: Optional pre‑configured BigQuery client.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        target_column: str,
        iv_threshold: float = 0.02,
        stddev_threshold: float = 1e-5,
        min_bin_n_event: int = 1,
        min_bin_n_nonevent: int = 1,
        bins: int = 10,
        batch_size: int = 15,
        binary_columns: Optional[List[str]] = None,
        bq_client: Optional[bigquery.Client] = None,
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.target_column = target_column
        self.iv_threshold = iv_threshold
        self.stddev_threshold = stddev_threshold
        self.min_bin_n_event = min_bin_n_event
        self.min_bin_n_nonevent = min_bin_n_nonevent
        self.bins = bins
        self.batch_size = batch_size
        self.binary_columns = binary_columns

        self.client = bq_client if bq_client else bigquery.Client(project=self.project_id)
        self.full_table_path = f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"

    def _detect_binary_numerical_columns(self, numerical_columns: List[str]) -> List[str]:
        """
        Executes a single‑row query to count distinct values in numerical columns.

        Args:
            numerical_columns: List of numerical column names.

        Returns:
            List of columns where the distinct count is <= 2 (binary flags).
        """
        if not numerical_columns:
            return []

        logger.info("🔍 Performing dynamic profiling on numerical columns to detect binary flags...")

        select_expressions = [
            f"COUNT(DISTINCT {col}) AS `{col}`" for col in numerical_columns
        ]
        query = f"SELECT {', '.join(select_expressions)} FROM {self.full_table_path}"

        try:
            job = self.client.query(query)
            row = next(job.result())

            detected_binary = [col for col in numerical_columns if row[col] <= 2]
            if detected_binary:
                logger.info(
                    f"✅ Dynamically detected {len(detected_binary)} binary numeric columns: "
                    f"{detected_binary}"
                )
            else:
                logger.info("ℹ️ No implicit binary numeric columns discovered.")
            return detected_binary

        except Exception as e:
            logger.error(f"❌ Failed to dynamically profile numeric columns: {e}. Proceeding without auto‑detection.")
            return []

    def _get_table_schema(self) -> Tuple[List[str], List[str]]:
        """
        Retrieves the table schema and categorises columns into continuous and categorical.

        Returns:
            Tuple (numerical_columns, categorical_columns).
        """
        logger.info(f"📋 Fetching metadata schema for table: {self.full_table_path}")
        table_ref = self.client.get_table(
            f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        )

        num_types = {"INTEGER", "FLOAT", "NUMERIC", "BIGNUMERIC"}
        cat_types = {"STRING", "BOOLEAN"}

        raw_numerical = []
        categorical_columns = []

        for field in table_ref.schema:
            if field.name == self.target_column:
                continue

            if field.field_type in num_types:
                raw_numerical.append(field.name)
            elif field.field_type in cat_types:
                categorical_columns.append(field.name)

        # Handle binary overrides / auto-detection
        if self.binary_columns is not None:
            logger.info("📌 Applying user‑defined binary column overrides.")
            actual_binary = [col for col in self.binary_columns if col in raw_numerical]
        else:
            actual_binary = self._detect_binary_numerical_columns(raw_numerical)

        numerical_columns = [col for col in raw_numerical if col not in actual_binary]
        categorical_columns.extend(actual_binary)

        logger.info(
            f"✅ Final categorization: {len(numerical_columns)} continuous features, "
            f"{len(categorical_columns)} categorical/binary features."
        )
        return numerical_columns, categorical_columns

    def _build_batch_query(
        self, numerical_chunk: List[str], categorical_chunk: List[str]
    ) -> str:
        """
        Constructs the optimised SQL query for a specific batch of columns.

        Args:
            numerical_chunk: List of numerical column names.
            categorical_chunk: List of categorical/binary column names.

        Returns:
            A single SQL string with a UNION ALL of per‑column IV calculations.
        """
        sql_parts = [f"""
        WITH global_stats AS (
            SELECT
                COUNTIF({self.target_column} = 0) AS total_goods,
                COUNTIF({self.target_column} = 1) AS total_bads
            FROM {self.full_table_path}
        )
        """]

        union_queries = []

        # IV calculation template with minimum bin protection
        iv_calculation_template = f"""
        SUM(
            CASE
                WHEN goods_in_bin < {self.min_bin_n_nonevent}
                  OR bads_in_bin < {self.min_bin_n_event}
                THEN 0
                ELSE
                    ((goods_in_bin / NULLIF(global_stats.total_goods, 0))
                     - (bads_in_bin / NULLIF(global_stats.total_bads, 0)))
                    * LN(
                        ((goods_in_bin / NULLIF(global_stats.total_goods, 0)) + 0.0001)
                        / ((bads_in_bin / NULLIF(global_stats.total_bads, 0)) + 0.0001)
                      )
            END
        ) AS naive_iv
        """

        # ---- Numerical columns ----
        for col in numerical_chunk:
            part = f"""
            SELECT
                '{col}' AS feature_name,
                (SELECT STDDEV({col}) FROM {self.full_table_path}) AS feature_stddev,
                {iv_calculation_template}
            FROM
            (
                SELECT
                    bin,
                    COUNTIF({self.target_column} = 0) AS goods_in_bin,
                    COUNTIF({self.target_column} = 1) AS bads_in_bin
                FROM
                (
                    SELECT
                        NTILE({self.bins}) OVER (ORDER BY {col}) AS bin,
                        {self.target_column}
                    FROM {self.full_table_path}
                    WHERE {col} IS NOT NULL
                )
                GROUP BY bin
            )
            CROSS JOIN global_stats
            """
            union_queries.append(part)

        # ---- Categorical/binary columns ----
        for col in categorical_chunk:
            part = f"""
            SELECT
                '{col}' AS feature_name,
                9999.0 AS feature_stddev,
                {iv_calculation_template}
            FROM
            (
                SELECT
                    CAST({col} AS STRING) AS bin,
                    COUNTIF({self.target_column} = 0) AS goods_in_bin,
                    COUNTIF({self.target_column} = 1) AS bads_in_bin
                FROM {self.full_table_path}
                WHERE {col} IS NOT NULL
                GROUP BY 1
            )
            CROSS JOIN global_stats
            """
            union_queries.append(part)

        sql_parts.append("\nUNION ALL\n".join(union_queries))
        return "\n".join(sql_parts)

    def screen_features(self) -> duckdb.DuckDBPyRelation:
        """
        Executes the pipeline and filters/sorts results via local DuckDB.

        Returns:
            A DuckDB relation (table) with columns: feature_name, feature_stddev, naive_iv.
        """
        numerical_columns, categorical_columns = self._get_table_schema()

        con = duckdb.connect()

        if not numerical_columns and not categorical_columns:
            logger.warning("⚠️ No valid numerical or categorical columns found to screen.")
            return con.sql(
                "SELECT NULL AS feature_name, NULL AS feature_stddev, NULL AS naive_iv WHERE 1=0"
            )

        con.execute("""
            CREATE TABLE all_results (
                feature_name VARCHAR,
                feature_stddev DOUBLE,
                naive_iv DOUBLE
            )
        """)

        # Process numerical columns in batches
        for i in range(0, len(numerical_columns), self.batch_size):
            chunk = numerical_columns[i:i + self.batch_size]
            logger.info(f"📊 Processing numerical batch {i // self.batch_size + 1}...")
            query = self._build_batch_query(numerical_chunk=chunk, categorical_chunk=[])
            arrow_table = self.client.query(query).to_arrow()
            con.execute("INSERT INTO all_results SELECT * FROM arrow_table")

        # Process categorical/binary columns in batches
        for i in range(0, len(categorical_columns), self.batch_size):
            chunk = categorical_columns[i:i + self.batch_size]
            logger.info(f"📊 Processing categorical/binary batch {i // self.batch_size + 1}...")
            query = self._build_batch_query(numerical_chunk=[], categorical_chunk=chunk)
            arrow_table = self.client.query(query).to_arrow()
            con.execute("INSERT INTO all_results SELECT * FROM arrow_table")

        # Apply thresholds and sort inside DuckDB
        retained_features_rel = con.sql(f"""
            SELECT
                feature_name,
                feature_stddev,
                naive_iv
            FROM all_results
            WHERE feature_stddev > {self.stddev_threshold}
              AND naive_iv >= {self.iv_threshold}
            ORDER BY naive_iv DESC
        """)

        retained_count = con.sql("SELECT COUNT(*) FROM retained_features_rel").fetchone()[0]
        total_screened = len(numerical_columns) + len(categorical_columns)
        dropped_count = total_screened - retained_count

        logger.info(
            f"✅ Screening complete. Retained: {retained_count} | Dropped: {dropped_count}"
        )

        return retained_features_rel