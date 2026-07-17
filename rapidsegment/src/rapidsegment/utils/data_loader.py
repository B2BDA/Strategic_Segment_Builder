"""
Unified Data Ingestion Layer
============================
Multi‑format data loader supporting Local Files (CSV, Parquet, Arrow, Excel),
In‑Memory PyArrow Tables, and Google Cloud BigQuery Storage API streams.

Author: Bishwarup Biswas + Gemini + DeepSeek
Python Version: 3.9+
"""

import logging
import os
from typing import Any, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_pq

logger = logging.getLogger("StrategicEngine.DataLoader")


class UniversalDataLoader:
    """
    Handles multi‑source data ingestion, normalising inputs into highly optimised
    in‑memory PyArrow Tables suitable for vectorised downstream compute engines.

    The loader automatically detects the source type based on constructor arguments.
    If a `fallback_data` object is passed to `load()`, it takes precedence.

    Args:
        project_id: (Optional) GCP project ID for BigQuery.
        dataset_id: (Optional) BigQuery dataset ID.
        table_id: (Optional) BigQuery table ID.
        file_path: (Optional) Local file path (CSV, Parquet, Arrow/Feather, Excel).
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.file_path = file_path

    def load(self, fallback_data: Optional[Any] = None) -> Union[pa.Table, str]:
        """
        Auto‑detects the source configuration and loads the dataset.

        Priority order:
            1. If `fallback_data` is provided, it is returned (with type normalisation).
            2. If BigQuery identifiers are set, load from BigQuery.
            3. If a local `file_path` is provided, load from file.

        Args:
            fallback_data: Optional pre‑loaded data (e.g., a PyArrow Table or any
                           object that can be passed to DuckDB directly).

        Returns:
            A PyArrow Table, or a DuckDB scan macro string (when BigQuery client
            is not available and fallback is not provided).
        """
        # Scenario 1: Direct in‑memory object
        if fallback_data is not None:
            if isinstance(fallback_data, pa.Table):
                logger.info("📥 Ingesting directly provided in‑memory PyArrow Table.")
                return self._cast_table_numerics_to_float(fallback_data)
            logger.info("📥 Using provided fallback data (non‑Arrow) as‑is.")
            return fallback_data

        # Scenario 2: BigQuery
        if self.dataset_id and self.table_id:
            return self._load_from_bigquery()

        # Scenario 3: Local file
        if self.file_path:
            return self._load_from_file()

        raise ValueError(
            "Invalid Configuration: You must provide either a valid `file_path`, "
            "BigQuery identifiers, or pass an explicit `fallback_data` object."
        )

    @staticmethod
    def _cast_table_numerics_to_float(table: pa.Table) -> pa.Table:
        """
        Casts all numeric columns in a PyArrow Table to float64.

        This ensures consistent numerical precision across downstream operations.

        Args:
            table: Input PyArrow Table.

        Returns:
            A new table with numeric columns cast to float64.
        """
        if not isinstance(table, pa.Table):
            return table

        new_columns = []
        new_fields = []

        for i, field in enumerate(table.schema):
            # Check if the type is integer, floating, or decimal
            if (
                pa.types.is_integer(field.type)
                or pa.types.is_floating(field.type)
                or pa.types.is_decimal(field.type)
            ):
                try:
                    casted_col = pc.cast(
                        table.column(i), pa.float64(), safe=False
                    )
                    new_columns.append(casted_col)
                    new_fields.append(
                        pa.field(field.name, pa.float64(), nullable=field.nullable)
                    )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to cast column {field.name} to float64. Reason: {e}"
                    )
                    new_columns.append(table.column(i))
                    new_fields.append(field)
            else:
                new_columns.append(table.column(i))
                new_fields.append(field)

        return pa.Table.from_arrays(new_columns, schema=pa.schema(new_fields))

    def _load_from_file(self) -> pa.Table:
        """
        Parses a local file using high‑performance C++ Arrow readers.

        Supports: .parquet, .csv, .arrow / .feather, .xlsx / .xls.

        Returns:
            PyArrow Table with numeric columns cast to float64.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at: {self.file_path}")

        ext = os.path.splitext(self.file_path)[-1].lower()
        logger.info(f"📂 Loading file: {self.file_path} (extension: {ext})")

        if ext == ".parquet":
            table = pa_pq.read_table(self.file_path)
        elif ext == ".csv":
            table = pa_csv.read_csv(self.file_path)
        elif ext in (".arrow", ".feather"):
            with pa.memory_map(self.file_path, "r") as source:
                table = pa.ipc.open_file(source).read_all()
        elif ext in (".xlsx", ".xls"):
            table = self._load_excel_to_arrow()
        else:
            raise ValueError(f"Unsupported file format: '{ext}'.")

        return self._cast_table_numerics_to_float(table)

    def _load_excel_to_arrow(self) -> pa.Table:
        """
        Parses an Excel file using openpyxl with positional column tracking.

        Returns:
            PyArrow Table.
        """
        logger.info("📊 Parsing Excel spreadsheet via positional column tracking...")
        try:
            import openpyxl

            wb = openpyxl.load_workbook(
                self.file_path, data_only=True, read_only=True
            )
            sheet = wb.active
            rows = sheet.iter_rows(values_only=True)

            headers = next(rows)
            if not headers:
                raise ValueError("The Excel file appears to be empty.")

            # Use column indices to prevent header‑shift corruption
            column_names = [
                f"{h}" if h is not None else f"_col_{i}"
                for i, h in enumerate(headers)
            ]
            data_columns = {name: [] for name in column_names}

            for row in rows:
                for i, name in enumerate(column_names):
                    val = row[i] if i < len(row) else None
                    data_columns[name].append(val)

            wb.close()
            return pa.Table.from_pydict(data_columns)

        except ImportError:
            raise ImportError(
                "Dependency missing: `pip install openpyxl` required for Excel files."
            )

    def _load_from_bigquery(self) -> Union[pa.Table, str]:
        """
        Resolves BigQuery ingestion using cost‑optimised metadata inspection.

        If the `google‑cloud‑bigquery` library is available, streams the table
        as a PyArrow Table. Otherwise, returns a DuckDB scan macro string for
        later execution (requires DuckDB's BigQuery extension).

        Returns:
            PyArrow Table or a DuckDB macro string.
        """
        full_bq_path = (
            f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            if self.project_id
            else f"{self.dataset_id}.{self.table_id}"
        )
        logger.info(f"☁️ Initialising BigQuery client for: {full_bq_path}")

        try:
            from google.cloud import bigquery

            bq_client = bigquery.Client(project=self.project_id)
            full_table_ref = (
                f"{self.project_id or bq_client.project}."
                f"{self.dataset_id}.{self.table_id}"
            )

            # Fetch schema via get_table (cheaper than INFORMATION_SCHEMA)
            table = bq_client.get_table(full_table_ref)

            select_clauses = []
            for field in table.schema:
                # Cast numeric types to FLOAT64 for consistency
                if field.field_type in (
                    "NUMERIC",
                    "BIGNUMERIC",
                    "DECIMAL",
                    "INTEGER",
                    "INT64",
                    "FLOAT",
                    "FLOAT64",
                ):
                    select_clauses.append(
                        f"SAFE_CAST(`{field.name}` AS FLOAT64) AS `{field.name}`"
                    )
                else:
                    select_clauses.append(f"`{field.name}`")

            query = f"SELECT {', '.join(select_clauses)} FROM `{full_table_ref}`"
            arrow_table = bq_client.query(query).to_arrow()
            logger.info(f"✅ Loaded {len(arrow_table):,} rows from BigQuery.")
            return arrow_table

        except ImportError:
            logger.warning(
                "⚠️ google‑cloud‑bigquery not found. Returning DuckDB scan macro. "
                "This macro requires the BigQuery extension to be loaded in DuckDB."
            )
            # Return a string that DuckDB can interpret as a table reference
            # (assuming the extension is loaded)
            return f"bigquery_scan('{self.project_id or 'default'}', '{self.dataset_id}', '{self.table_id}')"