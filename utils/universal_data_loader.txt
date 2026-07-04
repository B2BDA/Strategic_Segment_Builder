"""
Unified Data Ingestion Layer for Strategic Analytics
===================================================
A multi-format data loader abstraction supporting Local Files (CSV, Parquet, Arrow, Excel),
In-Memory PyArrow Tables, and Google Cloud BigQuery Storage API streams.
"""

import os
import logging
from typing import Any, Optional, Union
import duckdb
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_pq

logger = logging.getLogger("StrategicEngine.DataLoader")


class UniversalDataLoader:
    """Handles multi-source data ingestion, normalizing inputs into highly optimized
    in-memory structures compatible with vectorized down-stream compute engines.
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
        """Auto-detects the source configuration parameters and loads the dataset
        into an optimized Arrow Table or native extension scan string.

        Args:
            fallback_data: Direct passing of an in-memory object (e.g., PyArrow Table).

        Returns:
            An ingestion asset that can be passed directly to StrategicSegmentBuilder.
        """
        # Scenario 1: Direct In-Memory Object Check
        if fallback_data is not None:
            if isinstance(fallback_data, pa.Table):
                logger.info("Ingesting directly provided in-memory PyArrow Table.")
                return fallback_data
            # If it's already a format DuckDB handles natively (Dict, list of dicts)
            return fallback_data

        # Scenario 2: BigQuery Parameters Detected
        if self.dataset_id and self.table_id:
            return self._load_from_bigquery()

        # Scenario 3: Local File Path Detection
        if self.file_path:
            return self._load_from_file()

        raise ValueError(
            "Invalid Configuration: You must provide either a valid `file_path`, "
            "BigQuery identifiers (`dataset_id` + `table_id`), or pass an explicit `fallback_data` object."
        )

    def _load_from_file(self) -> pa.Table:
        """Parses local file extensions and routes to low-overhead C++ Arrow readers
        or custom zero-Pandas tabular structures.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Target local data file not found at: {self.file_path}")

        ext = os.path.splitext(self.file_path)[-1].lower()
        logger.info(f"Detecting local file format handler for extension: '{ext}'...")

        if ext == ".parquet":
            return pa_pq.read_table(self.file_path)
        
        elif ext == ".csv":
            # Multi-threaded native streaming CSV parse
            return pa_csv.read_csv(self.file_path)
        
        elif ext in [".arrow", ".feather"]:
            with pa.memory_map(self.file_path, "r") as source:
                return pa.ipc.open_file(source).read_all()

        elif ext in [".xlsx", ".xls"]:
            return self._load_excel_to_arrow()
        
        raise ValueError(f"Unsupported file format extension: '{ext}'. Use Parquet, CSV, Arrow/Feather, or Excel.")

    def _load_excel_to_arrow(self) -> pa.Table:
        """Parses an Excel sheet into a dictionary layout and structures it directly 

        as a PyArrow Table without utilizing Pandas.
        """
        logger.info("Parsing Excel spreadsheet via raw matrix extraction...")
        try:
            import openpyxl
            wb = openpyxl.load_workbook(self.file_path, data_only=True, read_only=True)
            sheet = wb.active  # Reads the primary active sheet
            
            rows_generator = sheet.iter_rows(values_only=True)
            headers = next(rows_generator)
            
            if not headers:
                raise ValueError(f"The active sheet in {self.file_path} appears to be completely empty.")
            
            # Initialize a dictionary of lists mapping headers to columns
            columns_dict = {str(h): [] for h in headers if h is not None}
            header_keys = list(columns_dict.keys())
            num_cols = len(header_keys)

            for row in rows_generator:
                # Pad or slice rows to ensure exact alignment with header lengths
                for idx in range(num_cols):
                    val = row[idx] if idx < len(row) else None
                    columns_dict[header_keys[idx]].append(val)
            
            wb.close()
            return pa.Table.from_pydict(columns_dict)

        except ImportError:
            raise ImportError(
                "To ingest Excel files without Pandas, you must install the lightweight "
                "dependency library: `pip install openpyxl`"
            )

    def _load_from_bigquery(self) -> Union[pa.Table, str]:
        """Resolves BigQuery ingestion using either the High-Speed Storage API
        (via explicit client dependency) or native DuckDB scanning hooks.
        """
        # Explicit target definition string construction
        project_prefix = f"{self.project_id}." if self.project_id else ""
        full_bq_path = f"{project_prefix}{self.dataset_id}.{self.table_id}"

        try:
            from google.cloud import bigquery
            logger.info(f"Initializing Google Cloud BigQuery Client storage stream for: {full_bq_path}")
            
            bq_client = bigquery.Client(project=self.project_id)
            query = f'SELECT * FROM `{full_bq_path}`'
            
            # Streams data via high-speed gRPC Storage API directly to internal Arrow chunks
            return bq_client.query(query).to_arrow()

        except ImportError:
            # Resilient Fallback: If google-cloud-bigquery package is missing, 
            # we return the raw native SQL scan macro string for DuckDB's native execution engine.
            if not self.project_id:
                raise ValueError(
                    "DuckDB Native BigQuery extension scans require an explicit `project_id` parameter specified."
                )
            
            logger.warning(
                "google-cloud-bigquery python package not found in current environment. "
                "Routing compilation to native DuckDB connection scan string macro macro."
            )
            return f"bigquery_scan('{self.project_id}', '{self.dataset_id}', '{self.table_id}')"