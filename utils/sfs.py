import logging
from typing import List, Optional
import pandas as pd
from google.cloud import bigquery

# Configure basic logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BigQueryFeatureSelector:
    """
    A scalable feature selection utility that computes Information Value (IV) 
    and standard deviation directly within Google BigQuery.
    
    This class bypasses the need to pull massive datasets into local memory by 
    constructing and executing distributed SQL queries. It automatically filters 
    out low-variance and low-predictive-power features based on user-defined thresholds.

    Attributes:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The BigQuery dataset ID containing the target table.
        table_id (str): The BigQuery table ID containing the features and target.
        target_column (str): The name of the binary target variable (must be 0 or 1).
        iv_threshold (float): Minimum Information Value required to retain a feature.
        stddev_threshold (float): Minimum standard deviation required to retain a feature.
        bins (int): The number of quantiles (ntiles) to use for the naive IV calculation.
        client (google.cloud.bigquery.Client): The BigQuery client instance.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        target_column: str,
        iv_threshold: float = 0.02,
        stddev_threshold: float = 1e-5,
        bins: int = 10,
        bq_client: Optional[bigquery.Client] = None
    ):
        """
        Initializes the BigQueryFeatureSelector with table coordinates and filtering parameters.
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.target_column = target_column
        self.iv_threshold = iv_threshold
        self.stddev_threshold = stddev_threshold
        self.bins = bins
        
        # Initialize or inject the BigQuery client
        self.client = bq_client if bq_client else bigquery.Client(project=self.project_id)
        
        self.full_table_path = f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"

    def _get_numerical_columns(self) -> List[str]:
        """
        Retrieves the table schema from BigQuery and extracts only the numerical columns.

        Returns:
            List[str]: A list of valid numerical column names, excluding the target column.
        """
        logger.info(f"Fetching schema for table: {self.full_table_path}")
        table_ref = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{self.table_id}")
        
        valid_types = {"INTEGER", "FLOAT", "NUMERIC", "BIGNUMERIC"}
        
        numerical_columns = [
            field.name 
            for field in table_ref.schema 
            if field.field_type in valid_types and field.name != self.target_column
        ]
        
        logger.info(f"Found {len(numerical_columns)} numerical features to evaluate.")
        return numerical_columns
    
    def _get_categorical_columns(self) -> List[str]:
        """
        Retrieves the table schema from BigQuery and extracts only the categorical columns.

        Returns:
            List[str]: A list of valid categorical column names, excluding the target column.
        """
        logger.info(f"Fetching schema for table: {self.full_table_path}")
        table_ref = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{self.table_id}")
        
        valid_types = {"STRING", "BOOLEAN"}
        
        categorical_columns = [
            field.name 
            for field in table_ref.schema 
            if field.field_type in valid_types and field.name != self.target_column
        ]
        
        logger.info(f"Found {len(categorical_columns)} categorical features to evaluate.")
        return categorical_columns

    def _build_sql_query(self, numerical_columns: List[str], categorical_columns: List[str]) -> str:
        """
        Constructs the dynamic parallel SQL query for variance and IV calculation.

        Args:
            numerical_columns (List[str]): The list of numerical column names to process.
            categorical_columns (List[str]): The list of categorical column names to process.

        Returns:
            str: The fully constructed executable SQL query string.
        """
        sql_parts = []
        
        for col in numerical_columns:
            part = f"""
                    SELECT 
                    '{col}' AS feature_name,
                    MAX(feature_stddev) AS feature_stddev,
                    SUM(
                    CASE WHEN (bads_in_bin = 0 OR goods_in_bin = 0) THEN 0
                    ELSE
                    ((goods_in_bin / NULLIF(total_goods, 0)) - (bads_in_bin / NULLIF(total_bads, 0))) * LN((goods_in_bin / NULLIF(total_goods, 0) + 0.0001) / (bads_in_bin / NULLIF(total_bads, 0) + 0.0001))
                    ) AS naive_iv
                    FROM (
                    SELECT 
                    bin,
                    COUNTIF({self.target_column} = 0) AS goods_in_bin,
                    COUNTIF({self.target_column} = 1) AS bads_in_bin,
                    SUM(COUNTIF({self.target_column} = 0)) OVER() AS total_goods,
                    SUM(COUNTIF({self.target_column} = 1)) OVER() AS total_bads,
                    MAX(feature_stddev) AS feature_stddev
                    FROM (
                    SELECT 
                    NTILE({self.bins}) OVER(ORDER BY {col}) AS bin,
                    {self.target_column},
                    STDDEV({col}) OVER() AS feature_stddev
                    FROM {self.full_table_path}
                    WHERE {col} IS NOT NULL
                    )
                    GROUP BY bin
                    )
            """
            sql_parts.append(part)

        for col in categorical_columns:
            part = f"""
                    SELECT 
                    '{col}' AS feature_name,
                    MAX(feature_stddev) AS feature_stddev,
                    SUM(
                    CASE WHEN (bads_in_bin = 0 OR goods_in_bin = 0) THEN 0
                    ELSE
                    ((goods_in_bin / NULLIF(total_goods, 0)) - (bads_in_bin / NULLIF(total_bads, 0))) * LN((goods_in_bin / NULLIF(total_goods, 0) + 0.0001) / (bads_in_bin / NULLIF(total_bads, 0) + 0.0001))
                    ) AS naive_iv
                    FROM (
                    SELECT 
                    bin,
                    COUNTIF({self.target_column} = 0) AS goods_in_bin,
                    COUNTIF({self.target_column} = 1) AS bads_in_bin,
                    SUM(COUNTIF({self.target_column} = 0)) OVER() AS total_goods,
                    SUM(COUNTIF({self.target_column} = 1)) OVER() AS total_bads,
                    MAX(feature_stddev) AS feature_stddev
                    FROM (
                    SELECT 
                    CAST({col} AS STRING) AS bin,
                    {self.target_column},
                    9999 AS feature_stddev
                    FROM {self.full_table_path}
                    WHERE {col} IS NOT NULL
                    )
                    GROUP BY bin
                    )
            """
            sql_parts.append(part)

        # Union all individual column calculations together
        union_query = "\nUNION ALL\n".join(sql_parts)

        # Wrap in an outer query for clean output and ordering
        final_query = f"""
        SELECT 
            feature_name, 
            feature_stddev, 
            naive_iv 
        FROM (
            {union_query}
        )
        ORDER BY naive_iv DESC;
        """
        return final_query

    def screen_features(self) -> pd.DataFrame:
        """
        Executes the screening pipeline: identifies columns, runs the BigQuery calculations, 
        and filters the results based on user-defined thresholds.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the retained features, 
                          their standard deviations, and Information Values.
        """
        numerical_columns = self._get_numerical_columns()
        categorical_columns = self._get_categorical_columns()
        columns_to_screen = numerical_columns + categorical_columns
        if not numerical_columns and not categorical_columns:
            logger.warning("No columns found to screen.")
            return pd.DataFrame()

        query = self._build_sql_query(numerical_columns, categorical_columns)

        logger.info("Executing distributed IV and Variance calculations in BigQuery...")
        try:
            query_job = self.client.query(query)
            results_df = query_job.to_dataframe()
            logger.info("Successfully fetched screening metrics.")
        except Exception as e:
            logger.error(f"Failed to execute BigQuery job: {e}")
            raise

        # Apply the statistical filters
        retained_features_df = results_df[
            (results_df['feature_stddev'] > self.stddev_threshold) & 
            (results_df['naive_iv'] >= self.iv_threshold)
        ].copy()

        # Log the summary statistics
        dropped_count = len(columns_to_screen) - len(retained_features_df)
        logger.info(f"Screening complete. Retained: {len(retained_features_df)} | Dropped: {dropped_count}")
        
        return retained_features_df
    

    #Usage
    # if __name__ == "__main__":
    # # 1. Initialize the screener with your specific table and parameters
    # screener = BigQueryFeatureSelector(
    #     project_id="my-company-gcp-project",
    #     dataset_id="analytics_dataset",
    #     table_id="customer_base_30m",
    #     target_column="is_default",
    #     iv_threshold=0.02,        # Drop features with IV below 0.02
    #     stddev_threshold=0.0001,  # Drop zero-variance/constant features
    #     bins=10                   # Calculate using Deciles
    # )

    # # 2. Execute the screening process
    # selected_features_df = screener.screen_features()

    # # 3. View the surviving features
    # print(selected_features_df.head(20))
    
    # # 4. Extract just the feature names as a list for your next pipeline step
    # final_feature_list = selected_features_df['feature_name'].tolist()
