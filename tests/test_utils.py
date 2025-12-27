import sys
import os
import pytest
from pyspark.sql import SparkSession

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etl_ml_pipeline import clean_corporate_name

@pytest.fixture(scope="session")
def spark():
    """
    Creates a test Spark Session.
    We use local[1] to minimize resource usage during tests.
    """
    return SparkSession.builder \
        .master("local[1]") \
        .appName("TestSession") \
        .getOrCreate()

def test_clean_corporate_name(spark):
    """
    Test the name normalization logic (Entity Resolution Heuristic).
    """
    data = [
        ("Acme Corp.", "acme"),
        ("Beta, Inc", "beta"),
        ("Gamma LLC", "gamma"),
        ("Delta-Sigma Limited", "deltasigma"),
        ("The Omega Company", "the omega")
    ]
    
    # Create a DataFrame to test the PySpark function
    df = spark.createDataFrame(data, ["raw_name", "expected"])
    
    # Apply the function
    result_df = df.withColumn("cleaned", clean_corporate_name("raw_name"))
    
    # Collect results
    results = result_df.collect()
    
    # Verify
    for row in results:
        assert row["cleaned"] == row["expected"], \
            f"Failed: {row['raw_name']} -> {row['cleaned']} (Expected: {row['expected']})"