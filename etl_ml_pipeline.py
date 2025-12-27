import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

# --- Configuration for Local Execution ---
WAREHOUSE_PATH = os.path.abspath("warehouse")
ICEBERG_VERSION = "1.4.3" 
SPARK_MAJOR_VERSION = "3.5"
SCALA_VERSION = "2.12"
JAR_PACKAGES = f"org.apache.iceberg:iceberg-spark-runtime-{SPARK_MAJOR_VERSION}_{SCALA_VERSION}:{ICEBERG_VERSION}"

def get_spark_session():
    """
    Configures Spark to use a local directory as the Iceberg Warehouse.
    No S3 or Cloud credentials required.
    """
    return SparkSession.builder \
        .appName("IcebergDataPipeline") \
        .config("spark.jars.packages", JAR_PACKAGES) \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.local.type", "hadoop") \
        .config("spark.sql.catalog.local.warehouse", WAREHOUSE_PATH) \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def clean_corporate_name(col_name):
    """Normalize names for matching heuristic."""
    c = F.lower(F.col(col_name))
    c = F.regexp_replace(c, r"[.,\-]", "")
    c = F.regexp_replace(c, r"\b(inc|corp|llc|ltd|limited|company)\b", "")
    return F.trim(c)

def ingest_and_harmonize(spark):
    print(">>> Reading Data Sources...")
    df_s1 = spark.read.option("header", "true").csv("data_source/source_supply_chain.csv")
    df_s2 = spark.read.option("header", "true").csv("data_source/source_financial.csv")

    # Create match keys
    df_s1 = df_s1.withColumn("match_key", clean_corporate_name("corporate_name_S1"))
    df_s2 = df_s2.withColumn("match_key", clean_corporate_name("corporate_name_S2"))

    print(">>> Harmonizing Data (Entity Resolution)...")
    harmonized_df = df_s1.alias("s1").join(
        df_s2.alias("s2"), on="match_key", how="full_outer"
    )

    # Consolidated View
    final_df = harmonized_df.select(
        F.coalesce(F.col("s1.corporate_name_S1"), F.col("s2.corporate_name_S2")).alias("corporate_name"),
        F.col("s1.address"),
        F.col("s1.top_suppliers"),
        F.col("s2.revenue").cast(DoubleType()),
        F.col("s2.profit").cast(DoubleType()),
        F.sha2(F.coalesce(F.col("s1.corporate_name_S1"), F.col("s2.corporate_name_S2")), 256).alias("corporate_id"),
        F.current_timestamp().alias("updated_at")
    ).fillna(0, subset=["revenue", "profit"])
    
    return final_df

def upsert_to_iceberg(spark, df):
    table_name = "local.db.corporate_registry"
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.db")
    
    # Check if table exists by trying to read it
    try:
        spark.read.table(table_name)
        table_exists = True
    except:
        table_exists = False

    if not table_exists:
        print(f">>> Creating Iceberg Table {table_name}...")
        # FIX: Removed partitioning by unique ID to prevent OOM
        df.writeTo(table_name).using("iceberg").create()
    else:
        print(f">>> Performing MERGE (Upsert) into {table_name}...")
        df.createOrReplaceTempView("incoming_data")
        query = f"""
        MERGE INTO {table_name} t
        USING incoming_data s
        ON t.corporate_id = s.corporate_id
        WHEN MATCHED THEN
            UPDATE SET t.corporate_name = s.corporate_name, t.revenue = s.revenue, t.profit = s.profit, t.updated_at = s.updated_at
        WHEN NOT MATCHED THEN
            INSERT *
        """
        spark.sql(query)

def train_model(spark):
    print(">>> Starting Local ML Training...")
    # Point MLflow to local directory
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Corporate_Profitability")

    data = spark.read.table("local.db.corporate_registry")
    
    # Feature Engineering
    data = data.withColumn("profit_margin", F.when(F.col("revenue") > 0, F.col("profit") / F.col("revenue")).otherwise(0))
    data = data.withColumn("label", F.when(F.col("profit_margin") > 0.10, 1.0).otherwise(0.0))
    data = data.withColumn("supplier_count", F.size(F.split(F.col("top_suppliers"), ","))).fillna(0, subset=["supplier_count"])

    assembler = VectorAssembler(inputCols=["revenue", "supplier_count"], outputCol="features", handleInvalid="skip")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    pipeline = Pipeline(stages=[assembler, lr])

    train, test = data.randomSplit([0.8, 0.2], seed=42)

    with mlflow.start_run():
        model = pipeline.fit(train)
        predictions = model.transform(test)
        auc = BinaryClassificationEvaluator().evaluate(predictions)
        
        print(f">>> Model AUC: {auc}")
        mlflow.log_metric("auc", auc)
        mlflow.spark.log_model(model, "model")
        print(">>> Model saved to local ./mlruns directory")

if __name__ == "__main__":
    spark = get_spark_session()
    try:
        # 1. Ingest
        df = ingest_and_harmonize(spark)
        # 2. Store
        upsert_to_iceberg(spark, df)
        # 3. Verify
        count = spark.read.table("local.db.corporate_registry").count()
        print(f">>> Total Records in Iceberg: {count}")
        # 4. Train
        train_model(spark)
    finally:
        spark.stop()