from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when

# ✅ Create Spark Session
spark = SparkSession.builder \
    .appName("Smart Agriculture Batch Processing") \
    .master("local[*]") \
    .getOrCreate()

# ✅ Input dataset path (your dataset path inside WSL)
input_path = "file:///home/adminvarshitha/bigdata/datasets/plant_health_dataset.csv"

# ✅ Output folder for processed results
output_path = "file:///home/adminvarshitha/bigdata/projects/output_enhanced_fixed"

# ✅ Read the dataset
df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

# Show schema and sample data
print("✅ Data Loaded Successfully!")
df.printSchema()
df.show(5)

# ✅ Data Cleaning: Handle nulls and replace with average
numeric_cols = [field.name for field in df.schema.fields if str(field.dataType) in ("IntegerType", "DoubleType")]

for col_name in numeric_cols:
    avg_value = df.select(avg(col(col_name))).first()[0]
    df = df.withColumn(col_name, when(col(col_name).isNull(), avg_value).otherwise(col(col_name)))

# ✅ Feature Engineering: Compute average plant health index
if "moisture" in df.columns and "temperature" in df.columns and "humidity" in df.columns:
    df = df.withColumn("plant_health_index", 
                       (col("moisture") * 0.4 + col("temperature") * 0.3 + col("humidity") * 0.3))

# ✅ Save cleaned and enhanced dataset to output folder
df.write.mode("overwrite").option("header", True).csv(output_path)

print("🌾 Batch processing completed successfully!")
print(f"✅ Output saved at: {output_path}")

spark.stop()
