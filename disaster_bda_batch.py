#!/usr/bin/env python3
"""
🔥 DisasterAI: Comprehensive Disaster Management & Prediction Pipeline (Enhanced)
------------------------------------------------------------------------------
Performs:
✅ Data Cleaning & Preprocessing
✅ Descriptive Analytics (Top disaster types, countries, continents)
✅ Geo Clustering (Latitude, Longitude)
✅ Regression & Classification (using RandomForest)
✅ Yearly Forecast (Linear Regression)
✅ Correlation Matrix Creation (for dashboard)
✅ Auto-Save Outputs & Model Files
"""

import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as _sum, count, desc, when, monotonically_increasing_id
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# -------------------------------------------------------------------
# 🗂 Configuration
# -------------------------------------------------------------------
DATA_PATH = "file:///home/adminvarshitha/bigdata/datasets/Disaster.csv"
OUTPUT_DIR = "/home/adminvarshitha/bigdata/projects/output_disasterAI"
MODEL_DIR = "/home/adminvarshitha/bigdata/projects/models_disasterAI"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "pipeline_summary.txt")

# -------------------------------------------------------------------
# ⚙️ Spark Session
# -------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("DisasterAI - Enhanced BDA") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("\n" + "=" * 80)
print("🚀 DISASTER AI: ENHANCED PIPELINE STARTED")
print("=" * 80)

# -------------------------------------------------------------------
# 📥 Load Dataset
# -------------------------------------------------------------------
print(f"Loading dataset from: {DATA_PATH}")
df = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)
print(f"Initial records: {df.count()}")

# -------------------------------------------------------------------
# 🧹 Data Cleaning
# -------------------------------------------------------------------
drop_cols = [
    "Seq", "Glide", "Disaster Subsubtype", "Associated Dis", "Associated Dis2",
    "OFDA Response", "Appeal", "Declaration", "Aid Contribution", "ISO"
]
df = df.drop(*[c for c in drop_cols if c in df.columns])

numeric_candidates = [
    "Total Deaths", "No Injured", "No Affected", "No Homeless",
    "Total Affected", "Insured Damages ('000 US$)", "Total Damages ('000 US$)",
    "Dis Mag Value", "Latitude", "Longitude", "Start Year", "Start Month",
    "Start Day", "End Year", "End Month", "End Day"
]
for c in numeric_candidates:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast("double"))

string_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == 'string']
numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() != 'string']
if string_cols: df = df.fillna("Unknown", subset=string_cols)
if numeric_cols: df = df.fillna(0, subset=numeric_cols)

print(f"Cleaned records: {df.count()}")

# -------------------------------------------------------------------
# 📊 Descriptive Analytics
# -------------------------------------------------------------------
def group_top(df, group_col, agg_col="*", func=count, alias="count"):
    return df.groupBy(group_col).agg(func(agg_col).alias(alias)).orderBy(desc(alias)).limit(10)

top_types = group_top(df, "Disaster Type") if "Disaster Type" in df.columns else None
top_countries = group_top(df, "Country", "Total Damages ('000 US$)", _sum, "total_damage") if "Country" in df.columns else None

year_col = next((c for c in ["Start Year", "Year", "Start_Year"] if c in df.columns), None)
yearly_trend = (
    df.groupBy(year_col)
      .agg(count("*").alias("num_events"),
           _sum("Total Affected").alias("total_affected"),
           _sum(col("Total Damages ('000 US$)")).alias("total_damage"))
      .orderBy(year_col)
) if year_col else None

continent_summary = (
    df.groupBy("Continent")
      .agg(count("*").alias("events"),
           _sum("Total Affected").alias("total_affected"),
           _sum(col("Total Damages ('000 US$)")).alias("total_damage"))
      .orderBy(desc("events"))
) if "Continent" in df.columns else None

# -------------------------------------------------------------------
# 🗺️ Geo Clustering (Latitude / Longitude)
# -------------------------------------------------------------------
if "Latitude" in df.columns and "Longitude" in df.columns:
    geo_points = df.filter((col("Latitude") != 0) & (col("Longitude") != 0))
    if geo_points.count() >= 10:
        df = df.withColumn("row_id", monotonically_increasing_id())
        assembler_geo = VectorAssembler(inputCols=["Latitude", "Longitude"], outputCol="features_geo")
        geo_feats = assembler_geo.transform(df.select("row_id", "Latitude", "Longitude"))
        kmeans = KMeans(k=6, featuresCol="features_geo", seed=42)
        model_geo = kmeans.fit(geo_feats)
        centers = model_geo.clusterCenters()
        df = df.join(model_geo.transform(geo_feats).select("row_id", "prediction"), "row_id", "left").withColumnRenamed("prediction", "geo_cluster")
        print(f"✅ Geo Clustering Done. Cluster centers: {len(centers)}")
    else:
        print("⚠️ Not enough geo points for clustering (min 10).")

# -------------------------------------------------------------------
# 🔢 Feature Engineering
# -------------------------------------------------------------------
cat_cols = [c for c in ["Disaster Type", "Disaster Group", "Disaster Subgroup", "Continent", "Region"] if c in df.columns]
num_features = [c for c in numeric_candidates if c in df.columns]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols], outputCols=[f"{c}_ohe" for c in cat_cols])
pipeline = Pipeline(stages=indexers + [encoders])
df_trans = pipeline.fit(df).transform(df)

assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in cat_cols] + num_features, outputCol="features", handleInvalid="keep")
data_ready = assembler.transform(df_trans)

# -------------------------------------------------------------------
# 🧮 Regression (Total Damages)
# -------------------------------------------------------------------
if "Total Damages ('000 US$)" in df.columns:
    reg_df = data_ready.select("features", col("Total Damages ('000 US$)").alias("label")).na.drop()
    if reg_df.count() > 50:
        tr, te = reg_df.randomSplit([0.8, 0.2], seed=42)
        model_reg = RandomForestRegressor(numTrees=40, maxDepth=10).fit(tr)
        preds = model_reg.transform(te)
        rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(preds)
        model_reg.write().overwrite().save(os.path.join(MODEL_DIR, "rf_reg_total_damages"))
        print(f"📈 Regression RMSE (Total Damages): {rmse:.2f}")

# -------------------------------------------------------------------
# 🎯 Classification (Severity)
# -------------------------------------------------------------------
if "Total Affected" in df.columns:
    df_cls = df_trans.withColumn("severity_label",
        when(col("Total Affected") <= 1000, "Low")
        .when(col("Total Affected") <= 10000, "Medium")
        .otherwise("High")
    )
    idx = StringIndexer(inputCol="severity_label", outputCol="label").fit(df_cls)
    df_cls = idx.transform(df_cls)
    cls_df = assembler.transform(df_cls).select("features", "label").na.drop()
    if cls_df.count() > 100:
        tr_c, te_c = cls_df.randomSplit([0.8, 0.2], seed=42)
        rf_cls = RandomForestClassifier(numTrees=40)
        model_cls = rf_cls.fit(tr_c)
        acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(model_cls.transform(te_c))
        model_cls.write().overwrite().save(os.path.join(MODEL_DIR, "rf_classifier_severity"))
        print(f"🎯 Classification Accuracy: {acc*100:.2f}%")

# -------------------------------------------------------------------
# 🔮 Forecast (Yearly Total Affected)
# -------------------------------------------------------------------
if yearly_trend is not None:
    yr_df = yearly_trend.select(col(year_col).alias("year_num"), "total_affected").na.drop()
    assembler_time = VectorAssembler(inputCols=["year_num"], outputCol="features_time")
    data_time = assembler_time.transform(yr_df.select(col("year_num").cast("double"), col("total_affected").alias("label")))
    if data_time.count() > 5:
        tr_t, te_t = data_time.randomSplit([0.8, 0.2], seed=42)
        lr_time = LinearRegression(featuresCol="features_time", labelCol="label").fit(tr_t)
        rmse_time = RegressionEvaluator(metricName="rmse").evaluate(lr_time.transform(te_t))
        print(f"📊 Yearly Forecast RMSE: {rmse_time:.2f}")

# -------------------------------------------------------------------
# 🔗 Correlation Analysis
# -------------------------------------------------------------------
corr_file = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
try:
    pdf = df.select(*num_features).toPandas().corr()
    pdf.to_csv(corr_file)
    print(f"🧩 Correlation matrix saved: {corr_file}")
except Exception as e:
    print(f"⚠️ Correlation computation failed: {e}")

# -------------------------------------------------------------------
# 💾 Save Summary Outputs
# -------------------------------------------------------------------
def save_df(dfobj, name):
    if dfobj:
        dfobj.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(OUTPUT_DIR, name))

save_df(top_types, "top_disaster_types")
save_df(top_countries, "top_countries_damage")
save_df(yearly_trend, "yearly_trend")
save_df(continent_summary, "continent_summary")

with open(SUMMARY_TXT, "w") as f:
    f.write("=== DisasterAI Summary Report ===\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Records after cleaning: {df.count()}\n")
    f.write(f"Models saved: {MODEL_DIR}\n")
    f.write(f"Outputs saved: {OUTPUT_DIR}\n")

print("\n✅ Pipeline Completed Successfully!")
print(f"📂 Outputs: {OUTPUT_DIR}")
print(f"🤖 Models: {MODEL_DIR}")
spark.stop()
