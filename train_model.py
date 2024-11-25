from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WineQualityTraining") \
    .getOrCreate()

# HDFS path for the training dataset
training_data_path = os.getenv("TRAINING_DATA_PATH")
model_output_path = os.getenv("MODEL_OUTPUT_PATH")

# Load the training dataset from HDFS
print(f"Loading training dataset from: {training_data_path}")
training_data = spark.read.csv(
    training_data_path,
    header=True,
    inferSchema=True,
    sep=";"
)

# Clean column names to remove extra double quotes
training_data = training_data.toDF(*[col.replace('"', '') for col in training_data.columns])

# Assemble feature columns into a single vector
feature_columns = training_data.columns[:-1]  # All columns except the target
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Prepare the Random Forest classifier
rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=100)

# Create a pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Train the model
print("Training the model...")
model = pipeline.fit(training_data)

# Save the trained model to HDFS
print(f"Saving the model to: {model_output_path}")
model.write().overwrite().save(model_output_path)

print("Training completed successfully!")

# Stop the Spark session
spark.stop()
