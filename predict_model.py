from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .master("local[*]") \
    .getOrCreate()

model_path = os.getenv("MODEL_INPUT_PATH")
validation_data_path = os.getenv("VALIDATION_DATA_PATH")
output_path =  os.getenv("OUTPUT_FILE_PATH")

# Load the model from the local filesystem
print(f"Loading model from: {model_path}")
try:
    model = PipelineModel.load(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    spark.stop()
    exit(1)

# Load the validation dataset from the local filesystem
print(f"Loading validation data from: {validation_data_path}")
try:
    validation_data = spark.read.csv(
        validation_data_path,
        header=True,
        inferSchema=True,
        sep=";"
    )
except Exception as e:
    print(f"Failed to load validation dataset: {e}")
    spark.stop()
    exit(1)

validation_data = validation_data.toDF(*[col.replace('"', '') for col in validation_data.columns])

# Make predictions
print("Making predictions...")
predictions = model.transform(validation_data)

# Evaluate predictions using F1 score
evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="f1"
)
f1_score = evaluator.evaluate(predictions)

# Print F1 score
print(f"Validation F1 Score: {f1_score}")

# Save F1 score to a local file
try:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Validation F1 Score: {f1_score}\n")
    print(f"F1 score saved to: {output_path}")
except Exception as e:
    print(f"Failed to save F1 score: {e}")

# Stop the Spark session
spark.stop()
