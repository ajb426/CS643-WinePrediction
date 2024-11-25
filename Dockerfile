# Use a base image with Python and Spark pre-installed
FROM bitnami/spark:latest

# Set the working directory in the container
WORKDIR /home/hadoop/assignment2/

# Copy the prediction script and required files
COPY predict_model.py ./predict_model.py
COPY ValidationDataset.csv ./ValidationDataset.csv
COPY WineQualityModel ./WineQualityModel

ENV MODEL_INPUT_PATH="./WineQualityModel"
ENV VALIDATION_DATA_PATH="./ValidationDataset.csv"
ENV OUTPUT_FILE_PATH="./results.txt"

# Install any additional Python dependencies if needed
RUN pip install pyspark
RUN pip install numpy

# Command to run the prediction script
CMD ["spark-submit", "predict_model.py"]
