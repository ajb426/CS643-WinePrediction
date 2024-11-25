# Wine Quality Prediction

This project uses a machine learning model to predict wine quality based on various attributes. The script is designed to be run within a Docker container or directly on a single Spark node.

##Links
1. Github: https://github.com/ajb426/CS643-WinePrediction
2. Dockerhub: https://hub.docker.com/repository/docker/ajb426/wine-quality-predictor/tags

## Features

- Train a Random Forest model on a wine dataset.
- Evaluate the model on a validation dataset.
- Output the F1 score of the predictions.

---

## Prerequisites

### To run locally:
- Python 3.9
- Apache Spark 3.5
- Spark Cluster
- Hadoop 3.4.0

### To run using Docker:
- Docker installed on your system
- Access to a Docker-compatible terminal (e.g., bash, PowerShell)

---

## Project Structure

- `train_model.py`: Script for training the Random Forest model.
- `predict_model.py`: Script for generating predictions and calculating F1 score.
- `TrainingDataset.csv`: Training dataset used for training the model.
- `ValidationDataset.csv`: Validation dataset for predictions.
- `WineQualityModel/`: Directory containing the trained model.
- `Dockerfile`: Configuration to build a Docker image for running the prediction script.

---

## Steps to Run

### **1. Running Locally**

#### Setup the Environment
1. Clone this repository on all nodes in the cluster:
   ```bash
   git clone https://github.com/ajb426/CS643-WinePrediction.git
   cd CS643-WinePrediction ```

### Create Python Virtual Environment
1. Create a virutal environment and install required python libraries on all nodes
	```python
	python -m venv venv
	source venv/bin/activate
	```
	```python
	pip install -r requirements.txt```

### Upload Datasets
1. Create a working directory in hdfs.
	```bash
	hdfs dfs -mkdir /working/directory/path
	```
2. Add the TrainingDataset to this working directory.
	```bash
	hdfs dfs -put TrainingDataset.csv /assignment2/TrainingDataset.csv
	```
	Replace the with the relevant path for your HDFS.

### Train Model
1. Train and ouput the model for prediction:
	```bash
	spark-submit --master yarn --deploy-mode cluster --conf spark.yarn.appMasterEnv.TRAINING_DATA_PATH=hdfs:///hdfs/path/to/TrainingDataset.csv  --conf spark.yarn.appMasterEnv.MODEL_OUTPUT_PATH=hdfs:///hdfs/path/to/WineQualityModel train_model.py

 set the env variables to the relevant paths for your training and model paths.
 
2. Get the model output from HDFS
	```bash
	hdfs dfs -get /hdfs/path/to/WineQualityModel /local/file/path/WineQualityModel
	```
### Predictions
1. To run the predictions script locally on the master node:
	```bash
	export MODEL_INPUT_PATH=file:///local/file/path/WineQualityModel
	export VALIDATION_DATA_PATH=file:///local/file/path/ValidationDataset.csv
	export OUTPUT_FILE_PATH=local/file/path.txt
	```
2. Then run the predict_model.py
	```bash
	spark-submit predict_model.py
	```
The results will be saved in the output file you specified in OUTPUT_FILE_PATH

3. To run the predictions script using the docker image:
	```bash
	docker pull ajb426/wine-quality-predictor:latest
	```
	Then run this command and take note of the image ID.
	```bash
	docker images
	```
	Then run this command using the image ID you found from the previous command.
	```bash
	docker run --name test_container <ImageID>
	```
	Once that is complete run this command to extract the results file from the container.
	```bash
	docker export test_container | tar -x --wildcards -f - "home/hadoop/assignment2/= ./results.txt" --transform='s|.*/||'
	```
	Make sure you are logged into Docker in your shell before executing the above commands.
	
## Summary
Those are all the steps required to run my Spark cluster for the wine prediction model. Please reference this document provided by the professor for creating the Spark cluster if
there are any questions pertaining to that. [Setup Cluster using EMR](Final%20Guide.pdf)
	
	
	



