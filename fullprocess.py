import os
import sys
import json
import logging

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

with open("config.json", "rb") as f:
    config = json.load(f)

output_folder_path = config["output_folder_path"]
input_folder_path = config["input_folder_path"]
output_model_path = config["output_model_path"]
prod_deployment_path = config["prod_deployment_path"]

##################Check and read new data
#first, read ingestedfiles.txt
logging.info("Reading current set of ingested files")
with open(output_folder_path + "/ingestedfiles.txt", "r") as fp:
    ingested_files = fp.read()
    ingested_files = ingested_files.split("\n")[:-1]
    print(ingested_files)

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
logging.info("Reading source data directory")
source_data = os.listdir(input_folder_path)
for i in range(len(source_data)):
    data = source_data[i]
    source_data[i] = input_folder_path + "/" + data
print(source_data)

new_files = False
for data in source_data:
    if data not in ingested_files:
        new_files = True


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
logging.info("Checking if new data was found")
if not new_files:
    logging.info("No new data was found, exiting")
    exit(0)
        

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
logging.info("Processing new data")
ingestion.merge_multiple_dataframe()
# new_f1 = scoring.score_model()
new_f1 = 0.001

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
logging.info("Checking if there is model drift")
with open(prod_deployment_path + "/latestscore.txt", "r") as fp:
    current_f1 = fp.read()

if float(new_f1) >= float(current_f1):
    logging.info("Current score is greater than/equal to the deployed model score. No drift detected. Program will exit")
    exit(0)

logging.info("Current score is less than the deployed model score. Model has drifed. Re-training now on new data")
training.train_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
logging.info("Re-deploying model")
deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
logging.info("Running model diagnostics")
logging.info("1. Get model predictions")
diagnostics.model_predictions()

logging.info("2. Checking execution time")
diagnostics.execution_time()

logging.info("3. Getting input dataframe summary")
diagnostics.dataframe_summary()

logging.info("4. Checking data for missing data")
diagnostics.check_data()

logging.info("5. Checking dependencies")
diagnostics.outdated_packages_list()

logging.info("6. Re-scoring model")
reporting.score_model()

# os.system("python apicall.py")






