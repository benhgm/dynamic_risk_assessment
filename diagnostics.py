
import pandas as pd
import numpy as np
import pickle
import timeit
import os
import json
import subprocess
import sys


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deployment_path = os.path.join(config['prod_deployment_path'])


##################Function to get model predictions
def model_predictions():
    # read test data and remove unwanted columns
    test_data = pd.read_csv(test_data_path + "/testdata.csv")
    y_test = test_data.pop("exited")
    X_test = test_data.drop("corporation", axis=1)

    print(X_test)
    print(y_test)

    # load production model
    with open(deployment_path + "/trainedmodel.pkl", "rb") as fp:
        model = pickle.load(fp)
    
    # get predictions
    y_pred = model.predict(X_test)

    return y_pred, y_test


##################Function to get summary statistics
def dataframe_summary():
    # read dataset
    df = pd.read_csv(dataset_csv_path + "/finaldata.csv")
    
    # init a dictionary
    statistics = {}

    # store data in the dictionary
    statistics['mean'] = df.mean(axis=0)
    statistics['median'] = df.median(axis=0)
    statistics['std'] = df.std(axis=0)

    return statistics 


##################Function to get summary statistics
def check_data():
    # read dataset
    df = pd.read_csv(dataset_csv_path + "/finaldata.csv")

    # check if there are any null values
    if df.isnull().values.any():
        print("Missing data found, see below for missing data statistics")
        print(df.isnull().mean()*100)
    else:
        print("No missing data found")
    return df.isnull().mean()*100
        

##################Function to get timings
def execution_time():
    # init dictionary
    timings = {}
    execute = ["ingestion", "training"]

    # measure timing of execution
    for job in execute:
        start_time = timeit.default_timer()
        os.system(f'python {job}.py')
        timing = timeit.default_timer() - start_time
        timings[job] = timing

    return timings

##################Function to check dependencies
def outdated_packages_list():
    # get a list of outdated packages
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    check_data()
    execution_time()
    outdated_packages_list()





    
