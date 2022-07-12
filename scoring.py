from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    
    # load test data
    test_data = pd.read_csv(test_data_path + "/testdata.csv")
    y_test = test_data.pop("exited")
    X_test = test_data.drop("corporation", axis=1)

    # load model
    with open(output_model_path + "/trainedmodel.pkl", "rb") as fp:
        model = pickle.load(fp)
    
    # get predictions
    y_pred = model.predict(X_test)

    # calculate f1 score
    f1 = metrics.f1_score(y_test, y_pred)
    f1 = str(f1)

    # save score to .txt file
    with open(output_model_path + "/latestscore.txt", "w") as fp:
        fp.write(str(f1))
    
    return f1
    

if __name__ == "__main__":
    score_model()

