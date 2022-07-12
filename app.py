from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions, dataframe_summary, execution_time, check_data, outdated_packages_list
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    dataset_path = request.get_json()['dataset_path']
    y_pred, _ = model_predictions(dataset_path)
    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    f1 = score_model()
    return f1

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary = dataframe_summary()
    return str(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    execution = execution_time()
    missing_data = check_data()
    packages = outdated_packages_list()
    return str("execution_time:" + execution + "\nmissing_data:" + missing_data + "\noutdated_packages:" + packages)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
