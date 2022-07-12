import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    y_pred, y_test = model_predictions()
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    plot = sns.heatmap(confusion_matrix, annot=True)

    figure = plot.get_figure()
    figure.savefig(output_model_path + '/confusionmatrix2.png', dpi=400)


if __name__ == '__main__':
    score_model()
