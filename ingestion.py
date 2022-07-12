import pandas as pd
import numpy as np
import glob
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
data_columns = config['columns']

#############Function for data ingestion
def merge_multiple_dataframe():
    data_files = glob.glob(input_folder_path + '/*.csv')
    df = pd.DataFrame(columns=data_columns)
    
    with open(output_folder_path + "/ingestedfiles.txt", "w") as f:
        for file in data_files:
            current_df = pd.read_csv(file)
            df = df.append(current_df).reset_index(drop=True)
            f.write(file + "\n")
    
    df = df.drop_duplicates()
    df.to_csv(output_folder_path + "/finaldata.csv", index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
