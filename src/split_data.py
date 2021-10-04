# split the raw data 
# save it in data/processed folder

import os
import argparse
import pandas as pd
from get_data import read_params

def split_and_saved_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_train_data_path = config["load_data"]["raw_train_dataset_csv"]
    raw_test_data_path = config["load_data"]["raw_test_dataset_csv"]


    train = pd.read_csv(raw_train_data_path)
    test = pd.read_csv(raw_test_data_path)
    
    train.to_csv(train_data_path,sep=",", index=False)
    test.to_csv(test_data_path,sep=",", index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)