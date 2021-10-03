# read the data from data source
# save it in the data /raw for furthyr process
import os
from get_data import read_params, get_data
import argparse

def load_and_save(config_path):
    config = read_params(config_path)
    df_train, df_test = get_data(config_path)
    raw_train_data_path = config['load_data']['raw_train_dataset_csv']
    raw_test_data_path = config['load_data']['raw_test_dataset_csv']
    df_train.to_csv(raw_train_data_path)
    df_test.to_csv(raw_test_data_path)
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
    