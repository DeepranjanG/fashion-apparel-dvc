## read params
## process
## return the dataframe
import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path_train = config["data_source"]['s3_source_train']
    data_path_test = config["data_source"]['s3_source_test']
    df_train = pd.read_csv(data_path_train)
    df_test = pd.read_csv(data_path_test)
    return df_train, df_test

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
    