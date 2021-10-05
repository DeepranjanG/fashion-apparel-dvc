# load the train and test file
# train algo
# save the metrices, params

import os
import warnings
import sys
import pandas as pd
import numpy as np
from get_data import read_params
import argparse
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
from urllib.parse import urlparse


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir = config["model_dir"]

    target = [config["base"]["target_col"]]

    loss = config["estimators"]["CnnNetwork"]["params"]["loss"]
    optimiser = config["estimators"]["CnnNetwork"]["params"]["optimiser"]
    metrics = config["estimators"]["CnnNetwork"]["params"]["metrics"]
    activation = config["estimators"]["CnnNetwork"]["params"]["activation"]
    batch_size = config["estimators"]["CnnNetwork"]["params"]["batch_size"]
    epochs = config["estimators"]["CnnNetwork"]["params"]["epochs"]
    

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    train_images = np.array(train_x) / 255.0
    test_images = np.array(test_x) / 255.0

    
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        model = keras.Sequential([
        keras.Input(shape=(784)),
        layers.Reshape((28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation=activation)])

        
        model.compile(loss=loss, optimizer=optimiser, metrics=metrics)

        model.fit(x=train_images, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_y))

        eval_metrics = model.evaluate(test_images, test_y)

        mlflow.log_param("loss_function", loss)
        mlflow.log_param("optimiser", optimiser)
        mlflow.log_param("activation", activation)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        mlflow.log_metric("LOSS", eval_metrics[0])
        mlflow.log_metric("ACCURACY", eval_metrics[1])

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.keras.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.keras.load_model(model, "model")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
