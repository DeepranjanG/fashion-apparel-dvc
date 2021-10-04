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
import joblib
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

    model = keras.Sequential(
    [
        keras.Input(shape=(784)),
        layers.Reshape((28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation=activation),
    ])

    model.compile(loss=loss, optimizer=optimiser, metrics=metrics)

    model.fit(x=train_images, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_y))

    eval_metrics = model.evaluate(test_images, test_y)

    # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    print("  LOSS: %s" % eval_metrics[0])
    print("  ACCURACY: %s" % eval_metrics[1])

    
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]['params']

    with open(scores_file, "w") as f:
        scores = {
            "loss" : eval_metrics[0],
            "accuracy" : eval_metrics[1]
        }
        json.dump(scores, f, indent=4)
    
    with open(params_file, "w") as f:
        params = {
            "loss_function" : loss,
            "optimiser" : optimiser,
            "metrics" : metrics,
            "activation" : activation,
            "batch_size" : batch_size,
            "epochs" : epochs
        }
        json.dump(params, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)    
    model.save(os.path.join(model_dir, "full_mnist_model.h5"))

    # joblib.dump(model, model_path)




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
