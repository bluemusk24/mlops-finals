#!/usr/bin/env python
# coding: utf-8


import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

RUN_ID = os.getenv("RUN_ID", "6f4e767405144ea0abe4bdac7bdcbb43")

# data preprocessing function


def get_data(data):

    df = pd.read_csv(data)

    df = df.rename(
        columns={"V": "voltage", "H": "height", "S": "soil_types", "M": "mine_types"}
    )

    df["mine_types"] = df["mine_types"].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    df.reset_index(drop=True)

    num_variables = ["voltage", "height"]
    cat_variable = ["soil_types"]
    df[cat_variable] = df[cat_variable].astype(str)

    return df


# Splitting dataset function


def split_dataset(df):

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_full_train, df_train, df_val, df_test


# Prepare dictionaries function


def prepare_dictionaries(df):

    num_variables = ["voltage", "height"]
    cat_variable = ["soil_types"]

    dicts = df[cat_variable + num_variables].to_dict(orient="records")
    return dicts


# Function to get the target value


def get_y(df):
    y = df["mine_types"].values
    return y


# function of the loaded model


def load_model(run_id):

    logged_model = f"s3://mlops-remote-bucket/1/{RUN_ID}/artifacts/model-SVC"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


# Evaluation function of auc_score for individual target


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):

    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:

        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using scikit-learn method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


# function to apply the model


def apply_model(input_data, run_id):

    df = get_data(input_data)

    df_full_train, df_train, df_val, df_test = split_dataset(df)

    dicts = prepare_dictionaries(df)

    train_dicts = prepare_dictionaries(df_train)
    val_dicts = prepare_dictionaries(df_val)
    test_dicts = prepare_dictionaries(df_test)

    y = get_y(df)

    y_train = get_y(df_train)
    y_val = get_y(df_val)
    y_test = get_y(df_test)

    del df_train["mine_types"]
    del df_val["mine_types"]
    del df_test["mine_types"]

    model = load_model(run_id)
    y_pred = model.predict(test_dicts)

    roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)

    return y_pred, roc_auc_dict


# function to run the model pipeline


def run():

    input_data = "Mine_Dataset.csv"
    run_id = RUN_ID

    apply_model(input_data, run_id=RUN_ID)


if __name__ == "__main__":
    run()
