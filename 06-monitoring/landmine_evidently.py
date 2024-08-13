import datetime
import io
import logging
import random
import time
import uuid

import joblib
import pandas as pd
import psycopg
import pytz
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

# from prefect import task, flow


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists landmine_metrics;
create table landmine_metrics(
	timestamp timestamp,
	prediction_drift float,
	number_of_drifted_columns integer,
	share_of_missing_values float
)
"""


# Data preparation and transformation
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


df = get_data("Mine_Dataset.csv")


# Splitting dataset function
def split_dataset(df):

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_full_train, df_train, df_val, df_test


df_full_train, df_train, df_val, df_test = split_dataset(df)


# Get the target values
def get_y(df):
    y = df["mine_types"].values
    return y


y_train = get_y(df_train)
y_val = get_y(df_val)
y_test = get_y(df_test)

del df_train["mine_types"]
del df_val["mine_types"]
del df_test["mine_types"]


# Prepare dictionaries function
def prepare_dictionaries(df):

    num_variables = ["voltage", "height"]
    cat_variable = ["soil_types"]

    # Ensure categorical column is of type str
    df[cat_variable] = df[cat_variable].astype(str)

    dicts = df[cat_variable + num_variables].to_dict(orient="records")

    return dicts


train_dict = prepare_dictionaries(df_train)
val_dict = prepare_dictionaries(df_val)
test_dict = prepare_dictionaries(df_test)


# Applying Dictvectorizer
dv = DictVectorizer(sparse=False)

train_data = dv.fit_transform(train_dict)
val_data = dv.transform(val_dict)
test_data = dv.transform(test_dict)


# Get the model
with open("model_svc.bin", "rb") as f_in:
    model = joblib.load(f_in)


begin = datetime.datetime.now()  # datetime.datetime.(2022, 2, 1, 0, 0)
num_variables = ["voltage", "height"]
cat_variable = ["soil_types"]
# Ensure all elements are strings
cat_variable = [str(item) for item in cat_variable]

column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=num_variables,
    categorical_features=cat_variable,
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


# Prepare Database
def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)


# Calculate metrics in the database
def calculate_metrics_postgresql(curr, i):  # (curr):

    # Prepare train data as DataFrame for Evidently
    reference_data = pd.DataFrame(train_data, columns=dv.get_feature_names_out())
    reference_data["soil_types"] = (
        df_train["soil_types"].astype(str).reset_index(drop=True)
    )
    train_preds = model.predict(train_data)
    reference_data["prediction"] = train_preds

    # Prepare validation data
    val_dict = prepare_dictionaries(df_val)
    val_data = dv.transform(val_dict)

    current_data = pd.DataFrame(val_data, columns=dv.get_feature_names_out())
    current_data["soil_types"] = df_val["soil_types"].astype(str).reset_index(drop=True)

    val_preds = model.predict(val_data)
    current_data["prediction"] = val_preds

    # Run Evidently metrics
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        additional_data=test_dict,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    # Extract metrics
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    number_of_drifted_columns = result["metrics"][1]["result"][
        "number_of_drifted_columns"
    ]
    share_of_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]

    # Insert metrics into the database
    curr.execute(
        "insert into landmine_metrics(timestamp, prediction_drift, number_of_drifted_columns, share_of_missing_values) values (%s, %s, %s, %s)",
        (
            begin + datetime.timedelta(i),
            prediction_drift,
            number_of_drifted_columns,
            share_of_missing_values,
        ),
    )


# Batch processing function
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for i in range(0, 15):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()
