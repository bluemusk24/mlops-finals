import os
import pickle

import mlflow
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

# Using the model from mlflow ui with the RUN_ID '6f4e767405144ea0abe4bdac7bdcbb43'
RUN_ID = os.getenv("RUN_ID", "6f4e767405144ea0abe4bdac7bdcbb43")

# model from aws s3 bucket
logged_model = f"s3://mlops-remote-bucket/1/{RUN_ID}/artifacts/model-SVC/"

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)


def predict(landmine):
    preds = model.predict(landmine)
    return float(preds[0])


app = Flask("landmine-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():

    landmine = request.get_json()

    pred = predict(landmine)

    result = {"landmine_type": pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
