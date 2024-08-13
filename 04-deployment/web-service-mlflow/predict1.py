import os
import pickle

import mlflow
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

# Using the model from mlflow ui with the RUN_ID
RUN_ID = os.getenv("RUN_ID", "2441a3e55caa4e02b962150a04fa47fd")
logged_model = f"runs:/{RUN_ID}/model"

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

# get tracking_uri with MLflowClient and mlflow
mlflow_tracking_uri = "http://127.0.0.1:5000"

# mlflow.set_experiment('my-taxi-duration')
mlflow.set_tracking_uri(mlflow_tracking_uri)

# mlflow.get_experiment(experiment_id='1')
client = MlflowClient(tracking_uri=mlflow_tracking_uri)

# get artifacts from the mlflowclient
path = client.download_artifacts(
    run_id=RUN_ID, path="dict-preprocessor/dict_vectorizer.bin"
)
print(f"downloading the dict-vectorizer to {path}")

with open(path, "rb") as f_out:
    dv = pickle.load(f_out)


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
