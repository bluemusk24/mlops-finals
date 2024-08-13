import pickle

from flask import Flask, jsonify, request

with open("model_SVC.bin", "rb") as f_in:
    (dv, model_SVC) = pickle.load(f_in)


def predict(landmine):
    X = dv.transform(landmine)
    preds = model_SVC.predict(X)
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
