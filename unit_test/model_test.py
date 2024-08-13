import predict


def test_features():

    landmine = {"voltage": 0.341389, "height": 0.818182, "soil_types": 0.4}

    actual_features = predict.predict(landmine)

    expected_features = 3  # flask app and predict-test.py outcome

    assert actual_features == expected_features
