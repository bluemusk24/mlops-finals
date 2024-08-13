import json

import requests
from deepdiff import DeepDiff

sensor = {"voltage": 0.5867, "height": 0.3647, "soil_types": 0.4792}

url = "http://0.0.0.0:9696/predict"

response = requests.post(url, json=sensor)

actual_response = print(response.json())

# print actual_response for nicer look
print("actual_response:")

print(json.dumps(actual_response, indent=2))

expected_response = {"landmine_type": 1.0}

# assert actual_response == expected_response

diff = DeepDiff(actual_response, expected_response)
print(f"diff= {diff}")

# assert 'type_changes' not in diff
