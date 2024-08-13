import requests

sensor = {"voltage": 0.5867, "height": 0.3647, "soil_types": 0.4792}

# url = "http://0.0.0.0:9696/predict"
url = "http://a8b6fae0108944af5aa148677d8d3a2d-905019242.us-east-1.elb.amazonaws.com/predict"  # load balancer from AWS

response = requests.post(url, json=sensor)
print(response.json())


if response.json()["landmine_type"] == 0:
    print("landmine type is not detected")

elif response.json()["landmine_type"] == 1:
    print("landmine type is Anti-Tank")

elif response.json()["landmine_type"] == 2:
    print("landmine type is Anti-Personnel")

elif response.json()["landmine_type"] == 3:
    print("landmine type is Booby Trapped Anti-Personnel")

elif response.json()["landmine_type"] == 4:
    print("landmine type is M14 Anti-Personnel")

else:
    print("nothing")

# Another test of landmine type

detector = {"voltage": 0.1234, "height": 0.5678, "soil_types": 0.8910}


response = requests.post(url, json=detector)
print(response.json())

if response.json()["landmine_type"] == 0:
    print("landmine type is not detected")

elif response.json()["landmine_type"] == 1:
    print("landmine type is Anti-Tank")

elif response.json()["landmine_type"] == 2:
    print("landmine type is Anti-Personnel")

elif response.json()["landmine_type"] == 3:
    print("landmine type is Booby Trapped Anti-Personnel")

elif response.json()["landmine_type"] == 4:
    print("landmine type is M14 Anti-Personnel")

else:
    print("nothing")
