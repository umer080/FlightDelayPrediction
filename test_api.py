import requests

data = {
    "time": "2022-11-28 01:00",
    "origin": "DTW",
    "dest": "SYR",
    "airline": "WN",
}

TEST_URL = "http://52.199.26.152:5000/predict"
# TEST_URL = "http://127.0.0.1:5000/predict"

r = requests.post(TEST_URL, json=data)
print(r.json())
