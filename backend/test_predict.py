import requests

# Test simple para el endpoint /predict
url = 'http://127.0.0.1:5000/predict'

sample = {
    "age": 50,
    "gender": "Male",
    "bmi": 25.0,
    "alcohol_consumption": "Occasional",
    "smoking_status": "Never",
    "hepatitis_b": 0,
    "hepatitis_c": 0,
    "liver_function_score": 60.0,
    "alpha_fetoprotein_level": 5.0,
    "cirrhosis_history": 0,
    "family_history_cancer": 0,
    "physical_activity_level": "Moderate",
    "diabetes": 0
}

resp = requests.post(url, json=sample)
print('status', resp.status_code)
print(resp.json())
