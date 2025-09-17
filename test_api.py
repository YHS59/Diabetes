import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Pregnancies": 2,
    "Glucose": 160,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 10,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 1.2,
    "Age": 50
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
