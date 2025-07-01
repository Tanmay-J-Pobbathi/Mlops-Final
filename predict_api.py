from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

app = Flask(__name__)

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

# 🔹 ADD THIS ROUTE FOR ROOT "/"
@app.route('/')
def home():
    return "Welcome to the Iris Prediction API! Use POST /predict with flower features."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)