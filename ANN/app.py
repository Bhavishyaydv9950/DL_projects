from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

import joblib

app = Flask(__name__)

# Load the trained Keras model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        kms_driven = float(request.form.get('kms_driven', 0))
        owner = int(request.form.get('owner', 0))
        age = int(request.form.get('age', 0))
        power = float(request.form.get('power', 0))
        brand = int(request.form.get('brand', 0))

        # Format input for model
        input_data = np.array([[kms_driven, owner, age, power, brand]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0][0]
        price = round(prediction, 2)

        return render_template('result.html', prediction=price)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
