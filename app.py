from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
MODEL_PATH = 'model/model_saved.pkl'

# Load the trained model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        data = pd.read_excel(file_path)
        predictions = model.predict(data)
        data['Prediction'] = predictions
        result_path = os.path.join('uploads', 'results.xlsx')
        data.to_excel(result_path, index=False)
        return send_file(result_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
