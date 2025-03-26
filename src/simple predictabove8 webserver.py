import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify, render_template_string
import logging
from io import StringIO

# Configureer logging
log_stream = StringIO()
logging.basicConfig(stream=log_stream, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Laad de data
data = pd.read_csv('../data/datasheet.csv')

# Converteer de relevante kolom naar numeriek en handel komma's af als decimaalpunten
data['FSO_T00_SKPAuto_Opstarten'] = data['FSO_T00_SKPAuto_Opstarten'].str.replace(',', '.')
data['FSO_T00_SKPAuto_Opstarten'] = pd.to_numeric(data['FSO_T00_SKPAuto_Opstarten'], errors='coerce')

# Maak de doelvariabele
data['Target'] = (data['FSO_T00_SKPAuto_Opstarten'] > 8).astype(int)

# Bereid de data voor op modellering
X = np.arange(len(data)).reshape(-1, 1)  # Gebruik tijdsindex als kenmerk
y = data['Target']

# Splits de data in training- en testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Maak een pipeline met scaling en een Random Forest model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Train het model
pipeline.fit(X_train, y_train)

# Maak voorspellingen op de testset
y_pred = pipeline.predict(X_test)

# Evalueer het model
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Functie om voorspellingen te maken voor nieuwe data
def predict_above_8(time_index):
    prediction = pipeline.predict(np.array(time_index).reshape(-1, 1))
    logging.info(f"Prediction for time index {time_index}: {prediction.tolist()}")
    return prediction.tolist()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            time_index = data['time_index']
            prediction = predict_above_8(time_index)
            return jsonify({'prediction': prediction})
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return jsonify({'error': 'An error occurred while processing the request.'})
    else:
        log_output = log_stream.getvalue()
        return render_template_string('''
            <h1>Model Output</h1>
            <pre>{{ log_output }}</pre>
            <h2>Make a Prediction</h2>
            <form id="prediction-form">
                <input type="text" id="time-index" placeholder="Enter time index">
                <button type="submit">Predict</button>
            </form>
            <div id="prediction-result"></div>
            <script>
                document.getElementById('prediction-form').addEventListener('submit', function(e) {
                    e.preventDefault();
                    var timeIndex = document.getElementById('time-index').value;
                    fetch('/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({time_index: [parseInt(timeIndex)]}),
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('prediction-result').innerText = 'Prediction: ' + data.prediction;
                    });
                });
            </script>
        ''', log_output=log_output)

if __name__ == '__main__':
    app.run(debug=True)
