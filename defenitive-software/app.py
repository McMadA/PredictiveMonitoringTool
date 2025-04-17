from flask import Flask, request, render_template, make_response, send_file
import pandas as pd
import numpy as np
import io
import pickle
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

try:
    best_model = pickle.load(open('defenitive-software\\best_model.pkl', 'rb'))
    scaler = pickle.load(open('defenitive-software\\scaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    # Define a flag to check if models are loaded
    models_loaded = False
else:
    models_loaded = True

# Define window size as in your original code
window_size = 10

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return "Model files not loaded. Please check the server logs."
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        df = pd.read_csv(file, encoding='utf-8', skiprows=2)
        file.seek(0) # Reset file pointer to the beginning
        # Gemeten tijden onder elkaar zetten met transactie als kolomnaam
        df.columns = df.iloc[0] # Stel eerste rij in als header
        df = df[1:].reset_index(drop=True) # Verwijder eerste rij uit data
        df = df.T
        df.columns = df.iloc[0] # Stel eerste rij in als header
        df = df[1:].reset_index(drop=True) # Verwijder eerste rij uit data

        # Tijden onder elkaar zetten
        times = pd.read_csv(file, skiprows=2)
        times.iloc[0] = times.columns
        first_column = times.iloc[0, :]
        first_column = first_column.loc[~first_column.str.contains('Unnamed')]
        first_column = first_column.dropna().reset_index(drop=True)

        # Tijden toevoegen aan eerste dataframe
        df.insert(0, 'time', first_column)
        df.set_index('time', inplace=True)
        df = df.dropna(how="all")

        for col in df.columns:
            if col != 'time':
                df[col] = df[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) and x != '-' else None)

        # Drop FSO_T19_SKPAuto_AanvullendeVragen omdat deze afwijkende treshold heeft
        if 'FSO_T19_SKPAuto_AanvullendeVragen' in df.columns:
            df.drop(columns=['FSO_T19_SKPAuto_AanvullendeVragen'], inplace=True)

        # Replace missing values with the previous valid value
        df.fillna(method='ffill', inplace=True)

        # Convert 'time' column to datetime format
        df = df.reset_index()
        df['time'] = pd.to_datetime(df['time'], format='%d-%m %H:%M', errors='coerce')
        current_year = pd.Timestamp.now().year
        df['time'] = df['time'].apply(lambda x: x.replace(year=current_year) if not pd.isna(x) else x)
        df.set_index('time', inplace=True)

        # Create a dictionary to store results
        results = []

        # Make predictions for each column
        for col in df.columns:
            if col != 'time' and not col.endswith('_target'):
                # Only process if we have enough data points
                if len(df[col]) >= window_size:
                    # Get the last window of data
                    window = df[col].iloc[-window_size:].values

                    # Calculate additional features
                    mean_val = np.mean(window)
                    std_val = np.std(window)
                    trend = np.polyfit(range(window_size), window, 1)[0]

                    # Add time-based features
                    window_time = df.index[-1]
                    hour_of_day = window_time.hour
                    day_of_week = window_time.weekday()

                    # Combine features
                    features = np.concatenate([window, [mean_val, std_val, trend, hour_of_day, day_of_week]])
                    features = features.reshape(1, -1)

                    # Scale features
                    scaled_features = scaler.transform(features)

                    # Make prediction
                    probability = best_model.predict_proba(scaled_features)[0, 1]
                    prediction = best_model.predict(scaled_features)[0]
                    
                    # Determine risk level
                    if probability < 0.3:
                        risk_level = "low"
                    elif probability < 0.7:
                        risk_level = "medium"
                    else:
                        risk_level = "high"

                    #Latest value of each column
                    latest_value = df[col].iloc[-1]
                    # Add the latest value to the results


                    # Store in results list
                    results.append({
                        'transaction': col,
                        'latest_value': latest_value,
                        'probability': probability,
                        'prediction': int(prediction),
                        'risk_level': risk_level
                    })
        
        # Sort results by probability (highest first)
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Get timestamp for the report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Render the results template
        return render_template('results.html', results=results, timestamp=timestamp)


if __name__ == '__main__':
    app.run(debug=True)
