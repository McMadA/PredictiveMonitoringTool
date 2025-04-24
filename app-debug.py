from flask import Flask, request, render_template, make_response, send_file
import pandas as pd
import numpy as np
import io
import pickle
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

try:
    best_model = pickle.load(open('defenitive-software/best_model.pkl', 'rb'))
    scaler = pickle.load(open('defenitive-software/scaler.pkl', 'rb'))
except Exception as e:
    logging.error(f"Error loading model files: {str(e)}")
    models_loaded = False
else:
    models_loaded = True

window_size = 11

@app.route('/')
def index():
    logging.debug('Request for index page received')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if not models_loaded:
        logging.error('Model files not loaded')
        return "Model files not loaded. Please check the server logs."
    if 'file' not in request.files:
        logging.error('No file part in request')
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        logging.error('No selected file')
        return "No selected file"

    if file:
        try:
            # Start CSV processing
            df = pd.read_csv(file, encoding='utf-8', skiprows=2)
            file.seek(0)
            logging.debug(f"CSV file loaded with {len(df)} rows")

            # Data transformation
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df = df.T
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            logging.debug("Dataframe transformed successfully")

            # Time processing
            times = pd.read_csv(file, skiprows=2)
            times.iloc[0] = times.columns
            first_column = times.iloc[0, :]
            first_column = first_column.loc[~first_column.str.contains('Unnamed')]
            first_column = first_column.dropna().reset_index(drop=True)

            df.insert(0, 'time', first_column)
            df.set_index('time', inplace=True)
            df = df.dropna(how="all")
            logging.debug("Time index processed")

            # Data cleaning
            for col in df.columns:
                if col != 'time':
                    df[col] = df[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) and x != '-' else None)
            
            if 'FSO_T19_SKPAuto_AanvullendeVragen' in df.columns:
                df.drop(columns=['FSO_T19_SKPAuto_AanvullendeVragen'], inplace=True)
            
            df.fillna(method='ffill', inplace=True)
            logging.debug("Data cleaning completed")

            # Datetime conversion
            df = df.reset_index()
            df['time'] = pd.to_datetime(df['time'], format='%d-%m %H:%M', errors='coerce')
            current_year = pd.Timestamp.now().year
            df['time'] = df['time'].apply(lambda x: x.replace(year=current_year) if not pd.isna(x) else x)
            df.set_index('time', inplace=True)
            logging.debug("Datetime conversion completed")

            results = []
            logging.info(f"Processing {len(df.columns)} transactions")

            # Prediction loop
            for col in df.columns:
                if col != 'time' and not col.endswith('_target'):
                    if len(df[col]) >= window_size:
                        window_start = time.time()
                        window = df[col].iloc[-window_size:].values

                        # Feature engineering
                        mean_val = np.mean(window)
                        std_val = np.std(window)
                        trend = np.polyfit(range(window_size), window, 1)[0]
                        window_time = df.index[-1]
                        hour_of_day = window_time.hour
                        day_of_week = window_time.weekday()

                        features = np.concatenate([window, [mean_val, std_val, trend, hour_of_day, day_of_week]])
                        features = features.reshape(1, -1)

                        # Model prediction
                        scaled_features = scaler.transform(features)
                        probability = best_model.predict_proba(scaled_features)[0, 1]
                        prediction = best_model.predict(scaled_features)[0]

                        # Risk classification
                        if probability < 0.3:
                            risk_level = "low"
                        elif probability < 0.7:
                            risk_level = "medium"
                        else:
                            risk_level = "high"

                        latest_value = df[col].iloc[-1]

                        results.append({
                            'transaction': col,
                            'latest_value': latest_value,
                            'probability': probability,
                            'prediction': int(prediction),
                            'risk_level': risk_level
                        })
                        logging.debug(f"Processed {col} in {time.time()-window_start:.4f}s")

            # Sort and return results
            results.sort(key=lambda x: x['probability'], reverse=True)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            total_time = time.time() - start_time
            logging.info(f"Prediction completed in {total_time:.4f} seconds")
            
            return render_template('results.html', 
                                results=results, 
                                timestamp=timestamp,
                                processing_time=f"{total_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Critical error during processing: {str(e)}", exc_info=True)
            return f"System error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
