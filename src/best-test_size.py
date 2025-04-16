import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import matplotlib.dates as mdates
from xgboost import XGBClassifier
import pickle

# Define converters to handle specific formatting in the CSV
converters = {col: lambda x: float(x.replace(',', '.')) if x != '-' else None for col in pd.read_csv('../data/datasheet.csv').columns if col != 'time'}

# Read the CSV file with converters applied
df = pd.read_csv('../data/datasheet.csv', converters=converters)

# Drop FSO_T19_SKPAuto_AanvullendeVragen omdat deze afwijkende treshold heeft
df.drop(columns=['FSO_T19_SKPAuto_AanvullendeVragen'], inplace=True)

# Replace missing values with the previous valid value
df.fillna(method='ffill', inplace=True)

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], format='%d-%m %H:%M', errors='coerce')
current_year = pd.Timestamp.now().year
df['time'] = df['time'].apply(lambda x: x.replace(year=current_year) if not pd.isna(x) else x)

# Create target variables based on the threshold of 8 seconds
for col in df.columns:
    if col != 'time':
        df[f'{col}_target'] = (df[col] > 8).astype(int)

# Print the first few rows of the dataframe to check the data
#print(df.head())

# Feature engineering: create features based on the last 10 values
window_size = 11

# Initialize empty lists for features and target variables
X = []
y = []
window_times = []  # New list to store time information
feature_names = []  # List to store feature names


# Loop through each column in the dataframe
for col in df.columns:
    if col == 'time' or '_target' in col:
        continue
        
    target_col = f'{col}_target'
    
    # Loop through the dataframe starting from the window size
    for i in range(window_size, len(df)):
        window = df[col].iloc[i-window_size:i].values
        window_time = df['time'].iloc[i]  # Get the timestamp for this window
        
        # Calculate additional features
        mean_val = np.mean(window)
        std_val = np.std(window)
        trend = np.polyfit(range(window_size), window, 1)[0]
        
        # Add hour of day as a feature
        hour_of_day = window_time.hour
        day_of_week = window_time.weekday()
        
        # Combine the features into a single array  
        combined_features = np.concatenate([window, [mean_val, std_val, trend, hour_of_day, day_of_week]])
        
        X.append(combined_features)
        y.append(df[target_col].iloc[i])
        window_times.append(window_time)  # Store the timestamp

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)
window_times = np.array(window_times)

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example test sizes to evaluate

for test_size in test_sizes:
    print(f"\n=== Testing with test size: {test_size} ===\n")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train different models and evaluate their performance
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100),
        'SVM': SVC(probability=True, class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Train the model
        model.fit(X_train_scaled, y_train)
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        # Evaluate the model

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        binairyaccuracy =  (TP + TN) / (TP + FP + TN + FN)

        results[name] = {
            'f1': f1
        }
        print(f"{name} - F1 Score: {f1:.4f}")

        

    # Choose the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    print(f"\n>> Best model for test_size={test_size}: {best_model_name} with F1 Score = {results[best_model_name]['f1']:.4f}")
    print("===========================================================\n")
