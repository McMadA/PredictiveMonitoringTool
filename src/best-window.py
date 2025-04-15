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
# Initialize dictionary to store results for each window size
window_size_results = {}

# Loop through window sizes from 5 to 50 with step size 5
for window_size in range(5, 55, 5):
    print(f"\nEvaluating models with window size: {window_size}")
    
    # Initialize empty lists for features and target variables
    X = []
    y = []
    
    # Loop through each column in the dataframe
    for col in df.columns:
        if col == 'time' or '_target' in col:
            continue
            
        target_col = f'{col}_target'
        
        # Loop through the dataframe starting from the window size
        for i in range(window_size, len(df)):
            window = df[col].iloc[i-window_size:i].values
            window_time = df['time'].iloc[i]
            
            # Calculate additional features
            mean_val = np.mean(window)
            std_val = np.std(window)
            trend = np.polyfit(range(window_size), window, 1)[0]  # Extract the slope
            
            # Add time-based features
            hour_of_day = window_time.hour
            day_of_week = window_time.weekday()
            
            # Combine the features into a single array  
            combined_features = np.concatenate([window, [mean_val, std_val, trend, hour_of_day, day_of_week]])
            
            X.append(combined_features)
            y.append(df[target_col].iloc[i])
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
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
        # Train the model
        model.fit(X_train_scaled, y_train)
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        # Evaluate the model
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        results[name] = {
            'f1': f1
        }
        
        print(f"{name} - F1 Score: {f1:.4f}")
    
    # Choose the best model for this window size based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_f1_score = results[best_model_name]['f1']
    
    # Store the results for this window size
    window_size_results[window_size] = {
        'best_model': best_model_name,
        'f1_score': best_f1_score
    }
    
    print(f"Best model for window size {window_size}: {best_model_name} with F1 Score: {best_f1_score:.4f}")

# Find the best window size based on F1 score
best_window_size = max(window_size_results, key=lambda x: window_size_results[x]['f1_score'])
best_model = window_size_results[best_window_size]['best_model']
best_f1 = window_size_results[best_window_size]['f1_score']

print("\n=== Final Results ===")
print(f"Best window size: {best_window_size}")
print(f"Best model: {best_model}")
print(f"Best F1 Score: {best_f1:.4f}")

# Print all results for comparison
print("\n=== All Results ===")
for window_size in sorted(window_size_results.keys()):
    result = window_size_results[window_size]
    print(f"Window Size: {window_size}, Best Model: {result['best_model']}, F1 Score: {result['f1_score']:.4f}")