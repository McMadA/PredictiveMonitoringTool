import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.dates as mdates
from xgboost import XGBClassifier

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
window_size = 10

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

# Visualize the data with datetime on x-axis
plt.figure(figsize=(50, 8))
for col in df.columns:
    if col != 'time' and not col.endswith('_target'):
        plt.plot(df['time'], df[col], label=col)
plt.axhline(y=8, color='r', linestyle='--', label='Threshold (8 sec)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
plt.legend()
plt.xlabel('Date and Time')
plt.ylabel('Duration (seconds)')
plt.title('Time Series of All Columns')
plt.grid(True)
plt.show()

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
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Choose the best model based on F1 score
best_model_name = max(results, key=lambda x: results[x]['f1'])
print(f"\nBest model: {best_model_name} with F1 Score: {results[best_model_name]['f1']:.4f}")

# Hyperparameter tuning for the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(class_weight='balanced')
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.2, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2']
    }
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
else:  # SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear', 'poly']
    }
    model = SVC(probability=True, class_weight='balanced')

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evaluate the tuned model
y_pred = best_model.predict(X_test_scaled)
print("\nTuned Model Performance:")
print(classification_report(y_test, y_pred))


# Function to predict if the next load time will breach the threshold of 8 seconds
def predict_threshold_breach(model, scaler, new_data, datetime, window_size=10):
    """
    Voorspel of de laadtijd boven de 8 seconden zal komen op basis van recente metingen.
    
    Parameters:
    - model: Het getrainde model
    - scaler: De feature scaler
    - new_data: Array van recente laadtijden (minimaal window_size elementen)
    - window_size: Aantal voorgaande waarden om te gebruiken voor de voorspelling
    
    Returns:
    - Waarschijnlijkheid dat de volgende waarde boven de 8 seconden komt
    """
    if len(new_data) < window_size:
        raise ValueError(f"Niet genoeg data punten. Minimaal {window_size} nodig.")
    
    # Use the latest window_size data points
    recent_data = new_data[-window_size:]

    # Calculate additional features
    mean_val = np.mean(recent_data)
    std_val = np.std(recent_data)
    trend = np.polyfit(range(window_size), recent_data, 1)[0]
    
    # Bereken alle gebruikte features
    hour_of_day = datetime.now().hour
    day_of_week = datetime.now().weekday()

    features = np.concatenate([recent_data, [mean_val, std_val, trend, hour_of_day, day_of_week]])
    features = features.reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    probability = model.predict_proba(scaled_features)[0, 1]
    prediction = model.predict(scaled_features)[0]
    
    return probability, prediction == 1

#Functie die de nieuwe csv importeert, goed convert en per kolom de voorspelling doet
new_data = pd.read_csv('../data/datasheet_new.csv', converters=converters)
new_data.drop(columns=['FSO_T19_SKPAuto_AanvullendeVragen'], inplace=True)
new_data.fillna(method='ffill', inplace=True)
new_data['time'] = pd.to_datetime(new_data['time'], format='%d-%m %H:%M', errors='coerce')
new_data['time'] = new_data['time'].apply(lambda x: x.replace(year=current_year) if not pd.isna(x) else x)
for col in df.columns:
    if col != 'time':
        new_data[f'{col}_target'] = (df[col] > 8).astype(int)
X_new = []
y_new = []
window_times_new = []  
feature_names_new = []  
for col in new_data.columns:
    if col == 'time' or '_target' in col:
        continue
        
    target_col_new = f'{col}_target'
    
# Convert lists to numpy arrays
X_new = np.array(X)
y_new = np.array(y)
window_times_new = np.array(window_times_new)

for col in new_data.columns:
    if col != 'time' and not col.endswith('_target'):
        recent_load_times = new_data[col].iloc[-window_size:].values
        datetime = new_data['time'].iloc[-1]  # Get the last timestamp
        prob, will_breach = predict_threshold_breach(best_model, scaler, recent_load_times, datetime)
        print(f"\nKans dat de volgende laadtijd van {col} boven 8 seconden komt: {prob:.2f}")
  
# Analyse the feature importance for Random Forest
if hasattr(best_model, 'feature_importances_'):
    # For Random Forest
    importances = best_model.feature_importances_
    feature_names = [f"t-{window_size-i}" for i in range(window_size)] + ['mean', 'std', 'trend']
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    print("\nTop 5 belangrijkste features:")
    for i in range(5):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
