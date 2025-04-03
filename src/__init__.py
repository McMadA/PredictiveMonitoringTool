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

# Define converters to handle specific formatting in the CSV
converters = {col: lambda x: float(x.replace(',', '.')) if x != '-' else None for col in pd.read_csv('../data/datasheet.csv').columns if col != 'time'}

# Read the CSV file with converters applied
df = pd.read_csv('../data/datasheet.csv', converters=converters)

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
print(df.head())

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
        day_of_week = window_time.weekday()  # Optional: add day of the week as a feature
        weekday_mapping = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        day_of_week_str = weekday_mapping[day_of_week]
        
        # Combine the features into a single array  
        combined_features = np.concatenate([window, [mean_val, std_val, trend, hour_of_day, day_of_week_str]])
        
        X.append(combined_features)
        y.append(df[target_col].iloc[i])
        window_times.append(window_time)  # Store the timestamp

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)
window_times = np.array(window_times)

# Visualize the data with datetime on x-axis
plt.figure(figsize=(50, 8))

# Plot each column against the time
for col in df.columns:
    if col != 'time' and not col.endswith('_target'):
        plt.plot(df['time'], df[col], label=col)

# Add threshold line
plt.axhline(y=8, color='r', linestyle='--', label='Threshold (8 sec)')

# Format the x-axis to show dates properly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

plt.legend()
plt.xlabel('Date and Time')
plt.ylabel('Duration (seconds)')
plt.title('Time Series of All Columns')
plt.grid(True)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models and evaluate their performance
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100),
    'SVM': SVC(probability=True, class_weight='balanced')
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
def predict_threshold_breach(model, scaler, new_data, window_size=10):
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
    
    # Combine features
    features = np.concatenate([recent_data, [mean_val, std_val, trend]])
    features = features.reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    probability = model.predict_proba(scaled_features)[0, 1]
    prediction = model.predict(scaled_features)[0]
    
    return probability, prediction == 1

# Example usage of the prediction function
recent_load_times = df['FSO_T00_SKPAuto_Opstarten'].iloc[-window_size:].values
prob, will_breach = predict_threshold_breach(best_model, scaler, recent_load_times)
print(f"\nKans dat de volgende laadtijd boven 8 seconden komt: {prob:.2f}")
print(f"Voorspelling: {'Boven' if will_breach else 'Onder'} de drempel van 8 seconden")

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
