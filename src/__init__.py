import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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

# Initialate empty lists for features and target variables
X = []
y = []

# Loop through each column in the dataframe
for col in df.columns:
    if col == 'time' or '_target' in col:
        continue
        
    target_col = f'{col}_target'
    
    # Loop trough the dataframe starting from the window size
    for i in range(window_size, len(df)):
        window = df[col].iloc[i-window_size:i].values
        
        # Calculate additional features
        mean_val = np.mean(window)
        std_val = np.std(window)
        trend = np.polyfit(range(window_size), window, 1)[0]
        
        # Combine the features into a single array  
        combined_features = np.concatenate([window, [mean_val, std_val, trend]])
        
        X.append(combined_features)
        y.append(df[target_col].iloc[i])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

#visualize the data
plt.figure(figsize=(50, 8))
plt.plot(df['FSO_T00_SKPAuto_Opstarten'].values, label='FSO_T00_SKPAuto_Opstarten')  # Laadtijd
plt.plot(df['FSO_T01_SKPAuto_VerzekererdJaNee'].values, label='FSO_T01_SKPAuto_VerzekererdJaNee')
plt.plot(df['FSO_T02_SKPAuto_Kenteken'].values, label='FSO_T02_SKPAuto_Kenteken')
plt.plot(df['FSO_T03_SKPAuto_AutoCheck'].values, label='FSO_T03_SKPAuto_AutoCheck')   
plt.plot(df['FSO_T04_SKPAuto_Postcode'].values, label='FSO_T04_SKPAuto_Postcode')
plt.plot(df['FSO_T05_SKPAuto_Bestuurder'].values, label='FSO_T05_SKPAuto_Bestuurder')
plt.plot(df['FSO_T06_SKPAuto_GeboorteDatum'].values, label='FSO_T06_SKPAuto_GeboorteDatum')
plt.plot(df['FSO_T07_SKPAuto_SchadeVrijeJaren'].values, label='FSO_T07_SKPAuto_SchadeVrijeJaren')
plt.plot(df['FSO_T08_SKPAuto_Kilometers'].values, label='FSO_T08_SKPAuto_Kilometers')
plt.plot(df['FSO_T09_SKPAuto_Basisdekking'].values, label='FSO_T09_SKPAuto_Basisdekking')
plt.plot(df['FSO_T10_SKPAuto_Uitbreiding'].values, label='FSO_T10_SKPAuto_Uitbreiding')
plt.plot(df['FSO_T11_SKPAuto_Ingangsdatum'].values, label='FSO_T11_SKPAuto_Ingangsdatum')
plt.plot(df['FSO_T12_SKPAuto_Ongevallen'].values, label='FSO_T12_SKPAuto_Ongevallen')
plt.plot(df['FSO_T13_SKPAuto_Samenvatting1'].values, label='FSO_T13_SKPAuto_Samenvatting1')
plt.plot(df['FSO_T14_SKPAuto_PriveZakelijk'].values, label='FSO_T14_SKPAuto_PriveZakelijk')
plt.plot(df['FSO_T15_SKPAuto_KentekenOpNaam'].values, label='FSO_T15_SKPAuto_KentekenOpNaam')
plt.plot(df['FSO_T16_SKPAuto_Persoonsgegevens'].values, label='FSO_T16_SKPAuto_Persoonsgegevens')
plt.plot(df['FSO_T17_SKPAuto_Adres'].values, label='FSO_T17_SKPAuto_Adres')
plt.plot(df['FSO_T18_SKPAuto_TelefoonEmail'].values, label='FSO_T18_SKPAuto_TelefoonEmail')
plt.plot(df['FSO_T19_SKPAuto_AanvullendeVragen'].values, label='FSO_T19_SKPAuto_AanvullendeVragen')
plt.plot(df['FSO_T20_SKPAuto_GeweigerdOpgezegd'].values, label='FSO_T20_SKPAuto_GeweigerdOpgezegd')
plt.plot(df['FSO_T21_SKPAuto_StrafbaarFeit'].values, label='FSO_T21_SKPAuto_StrafbaarFeit')
plt.plot(df['FSO_T22_SKPAuto_EerderSchade'].values, label='FSO_T22_SKPAuto_EerderSchade')
plt.plot(df['FSO_T23_SKPAuto_MaandJaar'].values, label='FSO_T23_SKPAuto_MaandJaar')
plt.plot(df['FSO_T24_SKPAuto_Rekeningnummer'].values, label='FSO_T24_SKPAuto_Rekeningnummer')
plt.plot(df['FSO_T25_SKPAuto_Samenvatting2'].values, label='FSO_T25_SKPAuto_Samenvatting2')
plt.plot(df['FSO_T26_SKPAuto_EindeScript'].values, label='FSO_T26_SKPAuto_EindeScript')
plt.axhline(y=8, color='r', linestyle='--', label='Drempelwaarde (8 sec)')
plt.legend()
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
