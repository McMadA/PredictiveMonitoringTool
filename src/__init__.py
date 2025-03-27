import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Data inladen en voorbereiden
df = pd.read_csv('../data/datasheet.csv', converters={'FSO_T00_SKPAuto_Opstarten': lambda x: float(x.replace(',', '.')) if x != '-' else None})
df.dropna(subset=['FSO_T00_SKPAuto_Opstarten'], inplace=True)

# Selecteer alleen de kolommen die nodig zijn
df = df.iloc[:, :2]

# Datumconversie
df['time'] = pd.to_datetime(df['time'], format='%d-%m %H:%M', errors='coerce')
current_year = pd.Timestamp.now().year
df['time'] = df['time'].apply(lambda x: x.replace(year=current_year) if not pd.isna(x) else x)

# Creëer een target kolom die aangeeft of de waarde boven de 8 seconden is
df['target'] = (df['FSO_T00_SKPAuto_Opstarten'] > 8).astype(int)

# Visualisatie van de laadtijd met target classificatie
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['FSO_T00_SKPAuto_Opstarten'], label='Laadtijd')
plt.axhline(y=8, color='r', linestyle='--', label='Drempelwaarde (8 sec)')
plt.scatter(df['time'], df['FSO_T00_SKPAuto_Opstarten'], c=df['target'], cmap='coolwarm', alpha=0.6)
plt.title('Laadtijd met Target Classificatie')
plt.xlabel('Datum en Tijd')
plt.ylabel('Laadtijd (seconden)')
plt.legend()
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m %H:%M'))
plt.show()

# Feature engineering: creëer een window van voorgaande waarden
window_size = 10  # Dit kan worden aangepast op basis van domeinkennis

# Creëer features op basis van voorgaande waarden
X = []
y = []

for i in range(window_size, len(df)):
    # Neem de laatste window_size waarden als features
    features = df['FSO_T00_SKPAuto_Opstarten'].iloc[i-window_size:i].values
    
    # Voeg statistische features toe
    mean_val = np.mean(features)
    std_val = np.std(features)
    trend = np.polyfit(range(window_size), features, 1)[0]  # Helling van de lineaire trend
    
    # Combineer alle features
    all_features = np.concatenate([features, [mean_val, std_val, trend]])
    
    X.append(all_features)
    y.append(df['target'].iloc[i])

X = np.array(X)
y = np.array(y)



# Splits de data in training en test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schaal de features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train verschillende modellen
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100),
    'SVM': SVC(probability=True, class_weight='balanced')
}

results = {}

for name, model in models.items():
    # Train het model
    model.fit(X_train_scaled, y_train)
    
    # Maak voorspellingen
    y_pred = model.predict(X_test_scaled)
    
    # Evalueer het model
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

# Kies het beste model op basis van F1-score (balans tussen precision en recall)
best_model_name = max(results, key=lambda x: results[x]['f1'])
print(f"\nBest model: {best_model_name} with F1 Score: {results[best_model_name]['f1']:.4f}")

# Hyperparameter tuning voor het beste model
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

# Grid search voor optimale hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evalueer het getunede model
y_pred = best_model.predict(X_test_scaled)
print("\nTuned Model Performance:")
print(classification_report(y_test, y_pred))


# Functie om voorspellingen te maken voor nieuwe data
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
    
    # Gebruik de laatste window_size waarden
    recent_data = new_data[-window_size:]
    
    # Bereken extra features
    mean_val = np.mean(recent_data)
    std_val = np.std(recent_data)
    trend = np.polyfit(range(window_size), recent_data, 1)[0]
    
    # Combineer features
    features = np.concatenate([recent_data, [mean_val, std_val, trend]])
    features = features.reshape(1, -1)
    
    # Schaal de features
    scaled_features = scaler.transform(features)
    
    # Maak voorspelling
    probability = model.predict_proba(scaled_features)[0, 1]
    prediction = model.predict(scaled_features)[0]
    
    return probability, prediction == 1

# Voorbeeld van gebruik
recent_load_times = df['FSO_T00_SKPAuto_Opstarten'].iloc[-window_size:].values
prob, will_breach = predict_threshold_breach(best_model, scaler, recent_load_times)
print(f"\nKans dat de volgende laadtijd boven 8 seconden komt: {prob:.2f}")
print(f"Voorspelling: {'Boven' if will_breach else 'Onder'} de drempel van 8 seconden")

# Analyseer welke features het meest bijdragen aan de voorspelling
if hasattr(best_model, 'feature_importances_'):
    # Voor Random Forest
    importances = best_model.feature_importances_
    feature_names = [f"t-{window_size-i}" for i in range(window_size)] + ['mean', 'std', 'trend']
    
    # Sorteer features op belangrijkheid
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
