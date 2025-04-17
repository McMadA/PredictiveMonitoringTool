import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

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
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Functie om voorspellingen te maken voor nieuwe data
def predict_above_8(time_index):
    return pipeline.predict(np.array(time_index).reshape(-1, 1))

# Voorbeeld gebruik
print("\nVoorspelling voor tijdsindexen 10, 20, 30:")
print(predict_above_8([10, 20, 30]))
