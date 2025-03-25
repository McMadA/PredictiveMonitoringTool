import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('..../data/datasheet.csv', converters={'FSO_T00_SKPAuto_Opstarten': lambda x: float(x.replace(',', '.')) if x != '-' else None})
df.dropna(subset=['FSO_T00_SKPAuto_Opstarten'], inplace=True)
# df.set_index('time', inplace=True)
df = df.iloc[:, :2]
df
column_data = df['FSO_T00_SKPAuto_Opstarten'].values
rows = []
for i in range (len(column_data) - 9):
    row = column_data[i:i+10]
    rows.append(row)
    
result_df = pd.DataFrame(rows)
result_df

X = result_df.iloc[:, :-1].values
y = result_df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print 3 predictions with the corresponding actual values
for i in range(3):
    print(f"Predicted: {y_pred[i]} Actual: {y_test[i]}")

