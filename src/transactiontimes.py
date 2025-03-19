import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/datasheet.csv')
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

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape) 


