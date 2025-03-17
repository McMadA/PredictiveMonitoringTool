import pandas as pd

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

