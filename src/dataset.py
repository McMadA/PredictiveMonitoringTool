from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import pandas as pd
df = pd.read_csv('../data/raw/report.csv', skiprows=2)

#gemeten tijden onder elkaar zetten met transactie als 
df.columns = df.iloc[0]  # Set first row as header
df = df[1:].reset_index(drop=True)  # Remove the first row from data
df = df.T
df.columns = df.iloc[0]  # Set first row as header
df = df[1:].reset_index(drop=True)  # Remove the first row from data


#tijden onder elkaar zetten
times = pd.read_csv('../data/raw/report.csv', skiprows=2)
times.iloc[0] = times.columns
times.iloc[0] = times.columns
first_column = times.iloc[0, :] 
first_column = first_column.loc[~first_column.str.contains('Unnamed')]
# first_column = first_column.melt(var_name='time slot', value_name='time')
first_column = first_column.dropna().reset_index(drop=True)


#tijden toevoegen aan eerste dataframe
df.insert(0, 'time', first_column)
df.set_index('time', inplace=True)

df = df.dropna(how="all")

df.to_csv('../data/processed/report.csv')






















# Handle missing values
# Assuming numerical columns are filled with the mean and categorical columns with the most frequent value
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into features and target
X = df.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
y = df['target_column']  # Replace 'target_column' with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the transformations to the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Now X_train and X_test are ready for machine learning
