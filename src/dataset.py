from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# import pandas 
import pandas as pd
# load data
df = pd.read_csv('../data/raw/report.csv', skiprows=2)
# invert columns and rows
df = df.T
# remove first unnecessary row
df = df.drop(columns=[0])
# remove empty cells from first column

counting_row = pd.DataFrame([range(1, len(df.columns) + 1)], columns=df.columns)

# Add it to the top of the DataFrame
df = pd.concat([counting_row, df], ignore_index=True)
first_column = df.iloc[0, :]










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
