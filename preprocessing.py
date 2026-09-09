import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['SepalLength', 'SepalWidth', 'PetalLength',
           'PetalWidth', 'Species']
df = pd.read_csv(url, names=columns)
print("First Five Records")
print(df.head())
print("\nDataset Shape")
print(df.shape)
print("\nDataset Information")
print(df.info())
print("\nStatistical Summary")
print(df.describe())
print("\nMissing Values")
print(df.isnull().sum())
df.loc[5, 'SepalLength'] = np.nan
df.loc[20, 'PetalWidth'] = np.nan
print("\nMissing Values After Introducing")
print(df.isnull().sum())
df['SepalLength'] = df['SepalLength'].fillna(df['SepalLength'].mean())
df['PetalWidth'] = df['PetalWidth'].fillna(df['PetalWidth'].mean())
print("\nMissing Values After Filling")
print(df.isnull().sum())
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])
print("\nEncoded Dataset")
print(df.head())
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nFirst Five Rows of Scaled Training Data")
print(X_train[:5])
print("\nData Preprocessing Completed Successfully.")
