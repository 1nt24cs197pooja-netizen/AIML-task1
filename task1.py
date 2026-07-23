import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop("Cabin", axis=1)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
# Convert 'Sex' column into numerical values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert 'Embarked' column into numerical values
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Display first 5 rows after encoding
print("\nDataset after Encoding:")
print(df.head())
from sklearn.preprocessing import StandardScaler

# Create StandardScaler object
scaler = StandardScaler()

# Select numerical columns
numerical_columns = ["Age", "Fare", "SibSp", "Parch"]

# Apply Standardization
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display first 5 rows
print("\nDataset after Standardization:")
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# Draw boxplots for numerical columns
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["Age", "Fare", "SibSp", "Parch"]])

plt.title("Boxplot for Outlier Detection")
plt.show()