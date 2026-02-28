
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

# Load Dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Check Class Distribution
print("Class Distribution:")
print(df['Class'].value_counts())

# Handling Imbalanced Data (Downsampling)
fraud = df[df.Class == 1]
normal = df[df.Class == 0]

normal_downsampled = resample(normal,
                               replace=False,
                               n_samples=len(fraud),
                               random_state=42)

balanced_df = pd.concat([fraud, normal_downsampled])

# Split Data
X = balanced_df.drop("Class", axis=1)
y = balanced_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
