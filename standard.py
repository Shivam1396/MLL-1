import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Use only Petal Length and Petal Width
X_visual = X[:, [2, 3]]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_visual,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("SVM Classification Results")
print("--------------------------")
print("Accuracy:", accuracy * 100, "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# Combine training and testing data for visualization
X_all = np.vstack((X_train, X_test))
y_all = np.hstack((y_train, y_test))

# Create mesh grid
x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predict each point in the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

# Plot each class
for class_value in np.unique(y):
    plt.scatter(
        X_all[y_all == class_value, 0],
        X_all[y_all == class_value, 1],
        label=iris.target_names[class_value]
    )

plt.xlabel("Petal Length (standardized)")
plt.ylabel("Petal Width (standardized)")
plt.title("SVM Decision Boundaries on Iris Dataset")
plt.legend()
plt.show()
