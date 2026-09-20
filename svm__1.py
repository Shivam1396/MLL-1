import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load dataset
b_cancer = load_breast_cancer()
x = b_cancer.data
y = b_cancer.target

# select only 2 features (columns 2 and 3) so we can visualize in 2D
x_visual = x[:, [2, 3]]

# split into train/test, preserving class balance
x_train, x_test, y_train, y_test = train_test_split(
    x_visual,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # fit + transform on train
x_test = scaler.transform(x_test)         # transform ONLY on test

# train SVM
model = SVC(kernel="linear")
model.fit(x_train, y_train)

# predict
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("SVM classification result")
print("_____________________________")
print(f"accuracy: {accuracy*100:.2f}%")
print("confusion_matrix")
print(confusion_matrix(y_test, y_pred))

print("classification_report")
print(classification_report(y_test, y_pred, target_names=b_cancer.target_names))

# ---- visualize the decision boundary ----

# combine train+test (already scaled) just for plotting min/max ranges
x_all = np.vstack((x_train, x_test))
y_all = np.hstack((y_train, y_test))

# create a mesh grid covering the full feature space
x_min, x_max = x_all[:, 0].min() - 1, x_all[:, 0].max() + 1
y_min, y_max = x_all[:, 1].min() - 1, x_all[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

# predict over every point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# plot training points
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm,
            edgecolors='k', label='Train')

# plot testing points
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.coolwarm,
            marker='x', s=100, label='Test')

plt.xlabel('Mean Perimeter (scaled)')
plt.ylabel('Mean Area (scaled)')
plt.title('SVM Decision Boundary - Breast Cancer Dataset')
plt.legend()
plt.tight_layout()
plt.show()
