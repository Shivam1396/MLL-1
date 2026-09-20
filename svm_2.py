import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load dataset
b_cancer = load_breast_cancer()
x = b_cancer.data
y = b_cancer.target

# split into train/test using ALL 30 features (much more accurate)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42, stratify=y
)

# standardize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train SVM on all 30 features
model = SVC(kernel="linear")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("SVM classification result (all 30 features)")
print("_____________________________")
print(f"accuracy: {accuracy*100:.2f}%")
print("confusion_matrix")
print(confusion_matrix(y_test, y_pred))
print("classification_report")
print(classification_report(y_test, y_pred, target_names=b_cancer.target_names))

# ---- visualize using PCA (compress 30 features down to 2 for plotting only) ----
# Note: this is JUST for visualization. The model above was trained on all 30 features.
pca = PCA(n_components=2)
x_train_2d = pca.fit_transform(x_train)
x_test_2d = pca.transform(x_test)

# train a SEPARATE model on the 2D PCA-reduced data, only so we can draw a boundary
vis_model = SVC(kernel="linear")
vis_model.fit(x_train_2d, y_train)

x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.05),
    np.arange(y_min, y_max, 0.05)
)
Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(x_train_2d[:, 0], x_train_2d[:, 1], c=y_train, cmap=plt.cm.coolwarm,
            edgecolors='k', label='Train')
plt.scatter(x_test_2d[:, 0], x_test_2d[:, 1], c=y_test, cmap=plt.cm.coolwarm,
            marker='x', s=100, label='Test')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary (PCA-reduced view, trained on all 30 features)')
plt.legend()
plt.tight_layout()
plt.show()
