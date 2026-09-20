import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# create mock dataset
# age, salary and purchased
data = np.array([[23, 2345, 0],
                  [24, 256778, 0],
                  [25, 234667, 0],
                  [26, 243574, 1],
                  [27, 267447, 1]])

x = data[:, :-1]   # features: age and salary
y = data[:, -1]    # labels: purchased 0 or 1

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train an svm model
svm_model = SVC(kernel="linear")
svm_model.fit(x_train, y_train)

# test the model
y_pred = svm_model.predict(x_test)

# evaluate the model
print(f"confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# create a grid of points to plot the decision boundary
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

# predict over every point in the grid
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# plot the actual training points on top
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', label='Train')

# plot the testing points too, marked differently
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', s=100, label='Test')

plt.xlabel('Age (scaled)')
plt.ylabel('Salary (scaled)')
plt.title('SVM Decision Boundary')
plt.legend()
plt.show()
