from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Correct Predictions:")
print("--------------------")

correct = 0
wrong = 0

for i in range(len(y_test)):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]

    if y_test[i] == y_pred[i]:
        print("Actual:", actual, "Predicted:", predicted, "-> Correct")
        correct += 1

print("\nWrong Predictions:")
print("-----------------")

for i in range(len(y_test)):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]

    if y_test[i] != y_pred[i]:
        print("Actual:", actual, "Predicted:", predicted, "-> Wrong")
        wrong += 1

print("\nTotal Correct Predictions:", correct)
print("Total Wrong Predictions:", wrong)
print("Total Test Samples:", len(y_test))
