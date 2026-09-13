import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
print("Dataset Shape:", X.shape)
X = X[:, [2]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully.")
y_pred = model.predict(X_test)
print("\nIntercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance")
print("-----------------")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Diabetes Progression")
plt.show()
