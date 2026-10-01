import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "attendance": np.random.randint(40, 101, n),
    "assignment_marks": np.random.randint(0, 21, n),
    "internal_marks": np.random.randint(0, 31, n),
    "study_hours": np.round(np.random.uniform(0, 8, n), 1)
})
score = (0.3 * df["attendance"] + df["assignment_marks"]
         + df["internal_marks"] + 2.0 * df["study_hours"])
df["result"] = (score >= score.median()).astype(int) 
X = df[["attendance", "assignment_marks", "internal_marks", "study_hours"]]
y = df["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1-score :", round(f1_score(y_test, y_pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(export_text(model, feature_names=list(X.columns)))
new_student = pd.DataFrame([[80, 15, 22, 4.0]], columns=X.columns)
print("New student prediction:", "PASS" if model.predict(new_student)[0] == 1 else "FAIL")
