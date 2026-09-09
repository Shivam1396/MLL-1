import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = {
    'Outlook': [
        'Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
        'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
        'Sunny', 'Overcast', 'Overcast', 'Rain'
    ],
    'Temperature': [
        'Hot', 'Hot', 'Hot', 'Mild', 'Cool',
        'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
        'Mild', 'Mild', 'Hot', 'Mild'
    ],
    'Humidity': [
        'High', 'High', 'High', 'High', 'Normal',
        'Normal', 'Normal', 'High', 'Normal', 'Normal',
        'Normal', 'High', 'Normal', 'High'
    ],
    'Wind': [
        'Weak', 'Strong', 'Weak', 'Weak', 'Weak',
        'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
        'Strong', 'Strong', 'Weak', 'Strong'
    ],
    'PlayTennis': [
        'No', 'No', 'Yes', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes', 'No'
    ]
}

df = pd.DataFrame(data)
print("PLAY TENNIS DATASET")
print(df)

X = pd.get_dummies(df[['Outlook', 'Temperature', 'Humidity', 'Wind']])
y = df['PlayTennis'].map({'No': 0, 'Yes': 1})

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X, y)
print("\nDecision Tree Model Trained Successfully.")

new_sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
})

new_sample_encoded = pd.get_dummies(new_sample)
new_sample_encoded = new_sample_encoded.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(new_sample_encoded)

print("\nNew Sample:")
print(new_sample)
if prediction[0] == 1:
    print("Prediction: YES - Play Tennis")
else:
    print("Prediction: NO - Do Not Play Tennis")

probability = model.predict_proba(new_sample_encoded)
print("\nPrediction Probability:")
print("No  :", round(probability[0][0], 3))
print("Yes :", round(probability[0][1], 3))

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=['No', 'Yes'],
    filled=True
)
plt.title("Decision Tree using ID3 (Entropy)")
plt.show()
