# MLL-1

1 ] for standard.py SVM Classification Results
--------------------------
Accuracy: 91.11111111111111 %

Confusion Matrix:
[[15  0  0]
 [ 0 14  1]
 [ 0  3 12]]

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        15
  versicolor       0.82      0.93      0.88        15
   virginica       0.92      0.80      0.86        15

    accuracy                           0.91        45
   macro avg       0.92      0.91      0.91        45
weighted avg       0.92      0.91      0.91        45

![Uploading image.png…]()


screenshot on 31/08/2026


2 ] for KNN.py 

Correct Predictions:
--------------------
Actual: versicolor Predicted: versicolor -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: virginica Predicted: virginica -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: versicolor Predicted: versicolor -> Correct
Actual: setosa Predicted: setosa -> Correct
Actual: setosa Predicted: setosa -> Correct

Wrong Predictions:
-----------------

Total Correct Predictions: 45
Total Wrong Predictions: 0
Total Test Samples: 45


3 ] LinearRegression.py solution
Dataset Shape: (442, 10)
Training Samples: 309
Testing Samples: 133

Model trained successfully.

Intercept: 151.04202449316014
Coefficient: 988.419312489359

Model Performance
-----------------
Mean Squared Error (MSE): 3884.936720961032
Mean Absolute Error (MAE): 50.59307504375872
R² Score: 0.2803417492440603

and the screenshot on 01/09/2026

//the solution for the preprocessing.ppy
First Five Records
   SepalLength  SepalWidth  PetalLength  PetalWidth      Species
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa

Dataset Shape
(150, 5)

Dataset Information
<class 'pandas.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   SepalLength  150 non-null    float64
 1   SepalWidth   150 non-null    float64
 2   PetalLength  150 non-null    float64
 3   PetalWidth   150 non-null    float64
 4   Species      150 non-null    str    
dtypes: float64(4), str(1)
memory usage: 6.0 KB
None

Statistical Summary
       SepalLength  SepalWidth  PetalLength  PetalWidth
count   150.000000  150.000000   150.000000  150.000000
mean      5.843333    3.054000     3.758667    1.198667
std       0.828066    0.433594     1.764420    0.763161
min       4.300000    2.000000     1.000000    0.100000
25%       5.100000    2.800000     1.600000    0.300000
50%       5.800000    3.000000     4.350000    1.300000
75%       6.400000    3.300000     5.100000    1.800000
max       7.900000    4.400000     6.900000    2.500000

Missing Values
SepalLength    0
SepalWidth     0
PetalLength    0
PetalWidth     0
Species        0
dtype: int64

Missing Values After Introducing
SepalLength    1
SepalWidth     0
PetalLength    0
PetalWidth     1
Species        0
dtype: int64

Missing Values After Filling
SepalLength    0
SepalWidth     0
PetalLength    0
PetalWidth     0
Species        0
dtype: int64

Encoded Dataset
   SepalLength  SepalWidth  PetalLength  PetalWidth  Species
0          5.1         3.5          1.4         0.2        0
1          4.9         3.0          1.4         0.2        0
2          4.7         3.2          1.3         0.2        0
3          4.6         3.1          1.5         0.2        0
4          5.0         3.6          1.4         0.2        0

Training Data Shape: (120, 4)
Testing Data Shape: (30, 4)

First Five Rows of Scaled Training Data
[[-1.48000784  1.22037928 -1.5639872  -1.33026696]
 [-0.13774752  3.02001693 -1.27728011 -1.06176431]
 [ 1.08248913  0.09560575  0.38562104  0.28074896]
 [-1.23596051  0.77046987 -1.21993869 -1.33026696]
 [-1.72405517  0.32056046 -1.39196294 -1.33026696]]

Data Preprocessing Completed Successfully.
Selection deleted


//decision tree result below

PLAY TENNIS DATASET
     Outlook Temperature Humidity    Wind PlayTennis
0      Sunny         Hot     High    Weak         No
1      Sunny         Hot     High  Strong         No
2   Overcast         Hot     High    Weak        Yes
3       Rain        Mild     High    Weak        Yes
4       Rain        Cool   Normal    Weak        Yes
5       Rain        Cool   Normal  Strong         No
6   Overcast        Cool   Normal  Strong        Yes
7      Sunny        Mild     High    Weak         No
8      Sunny        Cool   Normal    Weak        Yes
9       Rain        Mild   Normal    Weak        Yes
10     Sunny        Mild   Normal  Strong        Yes
11  Overcast        Mild     High  Strong        Yes
12  Overcast         Hot   Normal    Weak        Yes
13      Rain        Mild     High  Strong         No

Decision Tree Model Trained Successfully.

New Sample:
  Outlook Temperature Humidity    Wind
0   Sunny        Cool     High  Strong
Prediction: NO - Do Not Play Tennis

Prediction Probability:
No  : 1.0
Yes : 0.0

and also the screenshot 09/09/2026.lxl
