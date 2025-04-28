# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SRIRAM E
RegisterNumber: 212223040207
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
### Value of df
![Screenshot 2025-04-27 114859](https://github.com/user-attachments/assets/759ee2e4-a7b6-485a-8456-1088e8d504d7)
### df.head() and df.isnull().sum()
![Screenshot 2025-04-27 114914](https://github.com/user-attachments/assets/56a8dfa4-cd4f-435b-bb28-594f870c2e63)
### df.describe()
![Screenshot 2025-04-27 114934](https://github.com/user-attachments/assets/0685d091-a2cc-4335-973f-c83547b3456f)
### df.info() and Value counts
![Screenshot 2025-04-27 114945](https://github.com/user-attachments/assets/ad1d5258-322b-4f05-9eef-d0ec50e86005)
### df.head()
![Screenshot 2025-04-27 115003](https://github.com/user-attachments/assets/c4d39916-5bff-4a3c-a92a-d6193b363730)
### Value of x.head() and y
![Screenshot 2025-04-27 115003](https://github.com/user-attachments/assets/66bc82ad-9915-4e83-8950-0f6560a0db23)
### Value of Accuracy,Confusion_matrix and data prediction
![Screenshot 2025-04-27 120512](https://github.com/user-attachments/assets/9701d0bf-3034-438b-ab2a-c6cf2333e036)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
