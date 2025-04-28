# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SRIRAM E
RegisterNumber:  212223040207
*/
```
```python
import pandas as pd
data = pd.read_csv('Employee.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
#### Data.head()


![image](https://github.com/user-attachments/assets/78c9b2d7-c05c-4f2b-bae5-55edf9881154)


#### Data.info():


![image](https://github.com/user-attachments/assets/78febfb9-d2b6-4831-bc85-4293e14e5df3)



#### isnull() and sum():


![image](https://github.com/user-attachments/assets/e8aa4470-5a6c-4971-8c88-2bdf981391db)



#### Data Value Counts():


![image](https://github.com/user-attachments/assets/2bfa8f92-b592-493a-a787-2e7894567e6d)



#### Data.head() for salary:


![image](https://github.com/user-attachments/assets/d6140e12-30fe-4738-b67c-ac8b910a8ddb)



#### x.head():


![image](https://github.com/user-attachments/assets/b1aa30c8-f88c-44d2-af18-86de53a4416e)



#### Accuracy Value:


![image](https://github.com/user-attachments/assets/ef8a6650-616f-40c3-a0b7-3b830e3cae50)



#### Data Prediction:


![image](https://github.com/user-attachments/assets/e074336d-cd28-472a-bb4e-e8ea19fcd2eb)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
