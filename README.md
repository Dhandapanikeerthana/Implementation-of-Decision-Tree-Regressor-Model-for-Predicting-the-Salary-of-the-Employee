# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2. Calculate the null values present in the dataset and apply label encoder.

3. Determine test and training data set and apply decison tree regression in dataset.

4. Calculate Mean square error,data prediction and r2. 


## Program:
```

Developed by: KEERTHANA D
RegisterNumber: 212224040155

```
```
import pandas as pd
data = pd.read_csv("/content/Salary.csv")


data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])

```

## Output:

```
DATA HEAD:
<img width="482" height="310" alt="image" src="https://github.com/user-attachments/assets/8ede3e41-2d6e-4f3f-9197-a994a81ea988" />

```

DATA INFO:
<img width="445" height="294" alt="image" src="https://github.com/user-attachments/assets/2cd5bd8c-5b62-4278-a154-7199633d9282" />



isnull.sum()
<img width="274" height="280" alt="image" src="https://github.com/user-attachments/assets/6732901c-41e6-4227-a0cf-0afc8da6a1f7" />



DATA HEAD OF SALARY:
<img width="382" height="249" alt="image" src="https://github.com/user-attachments/assets/a871b637-da27-4dbc-829c-99c7d6a06d40" />



MEAN SQUARED ERROR:
<img width="180" height="30" alt="image" src="https://github.com/user-attachments/assets/6e70d5f5-8c77-4a4c-ba0f-a5a686702cf9" />



R2 VALUE:
<img width="213" height="43" alt="image" src="https://github.com/user-attachments/assets/987a110d-0f51-41f6-9366-3070ee068fc4" />


DATA PREDICTION:
<img width="902" height="61" alt="image" src="https://github.com/user-attachments/assets/cb56dd48-7bc3-4aa2-a4d2-31ae9357380c" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
