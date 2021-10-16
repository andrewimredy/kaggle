import numpy as np
import math
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Data cleaning
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
#turn string input into numeric input
x_train = train[features]
x_train['Sex'] = np.where(x_train['Sex'] == 'male', 1, 0)
y_train = train.Survived

x_test = test[features]
x_test['Sex'] = np.where(x_test['Sex'] == 'male', 1, 0)
x_test['Age'] = x_test['Age'].fillna(value=np.average(x_test['Age']))
x_test['Fare'][152] = 10 #null value

print(x_test)
print(x_test.isnull())

tree = DecisionTreeClassifier()
tree = tree.fit(x_train, y_train)

#Predictions & write to csv
y_pred =  tree.predict(x_test)
s = {'PassengerId': test['PassengerId'], 'Survived': y_pred}
submission = pd.DataFrame(data=s)
print(submission)
submission.to_csv('submission.csv', index=False)