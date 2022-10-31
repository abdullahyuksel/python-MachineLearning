import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("dataset.csv")

X = dataset.iloc[:,:2].values
Y = dataset.iloc[:,2:].values

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33)

lgr = LogisticRegression()

lgr.fit(x_train,y_train)

y_pred = lgr.predict(x_test)

cm = confusion_matrix(y_pred,y_test)
acc =accuracy_score(y_pred,y_test)
print(acc)
print(cm)
