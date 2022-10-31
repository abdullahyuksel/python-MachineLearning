import pandas as pd
from sklearn.linear_model import LinearRegression



dataset = pd.read_csv("dataset.csv")

x = dataset.iloc[:,:1].values
y = dataset.iloc[:,1:].values
print(y)


lr = LinearRegression()
lr.fit(x,y)



pred = lr.predict([[20]])


print(pred)
