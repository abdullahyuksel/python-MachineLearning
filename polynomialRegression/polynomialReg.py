import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("dataset.csv")


x = dataset.iloc[:,:1].values
y = dataset.iloc[:,1:].values


pl = PolynomialFeatures(degree=6)
poly_x = pl.fit_transform(x)


lr = LinearRegression()
lr.fit(poly_x,y)


lrn = LinearRegression()
lrn.fit(x,y)

plt.scatter(x,y)
plt.plot(x,lr.predict(poly_x),color="red")
plt.plot(x,lrn.predict(x),color="yellow")
plt.show()




