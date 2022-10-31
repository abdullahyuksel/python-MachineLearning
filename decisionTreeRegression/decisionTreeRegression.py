import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt




x = np.arange(0,15,0.09).reshape((-1,1))
y = np.cos(x).ravel()


dt = DecisionTreeRegressor()
dt.fit(x,y)


plt.scatter(x,y,color="red")
plt.plot(x,dt.predict(x))
plt.show()

