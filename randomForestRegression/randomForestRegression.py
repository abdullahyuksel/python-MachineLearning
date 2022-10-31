import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor




x = np.arange(0,15,0.09).reshape((-1,1))
y = np.cos(x).ravel()

rfr = RandomForestRegressor()
rfr.fit(x,y)




plt.scatter(x,y)
plt.plot(x,rfr.predict(x),color="red")
plt.show()