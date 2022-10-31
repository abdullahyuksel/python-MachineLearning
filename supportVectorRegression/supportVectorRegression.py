import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


x = np.arange(0,15,0.09).reshape((-1,1))
y = np.cos(x).ravel()


svr_rbf = SVR(kernel="rbf",gamma=0.22)
svr_linear = SVR(kernel="linear",gamma=0.22)
svr_poly = SVR(kernel="poly",gamma=0.22)

svr_rbf.fit(x,y)
svr_linear.fit(x,y)
svr_poly.fit(x,y)


plt.plot(x,svr_rbf.predict(x),color="darkorange")
plt.plot(x,svr_linear.predict(x),color="pink")
plt.plot(x,svr_poly.predict(x),color="green")
plt.scatter(x,y)
plt.show()



