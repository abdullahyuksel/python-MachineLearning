from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

x = iris["data"]
y = iris["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

acc = accuracy_score(y_pred,y_test)

print(acc)


