from sklearn.datasets import load_iris
from sklearn.ensamble import RandomForestClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()

x = iris.data
y = iris.target

rfc = RandomForestClassifier()
rfc.fit(x,y)


pred = rfc.predict(x)

acc = accuracy_score(pred,y)

print(acc)
