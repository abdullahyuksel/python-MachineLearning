from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


iris = load_iris()

x = iris.data
y = iris.target

dcc = DecisionTreeClassifier()
dcc.fit(x,y)


pred = dcc.predict(x)

cm = confusion_matrix(pred,y)

print(cm)
