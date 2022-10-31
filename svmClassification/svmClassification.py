from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

digits = load_digits()

img = digits["data"]
label = digits["target"]

x_train,x_test,y_train,y_test = train_test_split(img,label,test_size=0.33)



svm = SVC(gamma=0.22)
svm.fit(img,label)


pred = svm.predict(x_test)

cm = confusion_matrix(pred,y_test)
print(cm)
acc = accuracy_score(pred,y_test)
print(acc)