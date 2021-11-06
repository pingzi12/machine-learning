import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
        cancer.target, stratify=cancer.target, test_size=0.2,random_state=42)
#数据归一化
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(X_train)
X_train=mms.transform(X_train)
X_test=mms.transform(X_test)

#Kernel = ["linear", "poly", "rbf", "sigmoid"]
#for Kernel in Kernel:
start = datetime.datetime.now()
svc = SVC(kernel='poly',
          #degree=1,
          gamma="auto",C=5000)
svc.fit(X_train,y_train)
end = datetime.datetime.now()
#print("The accuracy on training set(svc) under kernel %s is %f" % (Kernel, svc.score(X_train, y_train)))
#print("The accuracy on test set(svc) under kernel %s is %f" % (Kernel, svc.score(X_test, y_test)))
print("The accuracy on training set under kernel poly(svc):{:6f}".format(svc.score(X_train, y_train)))
print("The accuracy on test set under kernel poly(svc):{:6f}".format(svc.score(X_test, y_test)))
print("run_time:",end - start)
