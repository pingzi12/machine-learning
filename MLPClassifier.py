from  sklearn.neural_network import MLPClassifier
import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
        cancer.target, stratify=cancer.target, random_state=42)
#数据缩放
from sklearn.preprocessing import StandardScaler
sds=StandardScaler()
sds.fit(X_train)
X_train=sds.transform(X_train)
X_test=sds.transform(X_test)

start = datetime.datetime.now()
mlp=MLPClassifier(random_state=0,max_iter=1000,alpha=1).fit(X_train,y_train)
end = datetime.datetime.now()
print("Training set score(LR): {:.3f}".format(mlp.score(X_train, y_train)))
print("Test set score(LR): {:.3f}".format(mlp.score(X_test, y_test)))
print("run_time:",end - start)
