#导包
import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
#划分数据集
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
        cancer.target, stratify=cancer.target, random_state=42)
#时间计算
start = datetime.datetime.now()
#函数使用，括号内进行调参
gbrt = GradientBoostingClassifier(random_state=0,max_depth=1,n_estimators=200,learning_rate=0.1).fit(X_train,y_train)
end = datetime.datetime.now()
print("Accuracy on training set(GBRT):{:3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on test set(GBRT):{:3f}".format(gbrt.score(X_test,y_test)))
print("run_time:",end - start)
