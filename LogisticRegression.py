#导包
import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#划分数据集
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
        cancer.target, stratify=cancer.target, random_state=42)
#计算运行时间
start = datetime.datetime.now()
#函数使用，括号内进行调参
LogR=LogisticRegression(C=100,solver='liblinear').fit(X_train, y_train)
end = datetime.datetime.now()
#打印结果
print("Training set score(LR): {:.6f}".format(LogR.score(X_train, y_train)))
print("Test set score(LR): {:.6f}".format(LogR.score(X_test, y_test)))
print("run_time:",end - start)
