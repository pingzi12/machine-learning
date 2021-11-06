import matplotlib.pyplot as plt#画图工具
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Import some data to play with
cancer = datasets.load_breast_cancer()
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
        cancer.target, stratify=cancer.target, random_state=42)
#  Learn to predict each class against the other
Lr=LinearRegression()
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = Lr.fit(X_train, y_train).predict(X_test)
print("The accuracy on training set(svc):{:6f}".format(Lr.score(X_train, y_train)))
print("The accuracy on test set(svc):{:6f}".format(Lr.score(X_test, y_test)))
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
print("auc:",roc_auc)#打印auc的值
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()