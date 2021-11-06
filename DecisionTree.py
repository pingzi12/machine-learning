import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
        cancer.target, stratify=cancer.target, random_state=42)
start = datetime.datetime.now()
tree = DecisionTreeClassifier(random_state=0,max_depth=4).fit(X_train,y_train)
end = datetime.datetime.now()
print("Accuracy on training set(tree):{:3f}".format(tree.score(X_train,y_train)))
print("Accuracy on test set(tree):{:3f}".format(tree.score(X_test,y_test)))
print("run_time:",end - start)
