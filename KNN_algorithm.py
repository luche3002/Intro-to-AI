from sklearn.datasets import load_digits
digits = load_digits()
type = (digits)
print (digits.DESCR)
digits.target[9]
digits.target[::4]
digits.data[0]
digits.images[9]
digits.data.shape

import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows = 4, ncols = 6, figsize = (6, 4))
for item in zip (axes.ravel(), digits.images, digits.target):
    axes, images, target = item
    axes.imshow (image, cmap = plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(digits.data, digits.target, random_state=11)
digits.data.shape
X_train.shape
y_train.shape
X_test.shape
1347/1797*100

# Creating the model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

# Prediction
predicated = knn.predict(X=X_test)
expected = y_test
expected
predicated
wrong=[(p,e) for (p,e) in zip(predicated, expected) if p !=e]
wrong
440/450*100
print(f'{knn.score(X_test, y_test):.2%}')
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_true=expected, y_pred=predicated)
confusion
