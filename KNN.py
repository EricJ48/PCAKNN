import numpy as np
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import dot
from numpy.linalg import norm
from numpy.linalg import eig

mnist = load_digits()
x = mnist.data
y = mnist.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=None)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_test = (x_test - np.mean(x_test)) / np.std(x_test)

#PCA calculations
means = np.mean(x_train, axis=0)
A = x_train - means
C = np.cov(A, rowvar=False)
Eigenvalues, Eigenvectors = eig(C)
indicies = np.argsort(Eigenvalues)[::-1]
vectorssorted = Eigenvectors[:, indicies]
valuessorted = Eigenvalues[indicies]
eigen95 = np.sum(valuessorted) * 0.95
eigen90 = np.sum(valuessorted) * 0.90
variance = np.cumsum(valuessorted,axis=0)
components90 = np.argmax(np.cumsum(valuessorted) >= eigen90) + 1
components95 = np.argmax(np.cumsum(valuessorted) >= eigen95) + 1

x_train90 = np.dot(x_train, vectorssorted[:, :components90])
x_test90 = np.dot(x_test, vectorssorted[:, :components90])
x_train95 = np.dot(x_train, vectorssorted[:, :components95])
x_test95 = np.dot(x_test, vectorssorted[:, :components95])




#euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

#manhattan distance function
def manhattan_distance(x1, x2):
    return np.sum(abs(x1 - x2))

#cosine distance function
def cosine_distance(x1, x2):
 return 1-dot(x1, x2)/(norm(x1)*norm(x2))

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

#predict function works by finding distance to all other vectors and sorts by length. returns k closest vectors
    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            d = []
            votes = []
            for j in range(len(x_train)):
                dist = euclidean_distance(x_train[j], x_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(y_train[j])
            ans = Counter(votes).most_common(1)[0][0]
            predictions.append(ans)

        return predictions

    def accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        return (predictions == y_test).sum() / len(y_test)


clf = KNN(5)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print("accuracy and confusion matrix without PCA")
print(clf.accuracy(x_test, y_test))
print(confusion_matrix(y_test, prediction))

x_train = x_train95
x_test = x_test95
clf = KNN(5)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print("95% PCA accuracy and confusion matrix")
print(clf.accuracy(x_test, y_test))
print(confusion_matrix(y_test, prediction))

x_train = x_train90
x_test = x_test90
print("")
clf = KNN(5)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print("90% PCA accuracy and confusion matrix")
print(clf.accuracy(x_test, y_test))
print(confusion_matrix(y_test, prediction))