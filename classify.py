import os
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import accuracy_score

if os.path.exists("training_vector.pkl"):
    print("Loading the cached X, y vectors for training data.")
    X_train, y_train = joblib.load("training_vector.pkl")

if os.path.exists("validation_vector.pkl"):
    print("Loading the cached X, y vectors for validating the data.")
    Xval, yval = joblib.load("validation_vector.pkl")

clf = svm.SVC(kernel=chi2_kernel, C=1)
#clf = svm.LinearSVC(C=0.3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
#print("Accuracy on Training Data: {}\nY Predicted: {}\n".format(np.array_equal(y_pred, y_train), y_pred))
y_pred = clf.predict(Xval)
print("Accuracy on Validation Data: {}\nY Predicted: {}\n".format(accuracy_score(yval, y_pred), y_pred))
