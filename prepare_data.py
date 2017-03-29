import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from KTH import KTH
import numpy as np


DATASET_FILE = "KTH.pkl"
overwrite = True

dataset = KTH(init_file=DATASET_FILE, verbose=True)
# Training Data

print("Creating X, y vectors for the data.")
X_train, y_train = dataset.train_features_labels()
Xval, yval = dataset.validation_features_labels()
X_test, y_test = dataset.test_features_labels()

if not os.path.exists(DATASET_FILE) and overwrite:
    dataset.write_to_file(DATASET_FILE)


def my_conf(n_labels, y_true, y_pred):

    confusion = np.zeros((n_labels, n_labels), dtype=np.int32)
    for i in range(len(y_true)):
        confusion[y_true[i], y_pred[i]] += 1
    return confusion

def predict(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    cf = confusion_matrix(y, y_pred, labels=range(len(KTH.labels)))
    pr_sc = precision_score(y, y_pred, labels=range(len(KTH.labels)), average='macro')
    re_sc = recall_score(y, y_pred, labels=range(len(KTH.labels)), average='macro')
    f_sc = f1_score(y, y_pred, labels=range(len(KTH.labels)), average='macro')
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix:\n{}".format(cf))
    print("F1 Score: {}\nPrecision: {}, Recall: {}".format(f_sc, pr_sc, re_sc))
    return accuracy, cf

# Classifying using K Nearest Neighbours

nn = 0
if not os.path.exists('clf.pkl'):
    n_neighbours = [ 1, 3, 7, 10]
    accuracy = []
    print("Starting the Classification of data using Nearest Neighour.") # Test with Manhattan Distance as well

    for nn in n_neighbours:
        print("NN {}:".format(nn))
        clf = KNeighborsClassifier(nn, n_jobs=3)
        clf.fit(X_train, y_train)
        a, cf = predict(clf, Xval, yval)
        accuracy.append(a)
    ind = np.argmax(accuracy)
    print("KNN accuracy is max on the Validation set at {} with accuracy: {}".format(n_neighbours[ind], accuracy[ind]))
    nn = n_neighbours[ind]
else:
    clf = joblib.load('clf.pkl')

nn = 3
clf = KNeighborsClassifier(nn, n_jobs=3)
clf.fit(X_train, y_train)
predict(clf, X_test, y_test)