import os
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from KTH import KTH


DATASET_FILE = "KTH.pkl"
overwrite = False

dataset = KTH(init_file=DATASET_FILE, verbose=True)
# Training Data

print("Creating X, y vectors for the data.")
X_train, y_train = dataset.train_features_labels()
Xval, yval = dataset.validation_features_labels()
X_test, y_test = dataset.test_features_labels()

if not os.path.exists(DATASET_FILE) and overwrite:
    dataset.write_to_file(DATASET_FILE)

# Classifying using K Nearest Neighbours
if not os.path.exists('clf.pkl'):
    n_neighbours = [1, 3, 7, 10, 13, 25, 50, 100, 130]
    # n_neighbours = [7]
    accuracy = []
    print("Starting the Classification of data using Nearest Neighour.") # Test with Manhattan Distance as well

    # confusion = np.zeros((len(6), len(6)))

    for nn in n_neighbours:
        clf = KNeighborsClassifier(nn, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(Xval)
        # count = 0
        # for row in Xval:
            # row = row.reshape(1, -1)
            # confusion[yval[count], y_pred] += 1
            # count += 1
            # print(confusion.transpose())
        accuracy.append(accuracy_score(yval, y_pred))
        print("Accuracy for nn value {}: {}".format(nn, accuracy[-1]))
    ind = np.argmax(accuracy)
    print("KNN accuracy is max on the Validation set at {} with accuracy: {}".format(n_neighbours[ind], accuracy[ind]))
else:
    clf = joblib.load('clf.pkl')

clf = KNeighborsClassifier(5, n_jobs=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy for nn value {}: {}".format(5, accuracy_score(y_test, y_pred)))