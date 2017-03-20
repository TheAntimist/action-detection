import os
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from KTH import KTH


dataset = KTH()

# Training Data

if os.path.exists("KTH.pkl"):
    print("Loading the cached X, y vectors for the data.")
    X_train, y_train, Xval, yval, *rest = dataset.read_from_file("KTH.pkl")
else:
    print("Creating X, y vectors for the data.")
    X_train, y_train = dataset.train_features_labels()
    Xval, yval = dataset.validation_features_labels()
    dataset.write_to_file("KTH.pkl")

# Classifying using K Nearest Neighbours
if not os.path.exists('clf.pkl'):
    # n_neighbours = [1, 3, 7, 10, 13, 25, 50, 100, 130]
    n_neighbours = [7]
    accuracy = []
    print("Starting the Classification of data using Nearest Neighour.") # Test with Manhattan Distance as well

    confusion = np.zeros((len(6), len(6)))

    for nn in n_neighbours:
        clf = KNeighborsClassifier(nn, n_jobs=3)
        clf.fit(X_train, y_train)
        count = 0
        for row in Xval:
            row = row.reshape(1, -1)
            y_pred = clf.predict(row)
            confusion[yval[count], y_pred] += 1
            count += 1
        print(confusion.transpose())
        # accuracy.append(accuracy_score(yval, y_pred))
        # print("Accuracy for nn value {}: {}".format(nn, accuracy[-1]))
    # ind = np.argmax(accuracy)
    # print("KNN accuracy is max on the Validation set at {} with accuracy: {}".format(n_neighbours[ind], accuracy[ind]))
    # print(accuracy)


    # print("Checking if Manhattan distance or Euclidean distance is better.")
    #
    # accuracy.clear()
    # for i in range(1, 3):
    #     clf = KNeighborsClassifier(n_neighbours[ind], p=i, n_jobs=3)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(Xval)
    #     accuracy.append(accuracy_score(yval, y_pred))
    #     print("Accuracy for p value {}: {}".format(i, accuracy[-1]))
else:
    clf = joblib.load('clf.pkl')

