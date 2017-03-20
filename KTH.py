from dataset import base_dataset
import numpy as np
import os
import subprocess as sp
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.externals import joblib


class KTH(base_dataset):

    def __init__(self, n_clusters=1500, path_to_videos=None, path_to_features=None, sequence_file=None):
        self.path_to_videos = '/home/emanon/Desktop/BL/data/KTH/'
        self.path_to_features = self.path_to_videos + "descr/"
        self.n_clusters = n_clusters

        self.train = [11, 12, 13, 14, 15, 16, 17, 18]
        self.validation = [19, 20, 21, 23, 24, 25, 1, 4]
        self.test = [22, 2, 3, 5, 6, 7, 8, 9, 10]

        self.labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
        self.video_types = ["d1", "d2", "d3", "d4"]

        if path_to_videos:
            self.path_to_videos = path_to_videos


        if path_to_features:
            self.path_to_features = path_to_features

        if sequence_file:
            self.seq_dict = self.fill_sequence_dict(sequence_file)
        else:
            self.seq_dict = self.fill_sequence_dict()

        if not os.path.exists(self.path_to_features):
            print("Calculating STIP Features on the dataset.")
            os.mkdir(self.path_to_features)
            for file in os.listdir(path_to_videos):
                filename, ext = os.path.splitext(file)
                for sequence in self.seq_dict[filename]:
                    start, end = sequence.split('-')
                    print("Calculating features for {} at sequence {}-{}".format(filename, start, end))
                    self.calc_stip_features(path_to_videos, path_to_features + filename + "." + start + "-" + end + ".txt",
                                     start, end, filename)

    def fill_sequence_dict(self, file="sequences.txt"):
        '''
        Creates a dictionary of files as key and the frame sequences as values of data type list.
        :return:
        '''
        d = defaultdict(list)
        with open(file, "r") as sequence_file:
            for line in sequence_file:
                cols = line.split()
                frame_sequences = [str.rstrip(',') for str in cols[2:]]
                d[cols[0]].extend(frame_sequences)
        return d

    def getn_seq(self, video_number):
        count = 0
        for number in video_number:
            for label in self.labels:
                for type in self.video_types:
                    if number is 13 and type is 'd3' and label is 'handclapping':  # ignore missing file.
                        continue
                    seq_file = 'person' + str(number).zfill(2) + '_' + label + '_' + type

                    count += len(self.seq_dict[seq_file])
        return count

    def calc_stip_features(self, path_to_videos, path_to_output_file, start_frame, end_frame, file):
        '''
        Calculates the STIP features for a specified set of input files and stores them in the output location.

        :param video_files: List of video's files the STIP features are to be computed for. DO NOT USE
        :param path_to_videos: Directory containing the training dataset to be computed
        :param path_to_output_file: Output file location containing the STIP features.
        :param start_frame:
        :param end_frame:
        :return:
        '''

        path_to_stipdet = '/home/emanon/Desktop/BL/code/stip-2.0-linux/bin/stipdet'
        path_to_stipdet_lib = '/home/emanon/Desktop/BL/code/stip-2.0-linux/lib'
        # contains links to export the LD_LIBRARY_PATH correctly

        with open(path_to_videos + "video_list.txt", "w") as video_list:
            # if video_files is None:
            #     video_files = os.listdir(path_to_videos)
            #
            # for file in video_files:
            video_list.write(os.path.splitext(file)[0] + " " + start_frame + " " + end_frame + "\n")

        args = path_to_stipdet + " -i " + \
               path_to_videos + "video_list.txt " + "-vpath " + path_to_videos + " -o " + \
               path_to_output_file + " -det harris3d -vis no -stdout no"

        process = ["/bin/bash", "-c", args]

        proc = sp.Popen(process, env=dict(os.environ, LD_LIBRARY_PATH=path_to_stipdet_lib))
        proc.wait()
        try:
            os.remove(path_to_videos + "video_list.txt")
        except:
            print("Error removing video_list.txt\n")

    def getstipfeaturesfromfile(self, file):
        '''

        Assumes the file contains the features of only one single video.

        Format of features in the file:
        point-type y-norm x-norm t-norm y x t sigma2 tau2 dscr-hog(72) dscr-hof(90)

        Note, the point-type is skipped.
        :param file: File containing the STIP features.
        :return:
            position: Position of the Spatio Temporal Point on the the frames
            descriptors: HOG/HOF descriptors
        '''
        position = np.genfromtxt(file, comments='#', usecols=(4, 5, 6, 7, 8))
        descriptors = np.genfromtxt(file, comments='#')[:, 9:]

        return position, descriptors

    def getboffromdescriptors(self, descriptors, num_centroids=1500):
        '''

        :param descriptors: n x 160 matrix representing the descriptors for all points in all the videos.
        :return: Histogram of Bag of Features trained using a K-means clustering,
                  this can then be used as a Feature vector for each video
        '''

        n_iterations = 100
        kmeans = KMeans(n_clusters=num_centroids, n_jobs=3, max_iter=n_iterations)

        result_cluster = kmeans.fit_predict(descriptors)

        histogram = np.zeros(num_centroids, dtype=np.int32)
        for cluster in result_cluster:
            cluster = int(cluster)
            histogram[cluster] += + 1
        return histogram

    def getfeaturevectorspersequence(self, n, video_number, path_to_features):
        '''

        :param n: Number of videos to process, predefined to init the numpy matrices
        :param n_clusters: number of clusters for bag of features
        :param video_number: list containing the video number
        :param path_to_features: path to stip features folder.
        :return: X, the feature vector along with y, the array of labels
        '''

        y = np.empty((n,), dtype=np.int32)  # one video is missing
        X = np.empty((n, 72 + 90), dtype=np.int32)
        i = 0

        for number in video_number:
            for label in self.labels:
                for type in self.video_types:
                    if number is 13 and type is 'd3' and label is 'handclapping':  # ignore missing file.
                        continue
                    seq_file = 'person' + str(number).zfill(2) + '_' + label + '_' + type

                    for sequence in self.seq_dict[seq_file]:
                        start, end = sequence.split('-')
                        file = seq_file + "." + start + "-" + end + '.txt'
                        print("Calculating Descriptor vector for file {}".format(file))
                        position, descr = self.getstipfeaturesfromfile(path_to_features + file)

                        X[i] = descr
                        y[i] = self.labels.index(label)
                        i += 1

        return X, y

    def getVocabulary(self, n, video_number, path_to_features, n_clusters=1500):
        '''

        :param n: Number of videos to process, predefined to init the numpy matrices
        :param n_clusters: number of clusters for bag of features
        :param video_number: list containing the video number
        :param path_to_features: path to stip features folder.
        :return: X, the histogram of the Clusters vector along with y, the array of labels
        '''

        DESCRIPTOR_BINS = 72 + 90
        finalX = np.empty((len(self.labels), n_clusters), dtype=np.int32)
        finalY = np.empty((len(self.labels),), dtype=np.int32) # one video is missing
        i = 0
        finali = 0

        for label in self.labels:

            tempX = np.empty((0, DESCRIPTOR_BINS), dtype=np.int32)

            for number in video_number:
                for type in self.video_types:
                    if number is 13 and type is 'd3' and label is 'handclapping':  # ignore missing file.
                        continue

                    seq_file = 'person' + str(number).zfill(2) + '_' + label + '_' + type

                    for sequence in self.seq_dict[seq_file]:
                        start, end = sequence.split('-')
                        file = seq_file + "." + start + "-" + end + '.txt'
                        position, descr = self.getstipfeaturesfromfile(path_to_features + file)

                        tempX = np.concatenate((tempX, descr), axis=0)

            feature_hist = self.getboffromdescriptors(tempX, n_clusters)
            finalX[finali] = feature_hist
            finalY[finali] = self.labels.index(label)

            finali += 1
            print("Feature vector for {}: {}".format(label, feature_hist))

        return finalX, finalY

    def train_features_labels(self, from_file=None):
        if from_file:
            X, y = joblib.load(from_file)
        else:
            n_seq = self.getn_seq(self.train)

            X, y = self.getVocabulary(n_seq, self.train, self.path_to_features, self.n_clusters)

        return X, y


    def validation_features_labels(self, from_file=None):

        if from_file:
            X, y = joblib.load(from_file)
        else:
            n_val = self.getn_seq(self.validation)

            X, y = self.getVocabulary(n_val, self.validation, self.path_to_features, self.n_clusters)

        return X, y


    def test_features_labels(self, from_file=None):

        if from_file:
            X, y = joblib.load(from_file)
        else:
            n_test = self.getn_seq(self.test)

            X, y = self.getVocabulary(n_test, self.test, self.path_to_features, self.n_clusters)

        return X, y


    def write_to_file(self, file):
        X_tr, y_tr = self.train_features_labels()
        X_val, y_val = self.validation_features_labels()
        X_test, y_test = self.test_features_labels()
        joblib.dump((X_tr, y_tr, X_val, y_val, X_test, y_test), file)
        return X_tr, y_tr, X_val, y_val, X_test, y_test


    def read_from_file(self, file):
        X_tr, y_tr, X_val, y_val, X_test, y_test = joblib.load(file)
        return X_tr, y_tr, X_val, y_val, X_test, y_test