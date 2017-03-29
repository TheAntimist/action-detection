from dataset import base_dataset
import numpy as np
import os
import subprocess as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.externals import joblib


class KTH(base_dataset):
    labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
    video_types = ["d1", "d2", "d3", "d4"]

    train = [11, 12, 13, 14, 15, 16, 17, 18]
    validation = [19, 20, 21, 23, 24, 25, 1, 4]
    test = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    path_to_videos = '/home/emanon/Desktop/BL/data/KTH/'
    path_to_features = path_to_videos + "descr/"

    def __init__(self, n_clusters=1500, path_to_videos=None, path_to_features=None, sequence_file=None,
                 init_file=None, verbose=False, calc_features=False):

        self.n_clusters = n_clusters
        self.verbose = verbose

        self.X_tr = self.y_tr = self.X_val = self.y_val = self.X_test = self.y_test = None

        if path_to_videos:
            self.path_to_videos = path_to_videos


        if path_to_features:
            self.path_to_features = path_to_features

        if calc_features:
            self.log("Calculating STIP Features on the dataset.")
            if not os.path.exists(self.path_to_features): os.mkdir(self.path_to_features)

            for file in os.listdir(path_to_videos):
                filename, ext = os.path.splitext(file)
                for sequence in self.seq_dict[filename]:
                    start, end = sequence.split('-')
                    self.log("Calculating features for {} at sequence {}-{}".format(filename, start, end))
                    self.calc_stip_features(path_to_videos,
                                            path_to_features + filename + "." + start + "-" + end + ".txt",
                                            start, end, filename)

        if init_file is not None and os.path.exists(init_file):
            self.read_from_file(init_file)
        else:
            self.seq_dict = self.fill_sequence_dict()
            # KMeans
            self.kmeans = None
            self.get_train_vocabulary(self.train + self.validation + self.test, self.path_to_features)

    def get_feature_vector_for_file(self, number, label, video_type, start, end):
        file = "person" + str(number).zfill(2) + "_" + label + "_" + video_type + "." + str(start) + "-" + str(end) + ".txt"
        position, descr = self.get_stip_features_from_file(self.path_to_features + file)

        X = self.get_BOW_from_descriptors(descr, self.n_clusters)
        y = self.labels.index(label)
        return X, y

    def log(self, message):
        if self.verbose:
            print(message)

    def fill_sequence_dict(self, file="sequences.txt"):
        '''
        Creates a dictionary of files, file as key and the frame sequences as values of a list.
        :return:
        '''
        d = defaultdict(list)
        with open(file, "r") as sequence_file:
            for line in sequence_file:
                cols = line.split()
                frame_sequences = [str.rstrip(',') for str in cols[2:]]
                d[cols[0]].extend(frame_sequences)
        return d

    def get_seq_count(self, video_number):
        """
        :param video_number:
        :return: Returns the count of sequences present in the video_number.
        """
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
            self.log("Error removing video_list.txt\n")

    def get_stip_features_from_file(self, file):
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
        position = np.genfromtxt(file, comments='#', usecols=(4, 5, 6, 7, 8), dtype=np.int32)
        descriptors = np.genfromtxt(file, comments='#')

        x, *y = descriptors.shape

        if not y:
            descriptors = np.reshape(descriptors, (1, x))

        descriptors = descriptors[:, 9:]

        return position, descriptors

    def get_BOW_from_descriptors(self, descriptors, num_centroids=1500):
        '''

        :param descriptors: n x 160 matrix representing the descriptors for all points in all the videos.
        :return: Histogram of Bag of Features trained using a K-means clustering,
                  this can then be used as a Feature vector for each video
        '''

        assert self.kmeans is not None, "Vocabulary hasn't been trained over the points"

        result_cluster = self.kmeans.predict(descriptors)

        histogram = np.zeros(num_centroids, dtype=np.int32)
        for cluster in result_cluster:
            cluster = int(cluster)
            histogram[cluster] += 1
        return histogram

    def get_BOW_from_file(self, n, video_number, path_to_features):
        '''
        Retrieves the BOW for all the

        :param n: Number of videos to process, predefined to init the numpy matrices
        :param video_number: list containing the video number
        :param path_to_features: path to stip features folder.
        :return: X, the feature vector along with y, the array of labels
        '''

        y = np.empty((n,), dtype=np.int32)  # one video is missing
        X = np.empty((n, self.n_clusters), dtype=np.int32)
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
                        # print("Calculating Descriptor vector for file {}".format(file))
                        position, descr = self.get_stip_features_from_file(path_to_features + file)
                        X[i] = self.get_BOW_from_descriptors(descr, self.n_clusters)
                        y[i] = self.labels.index(label)
                        i += 1

        return X, y

    def get_train_vocabulary(self, video_number, path_to_features, n_clusters=1500):
        '''
        Clusters all the spatio-temporal points and produces a vocabulary of words.
        This is then used to get a Bag of Words, for each sequence.

        :param n_clusters: number of clusters for bag of features
        :param video_number: list containing the video number, almost always, this is all the videos.
        :param path_to_features: path to stip features folder.
        '''

        DESCRIPTOR_BINS = 72 + 90

        self.log("Creating a Vocabulary for the dataset.")
        if os.path.exists("descr.pkl"):
            descriptors = joblib.load("descr.pkl")
        else:
            descriptors = np.empty((0, DESCRIPTOR_BINS), dtype=np.int32)
            for label in self.labels:
                for number in video_number:
                    for type in self.video_types:
                        if number is 13 and type is 'd3' and label is 'handclapping':  # ignore missing file.
                            continue

                        seq_file = 'person' + str(number).zfill(2) + '_' + label + '_' + type

                        for sequence in self.seq_dict[seq_file]:
                            start, end = sequence.split('-')
                            file = seq_file + "." + start + "-" + end + '.txt'

                            position, descr = self.get_stip_features_from_file(path_to_features + file)

                            descriptors = np.concatenate((descriptors, descr), axis=0)

            joblib.dump(descriptors, "descr.pkl")

        self.log("Running KMeans on the points.")
        # KMeans to cluster the descriptors
        n_iterations = 100
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, verbose=self.verbose,
                                      max_iter=n_iterations, batch_size=1000, n_init=8)
        self.kmeans.fit(descriptors)

    def train_features_labels(self, from_file=None):
        if self.X_tr is not None and self.y_tr is not None:
            return self.X_tr, self.y_tr

        if from_file:
            X, y = joblib.load(from_file)
        else:
            n_seq = self.get_seq_count(self.train)

            X, y = self.get_BOW_from_file(n_seq, self.train, self.path_to_features)

        self.X_tr = X
        self.y_tr = y

        return X, y


    def validation_features_labels(self, from_file=None):
        if self.X_val is not None and self.y_val is not None:
            return self.X_val, self.y_val

        if from_file:
            X, y = joblib.load(from_file)
        else:
            n_val = self.get_seq_count(self.validation)

            self.X_val, self.y_val = self.get_BOW_from_file(n_val, self.train, self.path_to_features)

        return self.X_val, self.y_val


    def test_features_labels(self, from_file=None):
        if self.X_test is not None and self.y_test is not None:
            return self.X_test, self.y_test

        if from_file:
            X, y = joblib.load(from_file)
        else:
            n_test = self.get_seq_count(self.test)

            self.X_test, self.y_test = self.get_BOW_from_file(n_test, self.train, self.path_to_features)

        return self.X_test, self.y_test

    def write_to_file(self, file):
        X_tr, y_tr = self.train_features_labels()
        X_val, y_val = self.validation_features_labels()
        X_test, y_test = self.test_features_labels()
        joblib.dump((X_tr, y_tr, X_val, y_val, X_test, y_test, self.seq_dict, self.kmeans), file)

    def read_from_file(self, file):
        self.X_tr, self.y_tr, self.X_val,\
        self.y_val, self.X_test, self.y_test, self.seq_dict, self.kmeans  = joblib.load(file)

        return self.X_tr, self.y_tr, self.X_val, self.y_val, self.X_test, self.y_test, self.seq_dict, self.kmeans
