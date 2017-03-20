from abc import ABCMeta, abstractmethod
from sklearn.externals import joblib

class base_dataset(metaclass=ABCMeta):
    """
        Base class for all datasets.
    """

    @abstractmethod
    def train_features_labels(self):
        pass

    @abstractmethod
    def validation_features_labels(self):
        pass

    @abstractmethod
    def test_features_labels(self):
        pass

    @abstractmethod
    def write_to_file(self, file):
        pass

    @abstractmethod
    def read_from_file(self, file):
        pass
