import abc
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class Splitter(abc.ABC):
    """
    An abstract base class for splitters that splits the
    dataset into train and test data.
    """

    def __init__(self, test_size: float, random_state: int = 0):
        """
        Initialize the Splitter.

        :param test_size: The fraction that defines the test data split.
        :type test_size: float
        :param random_state: A controller that controls shuffling.
        :type random_state: int, optional
        :raises ValueError: If the test_size is not within the range [0.5, 0.95].
        """
        if not (0.05 <= test_size <= 0.95):
            raise ValueError("test_size must be between 0.5 and 0.95")

        self.test_size = test_size
        self.random_state = random_state

    @abc.abstractmethod
    def split_data(self, X,
                   y) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into training and testing data.

        :param X: Two-dimensional feature matrix of shape (n_samples, n_features).
        :type X: np.ndarray
        :param y: One-dimensional array for target variables of shape (n_samples).
        :type y: np.ndarray
        :return: A tuple of four arrays: X_train, X_test, y_train, y_test.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        pass


class SplitterDataset(abc.ABC):
    """
    An abstract base class for dataset splitters.
    """

    def __init__(self, splitter: Splitter):
        """
        Initialize the SplitterDataset.

        :param splitter: The splitter object used for splitting the dataset.
        :type splitter: Splitter
        """
        assert splitter is not None
        X, y = self.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = splitter.split_data(
            X, y)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the data for the specific dataset.

        :return: A tuple containing the feature data (X) and the target data (y).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        raise NotImplementedError(
            "Subclasses must implement the load_data method")


class Classifier(abc.ABC):
    """
    An abstract base class for classifiers.
    """

    def __init__(self, dataset: SplitterDataset):
        """
        Initialize the Classifier.

        :param dataset: The dataset object
        used for training and testing the classifier.
        :type dataset: SplitterDataset
        """
        self.dataset = dataset

    @abc.abstractmethod
    def fit(self):
        """
        Train the classifier using the training data.
        """
        raise NotImplementedError("Subclasses must implement the fit method")

    @abc.abstractmethod
    def predict(self) -> np.ndarray:
        """
        Make predictions using the trained classifier.

        :return: The predicted labels.
        :rtype: np.ndarray
        """
        raise NotImplementedError(
            "Subclasses must implement the predict method")
