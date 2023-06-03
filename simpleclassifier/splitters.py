from simpleclassifier.base import Splitter
from simpleclassifier.factory import SplitterFactory

from sklearn.model_selection import train_test_split


@SplitterFactory.register("percentage")
class PercentageSplitter(Splitter):
    """
    A splitter implementation that performs data splitting based
    on a specified percentage.

    This class splits the data into training and testing
    subsets based on the provided percentage.
    It supports splitting with or without a random state.
    If random state is not specified (i.e., random_state=None),
    the same split will be produced every time.
    """

    def split_data(self, X, y):
        """
        Perform the data splitting based on the specified percentage.

        :param X: The input feature matrix.
        :type X: array-like or sparse matrix
        :param y: The target variable.
        :type y: array-like
        :return: The training and testing subsets.
        :rtype: tuple (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False,
            stratify=None,
        )
        return X_train, X_test, y_train, y_test


@SplitterFactory.register("percentage_shuffle")
class PercentageShuffleSplitter(Splitter):
    """
    A splitter implementation that performs data splitting based on a
    specified percentage with shuffling.

    This class splits the data into training and testing subsets based on the
    provided percentage, and shuffles the data before splitting. It supports
    splitting with or without a random state.
    """

    def split_data(self, X, y):
        """
        Perform the data splitting based on the specified percentage with shuffling.

        :param X: The input feature matrix.
        :type X: array-like or sparse matrix
        :param y: The target variable.
        :type y: array-like
        :return: The training and testing subsets.
        :rtype: tuple (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=None,
        )
        return X_train, X_test, y_train, y_test


@SplitterFactory.register("percentage_stratified")
class PercentageStratifiedSplitter(Splitter):
    """
    A splitter implementation that performs data splitting based on a specified
    percentage with stratification.

    This class splits the data into training and testing subsets based on the
    provided percentage while maintaining the class proportions in both subsets.
    If the original dataset has a certain proportion of samples in each class,
    the training and testing sets will also have the same proportion of each class.
    """

    def split_data(self, X, y):
        """
        Perform the data splitting based on the specified percentage
        with stratification.

        :param X: The input feature matrix.
        :type X: array-like or sparse matrix
        :param y: The target variable.
        :type y: array-like
        :return: The training and testing subsets.
        :rtype: tuple (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test
