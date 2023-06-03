from simpleclassifier.base import SplitterDataset
from simpleclassifier.factory import SplitterDatasetFactory

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.utils import Bunch


@SplitterDatasetFactory.register("breast_cancer")
class BreastCancerDataset(SplitterDataset):
    """
    A dataset class for loading and splitting the breast cancer dataset.

    The breast cancer dataset contains features describing characteristics of breast
    masses and the corresponding labels indicating whether the masses are malignant
    or benign.
    """

    def load_data(self) -> Bunch:
        """
        Load the breast cancer dataset.

        :return: A dictionary-like object with the following attributes:
                 - 'data': The feature data (X) of shape (n_samples, n_features).
                 - 'target': The target data (y) of shape (n_samples,).
                 - 'feature_names': The names of the features.
                 - 'target_names': The names of the target classes.
        :rtype: sklearn.utils.Bunch
        """
        return load_breast_cancer(return_X_y=True)


@SplitterDatasetFactory.register("iris")
class IrisDataset(SplitterDataset):
    """
    A dataset class for loading and splitting the iris dataset.

    The iris dataset contains measurements of four features of iris flowers along with
    the corresponding species labels.
    """

    def load_data(self) -> Bunch:
        """
        Load the iris dataset.

        :return: A dictionary-like object with the following attributes:
                 - 'data': The feature data (X) of shape (n_samples, n_features).
                 - 'target': The target data (y) of shape (n_samples,).
                 - 'feature_names': The names of the features.
                 - 'target_names': The names of the target classes.
        :rtype: sklearn.utils.Bunch
        """
        return load_iris(return_X_y=True)


@SplitterDatasetFactory.register("wine")
class WineDataset(SplitterDataset):
    """
    A dataset class for loading and splitting the wine dataset.

    The wine dataset contains measurements of various chemical properties of wines and
    the corresponding labels indicating their origin.
    """

    def load_data(self) -> Bunch:
        """
        Load the wine dataset.

        :return: A dictionary-like object with the following attributes:
                 - 'data': The feature data (X) of shape (n_samples, n_features).
                 - 'target': The target data (y) of shape (n_samples,).
                 - 'feature_names': The names of the features.
                 - 'target_names': The names of the target classes.
        :rtype: sklearn.utils.Bunch
        """
        return load_wine(return_X_y=True)
