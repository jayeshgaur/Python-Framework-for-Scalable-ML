from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
from sklearn.exceptions import ConvergenceWarning

from simpleclassifier.base import Classifier
from simpleclassifier.factory import ClassifierFactory
from simpleclassifier.hyperparameter_tuner import SKLearnHyperparameterTuner

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@ClassifierFactory.register("knn")
class KNNClassifier(Classifier):
    """
    A classifier implementation using the k-nearest neighbors
    algorithm.

    The k-nearest neighbors (KNN) algorithm is a non-parametric
    classification method that classifies new instances based
    on their similarity to the k most similar instances in
    the training data.

    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.model = KNeighborsClassifier()

    def fit(self):
        """
        Fit the k-nearest neighbors model using hyperparameter tuning.

        The KNN model is tuned on the following hyperparameters:

        - 'n_neighbors': The number of neighbors to consider.

        - 'weights': The weight function used in prediction.

        - 'algorithm': The algorithm used to compute nearest neighbors.

        - 'p': The power parameter for the Minkowski metric.

        """
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
        tuner = SKLearnHyperparameterTuner(self.dataset)
        self.model = tuner.tune_model(self.model, param_grid)

    def predict(self):
        """
        Make predictions using the trained k-nearest neighbors model.

        :return: The predicted labels.
        :rtype: np.ndarray
        """
        return self.model.predict(self.dataset.X_test)


@ClassifierFactory.register("lr")
class LogisticRegressionClassifier(Classifier):
    """
    A classifier implementation using logistic regression.

    Logistic regression is a linear classification
    algorithm that models the relationship between the
    input features and the binary target variable
    using a logistic function.

    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.model = LogisticRegression()

    def fit(self):
        """
        Fit the logistic regression model using
        hyperparameter tuning.

        The logistic regression model is tuned
        on the following hyperparameters:

        - 'C': Inverse of regularization strength.

        - 'penalty': Regularization term type.

        - 'solver': Algorithm to use in optimization.

        """
        param_grid = {
            'C': [0.1, 1.0, 5.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        tuner = SKLearnHyperparameterTuner(self.dataset)
        self.model = tuner.tune_model(self.model, param_grid)

    def predict(self):
        """
        Make predictions using the trained
        logistic regression model.

        :return: The predicted labels.
        :rtype: np.ndarray
        """
        return self.model.predict(self.dataset.X_test)


@ClassifierFactory.register("rf")
class RandomForestEnsembleClassifier(Classifier):
    """
    A classifier implementation using random forest.

    Random Forest is an ensemble learning method
    that combines multiple decision trees to
    make predictions. Each tree is trained
    on a random subset of the training data, and the
    final prediction is made by aggregating the
    predictions of all the individual trees.

    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.model = RandomForestClassifier()

    def fit(self):
        """
        Fit the random forest model using
        hyperparameter tuning.

        The random forest model is tuned on the
        following hyperparameters:

        - 'n_estimators': The number of trees in the forest.

        - 'criterion': The function to measure the quality of a split.

        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy']
        }
        tuner = SKLearnHyperparameterTuner(self.dataset)
        self.model = tuner.tune_model(self.model, param_grid)

    def predict(self):
        """
        Make predictions using the trained
        random forest model.

        :return: The predicted labels.
        :rtype: np.ndarray
        """
        return self.model.predict(self.dataset.X_test)


@ClassifierFactory.register("svm")
class SVMClassifier(Classifier):
    """
    A classifier implementation using support
    vector machines (SVM).

    Support Vector Machines (SVM) is a powerful
    classification algorithm that separates
    data points into different classes by
    finding the optimal hyperplane that maximally
    separates the classes.

    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.model = SVC()
        self.tuner = SKLearnHyperparameterTuner(dataset)

    def fit(self):
        """
        Fit the support vector machine (SVM) model
        using hyperparameter tuning.

        The SVM model is tuned on the following
        hyperparameters:

        - 'C': Penalty parameter of the error term.

        - 'kernel': Specifies the kernel type to be used in the algorithm.

        - 'degree': Degree of the polynomial kernel function.

        - 'gamma': Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.

        """
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }
        tuner = SKLearnHyperparameterTuner(self.dataset)
        self.model = tuner.tune_model(self.model, param_grid)

    def predict(self):
        """
        Make predictions using the trained support
        vector machine (SVM) model.

        :return: The predicted labels.
        :rtype: np.ndarray
        """
        return self.model.predict(self.dataset.X_test)
