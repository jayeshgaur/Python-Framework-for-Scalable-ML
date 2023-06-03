from simpleclassifier.classifiers import (KNNClassifier,
                                          LogisticRegressionClassifier,
                                          RandomForestEnsembleClassifier,
                                          SVMClassifier)
from simpleclassifier.splitter_datasets import SplitterDataset
from simpleclassifier.splitters import PercentageSplitter

import pytest
from sklearn.datasets import make_classification
import warnings
from sklearn.exceptions import ConvergenceWarning
from functools import wraps


def ignore_convergence_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            return func(*args, **kwargs)

    return wrapper


@pytest.fixture
def dataset():
    class FakeDataset(SplitterDataset):
        def load_data(self):
            return make_classification(n_samples=100,
                                       n_features=10,
                                       random_state=42)

    return FakeDataset(splitter=PercentageSplitter(test_size=0.3))


@ignore_convergence_warnings
def test_knn_clf(dataset):
    clf = KNNClassifier(dataset=dataset)
    clf.fit()
    predictions = clf.predict()
    assert len(predictions) == len(dataset.y_test)


@ignore_convergence_warnings
def test_logistic_regression_classifier(dataset):
    lr_classifier = LogisticRegressionClassifier(dataset=dataset)
    lr_classifier.fit()
    predictions = lr_classifier.predict()
    assert len(predictions) == len(dataset.y_test)


def test_random_forest_classifier(dataset):
    rf_classifier = RandomForestEnsembleClassifier(dataset=dataset)
    rf_classifier.fit()
    predictions = rf_classifier.predict()
    assert len(predictions) == len(dataset.y_test)


def test_svm_classifier(dataset):
    svm_classifier = SVMClassifier(dataset=dataset)
    svm_classifier.fit()
    predictions = svm_classifier.predict()
    assert len(predictions) == len(dataset.y_test)
