from sklearn.datasets import make_classification

from simpleclassifier.factory import (ClassifierFactory,
                                      SplitterDatasetFactory, SplitterFactory)
from simpleclassifier.base import Classifier, SplitterDataset, Splitter
from simpleclassifier.classifiers import (KNNClassifier,
                                          LogisticRegressionClassifier,
                                          RandomForestEnsembleClassifier,
                                          SVMClassifier)
from simpleclassifier.splitter_datasets import (BreastCancerDataset,
                                                IrisDataset, WineDataset)
from simpleclassifier.splitters import (PercentageSplitter,
                                        PercentageShuffleSplitter,
                                        PercentageStratifiedSplitter)

import pytest


@pytest.fixture
def dataset():
    class FakeDataset(SplitterDataset):
        def load_data(self):
            return make_classification(n_samples=100,
                                       n_features=10,
                                       random_state=42)

    return FakeDataset(splitter=PercentageSplitter(test_size=0.3))


@pytest.fixture
def splitter():
    return PercentageSplitter(test_size=0.3, random_state=42)


@pytest.mark.parametrize("codename",
                         ["knn", "lr", "rf", "svm", "invalid_name"])
def test_create_classifier_instance(codename, dataset):
    factory = ClassifierFactory()

    if codename == "invalid_name":
        with pytest.raises(ValueError) as excinfo:
            factory.create_instance(codename, dataset=dataset)

        assert str(
            excinfo.value) == f"No classifierfactory registered for {codename}, " \
                              f"registered classifierfactorys are: ['knn', 'lr', " \
                              f"'rf', 'svm']"
    else:
        instance = factory.create_instance(codename, dataset=dataset)
        assert isinstance(instance, Classifier)


def test_classifier_registration():
    factory = ClassifierFactory()
    assert factory.registry == {
        "knn": KNNClassifier,
        "lr": LogisticRegressionClassifier,
        "rf": RandomForestEnsembleClassifier,
        "svm": SVMClassifier
    }

    with pytest.raises(TypeError):

        @factory.register("invalid")
        class InvalidClassifier:
            pass

    with pytest.raises(ValueError):

        @factory.register("knn")
        class DuplicateKNNClassifier(Classifier):
            pass


@pytest.mark.parametrize("codename",
                         ["breast_cancer", "iris", "wine", "invalid_name"])
def test_create_splitter_dataset_instance(codename, splitter):
    factory = SplitterDatasetFactory()

    if codename == "invalid_name":
        with pytest.raises(ValueError) as excinfo:
            factory.create_instance(codename, splitter=splitter)

        assert str(
            excinfo.value) == f"No splitterdatasetfactory registered for {codename}, " \
                              f"registered splitterdatasetfactorys are: [" \
                              f"'breast_cancer', 'iris', 'wine']"
    else:
        instance = factory.create_instance(codename, splitter=splitter)
        assert isinstance(instance, SplitterDataset)


def test_splitter_dataset_registration():
    factory = SplitterDatasetFactory()
    assert factory.registry == {
        "breast_cancer": BreastCancerDataset,
        "iris": IrisDataset,
        "wine": WineDataset
    }

    with pytest.raises(TypeError):

        @factory.register("invalid")
        class InvalidSplitterDataset:
            pass

    with pytest.raises(ValueError):

        @factory.register("breast_cancer")
        class DuplicateBreastCancerSplitterDataset(SplitterDataset):
            pass


@pytest.mark.parametrize("codename", [
    "percentage", "percentage_shuffle", "percentage_stratified", "invalid_name"
])
def test_create_splitter_instance(codename, test_size=0.3):
    factory = SplitterFactory()

    if codename == "invalid_name":
        with pytest.raises(ValueError) as excinfo:
            factory.create_instance(codename, test_size=test_size)

        assert str(
            excinfo.value) == f"No splitterfactory registered for {codename}, " \
                              f"registered splitterfactorys are: ['percentage', " \
                              f"'percentage_shuffle', 'percentage_stratified']"
    else:
        instance = factory.create_instance(codename, test_size=test_size)
        assert isinstance(instance, Splitter)


def test_splitter_registration():
    factory = SplitterFactory()
    assert factory.registry == {
        "percentage": PercentageSplitter,
        "percentage_shuffle": PercentageShuffleSplitter,
        "percentage_stratified": PercentageStratifiedSplitter
    }

    with pytest.raises(TypeError):

        @factory.register("invalid")
        class InvalidSplitter:
            pass

    with pytest.raises(ValueError):

        @factory.register("percentage")
        class DuplicatePercentageSplitter(Splitter):
            pass
