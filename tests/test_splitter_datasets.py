from simpleclassifier.splitter_datasets import (
    BreastCancerDataset,
    IrisDataset,
    WineDataset,
)
from simpleclassifier.splitters import PercentageSplitter

import pytest


@pytest.fixture
def splitter():
    return PercentageSplitter(test_size=0.3, random_state=42)


def test_load_breast_cancer(splitter):
    dataset = BreastCancerDataset(splitter=splitter)
    assert dataset.X_train.shape == (398, 30)
    assert dataset.X_test.shape == (171, 30)
    assert dataset.y_train.shape == (398, )
    assert dataset.y_test.shape == (171, )


def test_load_iris(splitter):
    dataset = IrisDataset(splitter=splitter)
    assert dataset.X_train.shape == (105, 4)
    assert dataset.X_test.shape == (45, 4)
    assert dataset.y_train.shape == (105, )
    assert dataset.y_test.shape == (45, )


def test_load_wine(splitter):
    dataset = WineDataset(splitter=splitter)
    assert dataset.X_train.shape == (124, 13)
    assert dataset.X_test.shape == (54, 13)
    assert dataset.y_train.shape == (124, )
    assert dataset.y_test.shape == (54, )
