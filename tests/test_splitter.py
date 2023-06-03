from simpleclassifier.splitters import (
    PercentageSplitter,
    PercentageShuffleSplitter,
    PercentageStratifiedSplitter,
)

import pytest
import numpy as np


@pytest.fixture
def x_y_data():
    mock_X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    mock_y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return mock_X, mock_y


@pytest.mark.parametrize("test_size", [0.3, 0.5, 1.0, 10.0])
def test_splitter_test_size(test_size):
    if 0.05 <= test_size <= 0.95:
        splitter = PercentageSplitter(test_size)
        assert splitter.test_size == test_size
    else:
        with pytest.raises(ValueError):
            splitter = PercentageSplitter(test_size)


@pytest.mark.parametrize("random_state", [42, 30, 14])
def test_random_state(x_y_data, random_state):
    mock_X, mock_y = x_y_data

    first_splitter = PercentageSplitter(test_size=0.3,
                                        random_state=random_state)
    first_split = first_splitter.split_data(mock_X, mock_y)
    second_splitter = PercentageSplitter(test_size=0.3,
                                         random_state=random_state)
    second_split = second_splitter.split_data(mock_X, mock_y)

    np.testing.assert_array_equal(first_split[0], second_split[0])
    np.testing.assert_array_equal(first_split[1], second_split[1])


def test_percentage_splitter(x_y_data):
    mock_X, mock_y = x_y_data

    splitter = PercentageSplitter(test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split_data(mock_X, mock_y)

    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 4


def test_percentage_shuffle_splitter(x_y_data):
    mock_X, mock_y = x_y_data

    shuffle_splitter = PercentageShuffleSplitter(test_size=0.3,
                                                 random_state=42)
    (
        X_train_shuffled,
        X_test_shuffled,
        y_train_shuffled,
        y_test_shuffled,
    ) = shuffle_splitter.split_data(mock_X, mock_y)

    assert X_train_shuffled.shape[0] == 8
    assert X_test_shuffled.shape[0] == 4

    non_shuffle_splitter = PercentageSplitter(test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = non_shuffle_splitter.split_data(
        mock_X, mock_y)

    assert not np.array_equal(X_train, X_train_shuffled)
    assert not np.array_equal(X_test, X_test_shuffled)


def test_percentage_stratified_splitter(x_y_data):
    mock_X, mock_y = x_y_data

    splitter = PercentageStratifiedSplitter(test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split_data(mock_X, mock_y)

    # Check that the sizes of the training and testing sets are correct
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 4

    # Check that the proportion of class 0 is preserved in the training and testing sets
    prop_train_0 = np.mean(y_train == 0).item()
    prop_test_0 = np.mean(y_test == 0).item()
    assert pytest.approx(prop_train_0, abs=0.05) == 0.5
    assert pytest.approx(prop_test_0, abs=0.05) == 0.5

    # Check that the proportion of class 1 is preserved in the training and testing sets
    prop_train_1 = np.mean(y_train == 1).item()
    prop_test_1 = np.mean(y_test == 1).item()
    assert pytest.approx(prop_train_1, abs=0.05) == 0.5
    assert pytest.approx(prop_test_1, abs=0.05) == 0.5
