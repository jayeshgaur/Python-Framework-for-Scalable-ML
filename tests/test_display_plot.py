import pytest
from unittest.mock import patch
from simpleclassifier.display import Display


def test_plot_no_classifiers():
    display = Display()
    results = {}
    with pytest.raises(ValueError):
        display.plot(
            results)  # Should raise ValueError as there are no classifiers


def test_plot_no_metric():
    display = Display()
    results = {'classifier1': {}}
    with pytest.raises(ValueError):
        display.plot(
            results)  # Should raise ValueError as there are no metrics


def test_plot_single_classifier_single_metric():
    display = Display()
    results = {'classifier1': {'metric1': 0.7}}
    with patch('matplotlib.pyplot.show'):
        # Ensures the function can run without raising exceptions
        try:
            display.plot(results)
        except Exception as e:
            pytest.fail(f"Unexpected Exception {e}")


def test_plot_multiple_classifiers_multiple_metrics():
    display = Display()
    results = {
        'classifier1': {
            'metric1': 0.7,
            'metric2': 0.8
        },
        'classifier2': {
            'metric1': 0.9,
            'metric2': 0.75
        }
    }
    with patch('matplotlib.pyplot.show'):
        # Ensures the function can run without raising exceptions
        try:
            display.plot(results)
        except Exception as e:
            pytest.fail(f"Unexpected Exception {e}")
