from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Profiler:
    """
    Profiler class to calculate various metrics for a given classifier.

    This application deals with "accuracy", "precision", "recall", and "f1" metrics.
    """

    def __init__(self, metrics: list[str]):
        """
        Constructor for Profiler class.

        :param metrics: List of metrics to calculate.
        :type metrics: list[str]
        """
        self.metrics = metrics

    def run(self, classifier):
        """
        Runs the profiler on the given classifier.

        :param classifier: Instance of the classifier to run the profiler on.
        :return: A dictionary containing the results of the profiler.
        :rtype: dict
        """

        # Predict the test set labels using the given classifier
        y_pred = classifier.predict()
        results = {}

        if self.metrics:
            # Calculate accuracy if requested
            if "accuracy" in self.metrics:
                results["accuracy"] = accuracy_score(classifier.dataset.y_test,
                                                     y_pred)
            # Calculate precision if requested
            if "precision" in self.metrics:
                results["precision"] = precision_score(
                    classifier.dataset.y_test, y_pred, average="weighted")
            # Calculate recall if requested
            if "recall" in self.metrics:
                results["recall"] = recall_score(classifier.dataset.y_test,
                                                 y_pred,
                                                 average="weighted",
                                                 zero_division=0)
            # Calculate F1 score if requested
            if "f1" in self.metrics:
                results["f1"] = f1_score(classifier.dataset.y_test,
                                         y_pred,
                                         average="weighted")
        else:
            raise RuntimeError("Please specify at least one metric "
                               "to evaluate the classifier(s)")

        return results
