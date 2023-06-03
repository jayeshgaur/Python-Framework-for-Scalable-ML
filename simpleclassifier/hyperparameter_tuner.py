from simpleclassifier.base import SplitterDataset

from sklearn.model_selection import GridSearchCV


class SKLearnHyperparameterTuner:
    """
    A helper class to tune SKLearn models.

    """

    def __init__(self, dataset: SplitterDataset):
        """
        Initialize the HyperparameterTuner.

        :param dataset: The dataset object used for training and testing the model.
        :type dataset: SplitterDataset
        """
        self.dataset = dataset

    def tune_model(self, model, param_grid):
        """
        Tune the hyperparameters of a model using grid search.

        :param model: The model to tune.
        :param param_grid: Dictionary specifying the hyperparameters to search.
        :type param_grid: dict
        :return: The best estimator found by grid search.
        """
        grid_search = GridSearchCV(model, param_grid)
        grid_search.fit(self.dataset.X_train, self.dataset.y_train)
        return grid_search.best_estimator_
