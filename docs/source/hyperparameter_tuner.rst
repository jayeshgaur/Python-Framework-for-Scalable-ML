Hyperparameter Tuner
====================

Hyperparameter tuning is a crucial step in machine learning model development. It involves finding the best
combination of hyperparameters for a given model to optimize its performance on a specific task or dataset.
Hyperparameters are configuration settings that are not learned from the data but are set by the user before training the model.

Grid Search
###########

Grid search is a popular technique for hyperparameter tuning. It systematically searches through a predefined grid of
hyperparameter values to find the combination that yields the best performance. It works by evaluating the model's
performance on a validation set for each combination of hyperparameters and selecting the one with the highest
performance metric.

The :py:class:`~simpleclassifier.hyperparameter_tuner.SKLearnHyperparameterTuner` class in the Simple Classifier library
utilizes grid search to tune the hyperparameters of SKLearn models. It takes a dataset object as input and performs grid
search on the specified model and parameter grid. The `tune_model` method executes the grid search process, evaluating
the model's performance for each hyperparameter combination and returning the best estimator found by grid search.

SKLearn Model Hyperparameter Tuning
###################################
By leveraging the hyperparameter tuning capabilities provided by the SKLearnHyperparameterTuner, users can enhance the
performance and effectiveness of their machine learning models, leading to better predictions and more reliable results.

.. autoclass:: simpleclassifier.hyperparameter_tuner.SKLearnHyperparameterTuner
   :members:
   :show-inheritance:
   :noindex: