Classifiers
===========

Classifiers are fundamental components in the Simple Classifier library for training and predicting labels for datasets.
The `Classifier` class serves as the base class, providing a consistent interface and essential methods for implementing
various classification algorithms. Derived classifier classes inherit from :py:class:`~simpleclassifier.base.Classifier`
and customize the training and prediction logic to suit specific requirements. With the flexibility, extensibility, and integration
provided by `Classifier`, users can seamlessly build and evaluate different classification models for their machine learning tasks.

.. autoclass:: simpleclassifier.base.Classifier
   :members:


K Nearest Neighbors
###################

.. autoclass:: simpleclassifier.classifiers.KNNClassifier
   :members:
   :show-inheritance:
   :noindex:

Logistic Regression
###################

.. autoclass:: simpleclassifier.classifiers.LogisticRegressionClassifier
   :members:
   :show-inheritance:
   :noindex:

Random Forest Ensemble
######################

.. autoclass:: simpleclassifier.classifiers.RandomForestEnsembleClassifier
   :members:
   :show-inheritance:
   :noindex:

Support Vector Machine
#######################

.. autoclass:: simpleclassifier.classifiers.SVMClassifier
   :members:
   :show-inheritance:
   :noindex: