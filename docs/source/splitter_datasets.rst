Splitter Datasets
=================

Splitter Datasets are a set of predefined datasets in the Simple Classifier library that can be used for training and testing classifiers.
The `SplitterDataset` class serves as the base class for defining datasets and provides methods for loading and splitting the data.
Derived dataset classes inherit from :py:class:`~simpleclassifier.base.SplitterDataset` and implement the necessary methods for loading
specific datasets. Users can leverage these predefined datasets to quickly get started with classification tasks.

.. autoclass:: simpleclassifier.base.SplitterDataset
   :members:

Breast Cancer Dataset
#####################

.. autoclass:: simpleclassifier.splitter_datasets.BreastCancerDataset
   :members:
   :show-inheritance:
   :noindex:

Iris Dataset
###################

.. autoclass:: simpleclassifier.splitter_datasets.IrisDataset
   :members:
   :show-inheritance:
   :noindex:

Wine Dataset
###################

.. autoclass:: simpleclassifier.splitter_datasets.WineDataset
   :members:
   :show-inheritance:
   :noindex: