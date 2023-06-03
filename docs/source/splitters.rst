Splitters
==========

Splitters are an essential component in the Simple Classifier library for dividing datasets into training and testing subsets.
The `Splitter` class serves as the base class, offering a consistent interface and defining core methods for implementing
various splitting strategies. Derived splitter classes inherit from :py:class:`~simpleclassifier.base.Splitter` and customize
the splitting logic to suit specific requirements. With the flexibility, extensibility, and integration provided by `Splitter`,
users can seamlessly incorporate different data splitting approaches into their machine learning workflows.

.. autoclass:: simpleclassifier.base.Splitter
   :members:

Percentage Splitter
###################

.. autoclass:: simpleclassifier.splitters.PercentageSplitter
   :members:
   :show-inheritance:
   :noindex:

Percentage with Shuffle Splitter
################################

.. autoclass:: simpleclassifier.splitters.PercentageShuffleSplitter
   :members:
   :show-inheritance:
   :noindex:

Percentage with Stratify Splitter
#################################

.. autoclass:: simpleclassifier.splitters.PercentageStratifiedSplitter
   :members:
   :show-inheritance:
   :noindex: