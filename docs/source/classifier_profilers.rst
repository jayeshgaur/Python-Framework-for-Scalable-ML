Classifier Profilers
====================

Profiling multiple classifiers using different metrics and displaying the results can provide valuable insights
into the performance and effectiveness of the classifiers. The :py:class:`~simpleclassifier.classifier_profiler.ClassifierProfiler`
class in the Simple Classifier library facilitates this profiling process. It takes a list of classifiers, an instance
of the :py:class:`~simpleclassifier.profiler.Profiler` class for profiling, and an instance of the
:py:class:`~simpleclassifier.display.Display` class for showing the results.

Profiling Multiple Classifiers
------------------------------

The ``train`` method trains all the classifiers on the dataset, while the ``profile_classifiers`` method runs the ``profilers`` on
each classifier and stores the results. The ``display_results method`` allows displaying the profiling results in different
formats specified by the ``display_format`` parameter.

By leveraging the capabilities of the :py:class:`~simpleclassifier.classifier_profiler.ClassifierProfiler`, users can
efficiently profile multiple classifiers, assess their performance using various metrics, and visualize the results for
better understanding and decision-making in machine learning tasks.

Classifier Profiler
###################

.. autoclass:: simpleclassifier.classifier_profiler.ClassifierProfiler
   :members:
   :show-inheritance:
   :noindex:

Profiler
###################

.. autoclass:: simpleclassifier.profiler.Profiler
   :members:
   :show-inheritance:
   :noindex:

Display
###################

.. autoclass:: simpleclassifier.display.Display
   :members:
   :show-inheritance:
   :noindex: