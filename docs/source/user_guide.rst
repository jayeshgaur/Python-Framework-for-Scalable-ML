User Guide
==========

This user guide provides information on how to use the Simple Classifier library and configure machine learning models.
Essentially, to use this software you can do two things:

- Run it as a CLI tool

- Build your own software on top of our :doc:`API <api_reference>`

Using it as a CLI tool
----------------------
One of the reasons this library was created was to enable machine learning beginners to experiment different
models and datasets and run it in one go. To do that you just need to create your own config file and put it in samples.
This is a sample YML file that you can use as a config:

.. code:: yaml

   classifier_names:
     - knn
     - lr
     - rf
     - svm
   dataset_name: diabetes
   splitting_strategy: percentage
   test_size: 0.50
   profile_metrics:
     - accuracy
     - precision
     - recall
     - f1
   display_format: dump

There are many ways you can customize the YML file, which is by changing the fields in the YML file by their codenames.
You can explore them below:

Classifiers
############
+--------------+------------------------------------------+
| Codename     | Classifier                               |
+==============+==========================================+
| ``knn``      | K-Nearest Neighbors                      |
+--------------+------------------------------------------+
| ``lr``       | Logistic Regression                      |
+--------------+------------------------------------------+
| ``rf``       | Random Forest Ensemble                   |
+--------------+------------------------------------------+
| ``svm``      | Support Vector Machine                   |
+--------------+------------------------------------------+

Datasets
########
+----------------------+---------------------------------+
| Codename             | Datasets                        |
+======================+=================================+
| ``breast_cancer``    | Breast Cancer Dataset           |
+----------------------+---------------------------------+
| ``iris``             | Iris Dataset                    |
+----------------------+---------------------------------+
| ``wine``             | Wine Dataset                    |
+----------------------+---------------------------------+

Splitting Strategy
##################
+---------------------------+-------------------------------+
| Codename                  | Datasets                      |
+===========================+===============================+
| ``percentage``            | Percentage                    |
+---------------------------+-------------------------------+
| ``percentage_shuffle``    | Percentage with Shuffle       |
+---------------------------+-------------------------------+
| ``percentage_stratified`` | Percentage with Stratify      |
+---------------------------+-------------------------------+

Test Size
#########
You can specify any test size between ``0.05`` and ``0.95``

Profile Metrics
###############
+---------------+-------------+
| Codename      | Metrics     |
+===============+=============+
| ``accuracy``  | Accuracy    |
+---------------+-------------+
| ``precision`` | Precision   |
+---------------+-------------+
| ``recall``    | Recall      |
+---------------+-------------+
| ``f1``        | F1 Score    |
+---------------+-------------+

Display Format
###############
+---------------+--------------------+
| Codename      | Dispaly            |
+===============+====================+
| ``dump``      | Prints to console  |
+---------------+--------------------+
| ``json``      | JSON format        |
+---------------+--------------------+
| ``plot``      | Bar Plot           |
+---------------+--------------------+