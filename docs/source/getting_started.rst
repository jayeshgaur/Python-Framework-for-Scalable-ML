Getting Started
===============

Setting up Python Environment
-----------------------------

This project needs at least Python 3.9. Create a virtual environment
with your favorite tool (e.g. ``conda`` or ``virtualenv``), and install
the dependencies:

.. code:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

.. important::

   Before running any commands, it is crucial to set up the project's root directory in your
   ``PYTHONPATH`` environment variable to ensure the correct functioning of the Simple Classifier library.
   This is necessary because the project uses absolute import paths to pass the CI test.

Required Setup Steps:

1. Open a terminal in the project root directory.

2. For Linux and Mac users, run the following command to add the project root directory to your PYTHONPATH:

.. code-block:: bash

   ❯ export PYTHONPATH=$PYTHONPATH:$(pwd)

By following these steps, you can avoid import errors related to the Simple Classifier library.
Now, you should be able to use the library without any issues. Make sure to perform this setup before running any
commands or executing any code related to the Simple Classifier library.

Help
----
To use Simple Classifier, you can enter commands in the terminal. The command python simpleclassifier -h will provide a brief overview of the program's usage and its options. See the example below:

.. code-block:: none

   ❯ python simpleclassifier -h
   usage: simple classifier profiler [-h] -y YML

   A simple program to easily train, test, profile, and compare different classifiers on multiple datasets

   options:
   -h, --help show this help message and exit
   -y YML, --yml YML path to YAML configuration file

In this context, the -h or --help option displays a help message with a summary of how to use the program and exits.
The -y YML or --yml YML option allows you to specify a path to your YAML configuration file. This YAML file should
contain all the necessary configurations for the classifier including the dataset, split method, and the classifiers
to be used. This makes Simple Classifier a flexible tool, allowing customization of its operation to suit your
specific needs.

Running Command Line Interface
------------------------------

``simpleclassifier`` can be executed as a cli tool. To run it, use the
following command:

.. code:: bash

   ❯ python simpleclassifier -y <PATH_TO_YAML_CONFIG_FILE>

Create your own Machine Learning Configuration
----------------------------------------------

Sample YAML configuration files can be found in ``samples`` directory.
For example, to run the ``knn`` model on the ``diabetes`` dataset you can use
``samples/knn_breast_cancer_percentage_50_accuracy_dump.yml`` file:

.. code:: yaml

   classifier_names:
     - knn
   dataset_name: diabetes
   splitting_strategy: percentage
   test_size: 0.50
   profile_metrics:
     - accuracy
   display_format: dump

.. tip::

   Check out our :doc:`user_guide` for more detailed information on available configuration options.

To run it, use the following command:

.. code:: bash

   ❯ python simpleclassifier -y samples/knn_breast_cancer_percentage_50_accuracy_dump.yml

And watch the magic happens!

.. code-block:: none

   Training all classifiers...
   - KNNClassifier [Done]
   Profiling all classifiers...
   - KNNClassifier [Done]
   Displaying results...
   =====================================
   -------------------------------------
   accuracy
   - KNNClassifier: 0.968421052631579
   =====================================