Release Notes
=============

Version 1.0.0
-------------
Initial Release
###############

- Introduced the Splitter class as the base class for dividing datasets into training and testing subsets.
- Added derived splitter classes: PercentageSplitter, PercentageShuffleSplitter, and PercentageStratifiedSplitter for implementing different splitting strategies.
- Included SplitterDataset class as the base class for predefined datasets and provided methods for loading and splitting the data.
- Implemented BreastCancerDataset, IrisDataset, and WineDataset as predefined datasets for classification tasks.
- Added the Classifier class as the base class for training and predicting labels for datasets.
- Included derived classifier classes: KNNClassifier, LogisticRegressionClassifier, CustomRandomForestClassifier, and SVMClassifier.
- Introduced command-line interface (CLI) for running the Simple Classifier library with YAML configuration files.
- Provided sample YAML configuration files in the samples directory for users to create their own machine learning configurations.
- Supported profile metrics such as accuracy for evaluating classifier performance.
- Enabled different display formats, including "dump", for presenting the results of trained classifiers.
- Implemented logging to capture important information and progress during training and profiling processes.

We hope you find the Simple Classifier library useful for your machine learning tasks. Stay tuned for future updates and enhancements!

For more information and detailed usage instructions, please refer to the library documentation.