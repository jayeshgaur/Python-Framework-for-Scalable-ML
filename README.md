[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/8Z1lxzU_)

# Simple-Classifier

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fgabe-teaching-nyu-tandon%2Ffinalprojects23-simpleclassifier%2Fbadge%3Fref%3Dmain%26token%3D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&style=flat)](https://actions-badge.atrox.dev/gabe-teaching-nyu-tandon/finalprojects23-simpleclassifier/goto?ref=main&token=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)

Introducing “Simple Classifier”, a user-friendly library designed for individuals new to machine learning and seeking an accessible way to explore and experiment with various classification algorithms. This library streamlines the entire machine learning process, from data preparation and preprocessing to training, evaluation, and comparison of different algorithms, without requiring extensive coding knowledge.

Simple Classifier not only allows users to easily customize the splitting method for their specific needs, but also performs hyperparameter tuning to find the most optimal parameters for each model. With a comprehensive performance comparison, users can effortlessly identify the best algorithm for their dataset, making this library an essential tool for those starting their journey in machine learning.

Check out the [Getting Started](https://storage.googleapis.com/simple-classifier-documentation/getting_started.html) section for further information.

## Getting Started

This project needs at least Python 3.9. Create a virtual environment with your favorite tool (e.g. `conda` or `virtualenv`), and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Before running any commands, it is crucial to set up the project’s root directory in your PYTHONPATH environment variable to ensure the correct functioning of the Simple Classifier library. This is necessary because the project uses absolute import paths to pass the CI test.

Required Setup Steps:

- Open a terminal in the project root directory.

- For Linux and Mac users, run the following command to add the project root directory to your PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

By following these steps, you can avoid import errors related to the Simple Classifier library. Now, you should be able to use the library without any issues. Make sure to perform this setup before running any commands or executing any code related to the Simple Classifier library.

`simpleclassifier` can be executed as a cli tool. To run it, use the following command:

```bash
python simpleclassifier -y <PATH_TO_YAML_CONFIG_FILE>
```

Sample YAML configuration files can be found in [samples](samples/). For example, to run the `knn` model on the `diabetes` dataset, use the following command:

```yaml
classifier_names:
  - knn
dataset_name: diabetes
splitting_strategy: percentage
test_size: 0.50
profile_metrics:
  - accuracy
display_format: dump
```

To run it, use the following command:

```bash
python simpleclassifier -y samples/knn_breast_cancer_percentage_50_accuracy_dump.yml
```

And watch the magic happens!

```
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
```

## Documentation

The official documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/)
and hosted on this [website](https://storage.googleapis.com/simple-classifier-documentation/index.html).

You can build the documentation source files by running these commands

```bash
cd docs
make html
```

## Final Report

You can find the final report for this project 
[here](https://drive.google.com/file/d/1sxz3DGYw98rRh3lRoN5vcMBC7zlVnV-C/view?usp=sharing) 
or you can download it from the root directory of this repository.

## Contributors

- Junda Ai [ja4426@nyu.edu](ja4426@nyu.edu)
- Brian Catraguna [bmc9858@nyu.edu](bmc9858@nyu.edu)
- Jayesh Gaur [jjg9777@nyu.edu](jjg9777@nyu.edu)
- Shreyash Gondane [sg6874@nyu.edu](sg6874@nyu.edu)
- Gunjan Hirenkumar Dayani [gd2275@nyu.edu](gd2275@nyu.edu)
