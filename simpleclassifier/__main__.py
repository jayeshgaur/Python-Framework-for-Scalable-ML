import argparse
import yaml

from dataclasses import dataclass
from simpleclassifier.display import Display
from simpleclassifier.factory import ClassifierFactory
from simpleclassifier.factory import SplitterDatasetFactory
from simpleclassifier.factory import SplitterFactory
from simpleclassifier.profiler import Profiler
from simpleclassifier.classifier_profiler import ClassifierProfiler

from simpleclassifier.classifiers import *  # noqa
from simpleclassifier.splitter_datasets import *  # noqa
from simpleclassifier.splitters import *  # noqa


@dataclass
class Config:
    classifier_names: list[str]
    dataset_name: str
    splitting_strategy: str
    test_size: float
    profile_metrics: list[str]
    display_format: str


def main():
    config = parse_args()

    splitter = SplitterFactory.create_instance(config.splitting_strategy,
                                               test_size=config.test_size)

    dataset = SplitterDatasetFactory.create_instance(config.dataset_name,
                                                     splitter=splitter)

    classifiers = [
        ClassifierFactory.create_instance(classifier_name, dataset=dataset)
        for classifier_name in config.classifier_names
    ]
    profiler = Profiler(config.profile_metrics)
    display = Display()
    classifier_profiler = ClassifierProfiler(classifiers, profiler, display)

    classifier_profiler.train()
    classifier_profiler.profile_classifiers()
    classifier_profiler.display_results(config.display_format)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        "simple classifier profiler",
        description="""
A simple program to easily train, test, profile, and compare different
classifiers on multiple datasets""",
    )
    parser.add_argument("-y",
                        "--yml",
                        help="path to YAML configuration file",
                        required=True)
    args = parser.parse_args()

    try:
        with open(args.yml, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        if data is None:
            data = {}
        return Config(
            data.get("classifier_names", []),
            data.get("dataset_name", "breast_cancer"),
            data.get("splitting_strategy") or "percentage",
            data.get("test_size") or 0.2,
            data.get("profile_metrics", "accuracy"),
            data.get("display_format", "dump"),
        )
    except FileNotFoundError:
        print(f"configuration file '{args.yml}' was not found")
        parser.print_usage()
        parser.exit()


if __name__ == "__main__":
    main()
