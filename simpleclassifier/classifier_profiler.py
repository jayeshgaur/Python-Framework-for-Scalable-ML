from simpleclassifier.base import Classifier
from simpleclassifier.display import Display
from simpleclassifier.profiler import Profiler


class ClassifierProfiler:
    """
    A class for profiling multiple classifiers using different
    metrics and displaying results.
    """

    def __init__(self, classifiers: list[Classifier], profiler: Profiler,
                 display: Display):
        """
        Constructor for the ClassifierProfiler class.

        :param classifiers: A list of classifiers to be profiled.
        :type classifiers: list[Classifier]
        :param profiler: An instance of the Profiler class used for profiling.
        :type profiler: Profiler
        :param display: An instance of the Display class used for showing the results.
        :type display: Display
        """
        self.results = {}
        self.classifiers = classifiers
        self.profiler = profiler
        self.display = display

    def train(self):
        """
        Train all classifiers on the dataset.
        """
        print("Training all classifiers...")
        for classifier in self.classifiers:
            print("-", classifier.__class__.__name__, end=" ")
            classifier.fit()
            print("[Done]")

    def profile_classifiers(self):
        """
        Run all profilers on all classifiers and display the results.
        """
        print("Profiling all classifiers...")
        for classifier in self.classifiers:
            print("-", classifier.__class__.__name__, end=" ")
            self.results[classifier.__class__.__name__] = self.profiler.run(
                classifier)
            print("[Done]")

    def display_results(self, display_format: str):
        """
        Display the profiling results.

        :param display_format: The format in which to display the results.
            Possible values are "dump", "json", and "plot".
        :type display_format: str
        """
        print("Displaying results...")
        if display_format == "dump":
            self.display.dump(self.results)
        elif display_format == "json":
            self.display.json(self.results)
        elif display_format == "plot":
            self.display.plot(self.results)
        else:
            raise ValueError(f"Invalid display format: {display_format}. "
                             "Please provide one of the following: 'dump', "
                             "'json', 'plot'.")
