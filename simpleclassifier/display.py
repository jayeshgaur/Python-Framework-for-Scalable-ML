import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib.colors import ListedColormap


class Display:
    """
    A class for displaying profiling results.

    This class provides methods to dump the data to the console,
    return it as a JSON string, and plot the results
    for each classifier and metric.

    """

    def dump(self, results):
        """
        Dump the profiling results to the console.

        This method groups the results by metric and displays them in a tabular format.

        :param results: The profiling results for each classifier and metric.
        :type results: dict

        :return: None
        """

        # Group results by metric
        metrics = {}
        for classifier in results:
            for metric, score in results[classifier].items():
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append((classifier, score))

        # Print results by metric
        print("=====================================")
        for metric, scores in metrics.items():
            print("-------------------------------------")
            print(metric)
            for classifier, score in scores:
                print(f"- {classifier}: {score}")
        print("=====================================")

    def json(self, results):
        """
        Return the profiling results as a JSON string.

        :param results: The profiling results for each classifier and metric.
        :type results: dict

        :return: The profiling results as a JSON string.
        :rtype: str
        """
        print(json.dumps(results, indent=4))

    def plot(self, results):
        """
        Plot the profiling results for each classifier and metric.

        This method creates a bar plot showing the scores for
        each classifier and metric.

        :param results: The profiling results for each classifier and metric.
        :type results: dict

        :return: None
        """

        # Calculate the number of classifiers and create a colormap
        num_classifiers = len(results)
        if num_classifiers == 0:
            raise ValueError("No classifiers in results.")
        cmap = ListedColormap(
            plt.cm.turbo(np.linspace(  # type: ignore
                0, 1, num_classifiers)))

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract the classifiers, metrics, and scores from the results
        classifiers = list(results.keys())
        num_metrics = len(list(results.values())[0])
        if num_metrics == 0:
            raise ValueError("No profiler metrics given by user.")
        x_labels = list(results.values())[0].keys()

        # Plot the scores for each classifier and metric
        for i, classifier in enumerate(classifiers):
            scores = [results[classifier][metric] for metric in x_labels]
            ax.bar(
                [j + i * (0.8 / len(classifiers)) for j in range(num_metrics)],
                scores,
                width=0.8 / len(classifiers),
                color=cmap(i),
                label=classifier)

        # Set the plot properties
        ax.set_ylabel('Score')
        ax.set_xlabel('Metrics')
        ax.set_title('Classifier Performance Comparison')
        ax.set_xticks([i + 0.4 for i in range(num_metrics)])
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        # Display the plot
        plt.show()
