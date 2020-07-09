import Tools


# Calculate the probabilities of predicting each class for a given row
def estimate_class_probabilities(stats, instance):
    total_rows = sum([stats[label][0][2] for label in stats])
    probabilities = dict()
    for lbl, class_stats in stats.items():
        probabilities[lbl] = stats[lbl][0][2] / float(total_rows)
        for i in range(len(class_stats)):
            mean, stdev, _ = class_stats[i]
            probabilities[lbl] *= Tools.estimate_probability(instance[i], mean, stdev)
    return probabilities


# Calculate the mean, standard deviation and count for each column in a dataset
def get_stats_of_dataset(dataset):
    stats = [(Tools.mean(column), Tools.stdev(column), len(column)) for column in zip(*dataset)]
    del (stats[-1])
    return stats


class NaiveBayes:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    # Split the dataset by class values
    def split_classes(self, dataset):
        result = dict()
        for i in range(len(dataset)):
            instance = dataset[i]
            lbl = instance[-1]
            if (lbl not in result):
                result[lbl] = list()
            result[lbl].append(instance)
        return result

    # Calculate statistics for each class
    def get_stats_of_class(self, dataset):
        separated = self.split_classes(dataset)
        stats = dict()
        for lbl, rows in separated.items():
            stats[lbl] = get_stats_of_dataset(rows)
        return stats

    # Classify an object
    def classify(self, stats, instance):
        probabilities = estimate_class_probabilities(stats, instance)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    # Run Naive Bayes
    def run(self):
        stats = self.get_stats_of_class(self.train_set)
        predictions = list()
        for instance in self.test_set:
            output = self.classify(stats, instance)
            predictions.append(output)
        y_true = [instance[-1] for instance in self.test_set]
        accuracy = Tools.get_accuracy(y_true, predictions)
        return accuracy
