from math import sqrt
import Tools


class KNN:
    def __init__(self, train_set, test_set, k):
        self.train_set = train_set
        self.test_set = test_set
        self.k = k

    # Locate the most similar neighbors
    def calculate_neighbors(self, train, test_row):
        distances = list()
        for instance in train:
            dist = Tools.euclidean_distance(test_row, instance)
            distances.append((instance, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    # Classify an object
    def classify(self, train_set, instance):
        neighbors = self.calculate_neighbors(train_set, instance)
        output_values = [instance[-1] for instance in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    # Run KNN and return the accuracy
    def run(self):
        predications = []
        for instance in self.test_set:
            result = self.classify(self.train_set, instance)
            predications.append(result)
        y_true = [instance[-1] for instance in self.test_set]
        accuracy = Tools.get_accuracy(y_true, predications)
        return accuracy
