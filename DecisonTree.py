import Tools


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for instance in dataset:
        if instance[index] < value:
            left.append(instance)
        else:
            right.append(instance)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [instance[-1] for instance in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(instance[-1] for instance in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for instance in dataset:
            groups = test_split(index, instance[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, instance[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    result = [instance[-1] for instance in group]
    return max(set(result), key=result.count)


class DecisonTree:
    def __init__(self, train_set, test_set, max_depth, min_size):
        self.train_set = train_set
        self.test_set = test_set
        self.max_depth = max_depth
        self.min_size = min_size

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = to_terminal(left), to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = to_terminal(left)
        else:
            node['left'] = get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = to_terminal(right)
        else:
            node['right'] = get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    # Build a decision tree
    def build_tree(self, train, max_depth, min_size):
        root = get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

    # Classify an object
    def classify(self, node, instance):
        if instance[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.classify(node['left'], instance)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.classify(node['right'], instance)
            else:
                return node['right']

    # Run decision tree
    def run(self):
        tree = self.build_tree(self.train_set, self.max_depth, self.min_size)
        predictions = []
        for instance in self.test_set:
            prediction = self.classify(tree, instance)
            predictions.append(prediction)
        y_true = [instance[-1] for instance in self.test_set]
        accuracy = Tools.get_accuracy(y_true, predictions)
        return accuracy
