import pandas as pd
import numpy as np


# Representation of Root or leaf Node
class Node:

    def __init__(self, threshold=None, value=None, feature_index=None, left_side=None, right_side=None):
        self.threshold = threshold  # Threshold value to separate right and left
        self.value = value  # Will not be None if it is leaf node
        self.feature_index = feature_index  # Index of each node
        self.left_side = left_side  # Left Side
        self.right_side = right_side  # Right Side


# This Decision Tree is used as classifier
class _DecisionTreeClassifier:

    def __init__(self):
        self.features = None  # column number of x
        self.root = None  # it will contain all nodes

    # There will be max n criteria and pow(2, n) nodes
    # The aim is finding the best separating value for each criteria
    def fit(self, x, y):
        self.features = x.shape[1]
        self.root = list()
        # Will hold variances of x columns
        variance_list = [np.unique(x[:, i]) for i in range(self.features)]
        len_list = list()
        # The least varied column will contain the most labels in nodes
        for k in range(len(variance_list)):
            len_list.append(len(variance_list[k]))
        # Sort the list
        column_sort_list = np.argsort(len_list)

        # Column by column the tree will be generated
        self.root = self.generate_root(x, y, column_sort_list, 0)

    def generate_root(self, x, y, sorted_args, feature):

        if len(np.unique(y)) == 1 or feature == self.features:
            common_label = np.bincount(y)

            return Node(value=np.argmax(common_label))
        # Get the best threshold value to divide
        threshold = self.find_threshold(x, y, sorted_args[feature])
        # Separate left and right side indexes depending on threshold
        left_side_indexes = [i for i, num in enumerate(x[:, sorted_args[feature]]) if num <= threshold]
        right_side_indexes = [i for i, num in enumerate(x[:, sorted_args[feature]]) if num > threshold]
        # Continue to generating nodes recursively
        left_side = self.generate_root(x[left_side_indexes, :], y[left_side_indexes], sorted_args, feature + 1)
        right_side = self.generate_root(x[right_side_indexes, :], y[right_side_indexes],sorted_args, feature + 1)
        return Node(feature_index=sorted_args[feature], threshold=threshold,
                    left_side=left_side, right_side=right_side)

    # Calculate the entropy of the column
    def calculate_entropy(self, x):
        # E =  sum(- (p(x) * log2(p(x)))
        common_labels = np.bincount(x)
        mean_labels = common_labels / len(x)
        # Because of log doesnt get - or 0 as input label is checked
        entropy = np.sum([- (label * np.log2(label)) for label in mean_labels if label > 0])
        return entropy

    # Walk on nodes deeper and deeper
    def prediction(self, x):
        # Recursive
        predict = [self.walk_on_nodes(x_row, self.root) for x_row in x]
        return predict

    def walk_on_nodes(self, x, node):
        # Check if the node is leaf
        if node.value is not None:
            return node.value
        # Classify the input recursively
        if x[node.feature_index] > node.threshold:
            return self.walk_on_nodes(x, node.right_side)

        return self.walk_on_nodes(x, node.left_side)

    def accuracy(self, output_predicted, output_true):
        # All predictions
        comparison_list = [output_predicted[k] == output_true[k] for k in range(len(output_predicted))]
        # Get Percentage of correct predictions
        acc = comparison_list.count(True) / len(comparison_list)
        return acc

    def find_threshold(self, x, y, arg):
        # Initialize the gain to
        best_gain = -0.1
        thresholds = np.unique(x[:, arg])
        separating_threshold = None
        for threshold in thresholds:
            calculated_gain = self.calculate_gain(x[:, arg], y, threshold)
            if calculated_gain > best_gain:
                best_gain, separating_threshold = calculated_gain, threshold

        return separating_threshold

    def calculate_gain(self, x, y, separating_threshold):
        # Separate to right and left with respect to separating_threshold
        left_side_indexes = [i for i, num in enumerate(x) if num <= separating_threshold]
        right_side_indexes = [i for i, num in enumerate(x) if num > separating_threshold]
        # Calculate entropy of both nodes
        left_side_entropy = self.calculate_entropy(y[left_side_indexes])
        right_side_entropy = self.calculate_entropy(y[right_side_indexes])
        node_entropy = (len(left_side_indexes) / len(y)) * left_side_entropy +\
                        (len(right_side_indexes) / len(y)) * right_side_entropy
        # Calculate the gain
        calculated_gain = self.calculate_entropy(y) - node_entropy
        return calculated_gain


def dataset_preprocessing(csv_file="Breast_cancer_data.csv"):
    # Reading csv file
    dataset = pd.read_csv(csv_file)
    # Getting columns for x and y arrays
    # Assuming last column is output (y)
    dataset_array = dataset.to_numpy()
    np.random.shuffle(dataset_array)  # Uncomment this this row if dataset labels are sequential
    x, y = dataset_array[:, :-1], dataset_array[:, -1]
    y = np.array(y, dtype=np.int64)
    test_perc = 0.2  # test percentage
    test_size = int(test_perc * len(x))  # test size

    # Splitting to test and train arrays
    x_train, y_train, x_test, y_test = x[:-test_size], y[:-test_size], x[-test_size:], y[-test_size:]
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # Initialize model
    decision_tree = _DecisionTreeClassifier()
    # Get data
    x_train, y_train, x_test, y_test = dataset_preprocessing("iris.csv")
    # Fit the model
    decision_tree.fit(x_train, y_train)
    acc = decision_tree.accuracy(decision_tree.prediction(x_test), y_test)
    print(f"Accuracy of the tree is {round(acc,4)}")
    
    
    
