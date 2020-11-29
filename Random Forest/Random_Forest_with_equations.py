import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Decision_Tree.Decision_Tree_with_equations import _DecisionTreeClassifier


class RandomForestClassifier:

    def __init__(self, tree_number=5):
        self.tree_number = tree_number  # Number of tree that will be created
        self.tree_container = list()  # Tree container list

    def fit(self,x, y):
        # Initialize trees
        self.tree_container = list()
        random_ratio = np.linspace(0.2, 1, num=self.tree_number, endpoint=False)
        for ratio in random_ratio:
            tree = _DecisionTreeClassifier()
            x_sub, y_sub = self.create_subset(x=x, y=y, ratio=ratio)
            tree.fit(x_sub, y_sub)
            self.tree_container.append(tree)

    def prediction(self, x):
        tree_prediction_list = np.array(list(map(lambda tree: tree.prediction(x), self.tree_container)))
        prediction_list = list()
        for i in range(x.shape[0]):
            common_prediction = np.argmax(np.bincount(tree_prediction_list[:, i]))
            prediction_list.append(common_prediction)
        return prediction_list

    def accuracy(self, output_predicted, output_true):
        # Compare prediction and true labels and divide by numbers of prediction
        acc = np.sum(output_predicted == output_true) / len(output_predicted)

        return acc

    def create_subset(self, x, y, ratio):
        sample_num = x.shape[0]
        y = y.transpose()
        x_y = np.concatenate((x, y.reshape((1,len(y))).transpose()), axis=1)
        np.random.shuffle(x_y)
        x, y = np.array(x_y[int(sample_num * ratio):, :-1]), np.array(x_y[int(sample_num * ratio):, -1],dtype=np.int64)
        return x, y


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
    # Read csv
    x_train, y_train, x_test, y_test = dataset_preprocessing("iris.csv")
    best_accuracy = 0
    best_tree_num = 0
    # You may increase or decrease the tree number
    for k in range(2,16,2):
        random_forest_clf = RandomForestClassifier(tree_number=k)
        random_forest_clf.fit(x_train, y_train)
        y_pred = random_forest_clf.prediction(x_test)
        accuracy = random_forest_clf.accuracy(y_pred, y_test)
        # Because of shuffling on create_subset the accuracy may vary so if you want you can make comment line 41
        print(f"Accuracy of {k} tree is {round(accuracy,4)}")
        # Get the best accuracy and corresponding tree number
        if accuracy > best_accuracy:
            best_tree_num, best_accuracy = k, accuracy
            
    print(f"Best accuracy is {round(best_accuracy,4)} with {best_tree_num} tree number.")
    # Initialize and train best model
    random_forest_clf = RandomForestClassifier(tree_number=best_tree_num)
    random_forest_clf.fit(x_train, y_train)
    # Prediction
    label_list = random_forest_clf.prediction(x_test)
    # Color list
    color_list = ["r", "b", "g", "c", "y", "m", "w"] * 5
    # Initialize the subplot
    _, ax = plt.subplots(figsize=(8, 6))
    # Add labels
    for i, label in enumerate(label_list):
        ax.scatter(x_test[i][0], x_test[i][1], color=color_list[label_list[i]], linewidth=1)
    plt.grid()
    plt.show()
