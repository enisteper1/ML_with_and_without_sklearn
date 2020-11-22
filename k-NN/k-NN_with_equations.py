import numpy as np
from math import sqrt
import pandas as pd


class K_Nearest_Neighbours:
    def __init__(self, neighbours=5):
        self.neighbours = neighbours

    # Get distances between points with euclidean equation
    def euclidean_distance(self, p1, p2):

        # Calculate sum of distances for all dimensions of p1 and p2
        distance = sum((p1[h] - p2[h])**2 for h in range(len(p1)))
        # Get square root to finish euclidean distance formula
        distance = sqrt(distance)

        return distance

    def fit(self,x, y):
        self.x = x
        self.y = y
    # Compares predicted and expected labels and returns percentage of correct predictions in all predictions
    def accuracy(self, outputs, expectations):
        # All predictions
        comparison_list = [outputs[k] == expectations[k] for k in range(len(outputs))]
        # Get Percentage of correct predictions
        percentage = comparison_list.count(True) / len(comparison_list)

        return percentage

    def prediction(self, inputs):
        predictions = list()
        for inp in inputs:
            # Get all distances between train dataset (x_train)
            distances = [self.euclidean_distance(inp, point2) for point2 in self.x]
            # Obtain closest neighbours arguments with respect to self.neighbours number
            closest_neighbours = np.argsort(distances)[:self.neighbours]
            # Obtain labels
            closest_neighbours_labels = [self.y[index] for index in closest_neighbours]
            # Get most closest label
            label_counts = np.bincount(closest_neighbours_labels)
            predictions.append(np.argmax(label_counts))

        return predictions


def dataset_preprocessing(csv_file="Breast_cancer_data.csv"):
        # Reading csv file
        dataset = pd.read_csv(csv_file)
        # Getting columns for x and y arrays
        # Assuming last column is output (y)
        dataset_array = dataset.to_numpy()
        np.random.shuffle(dataset_array)  # Uncomment this this row if dataset labels are sequential
        x, y = dataset_array[:, :-1], dataset_array[:, -1]
        test_perc = 0.2  # test percentage
        test_size = int(test_perc * len(x))  # test size

        # Splitting to test and train arrays
        x_train, y_train, x_test, y_test = x[:-test_size], y[:-test_size], x[-test_size:], y[-test_size:]
        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # Applicable for any csv files that last column is output and does not contain any string
    x_train, y_train, x_test, y_test = dataset_preprocessing("breast_cancer.csv")
    max_percentage = 0
    # try neighbour numbers from 1 to 10
    for i in range(1,10):
        knn = K_Nearest_Neighbours(neighbours=i)
        knn.fit(x=x_train, y=y_train)
        score = knn.accuracy(knn.prediction(x_test), y_test)
        # Get maximum accuracy and corresponding neighbour number
        if score > max_percentage:
            neighbour_num, max_percentage = i, score

    print(f"Max percentage is {round(max_percentage,4)} with {neighbour_num} neighbour numbers.")
    # Predict a value from test dataset
    knn = K_Nearest_Neighbours(neighbours=neighbour_num)
    knn.fit(x=x_train, y=y_train)
    print(f"Predicted label: {knn.prediction(x_test)[0]}\nExpected label: {int(y_test[0])}")
