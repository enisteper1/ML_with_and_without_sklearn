import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self):
        self.theta_1 = None  # Bias
        self.theta_2 = None  # Weights

    def fit(self, x, y, iteration_num=8000, learning_rate=1E-2):
        # Get samples
        training_num = x.shape[0]
        # Initialize weights and bias
        self.theta_1 = 1
        self.theta_2 = np.ones(x.shape[1])
        # Train respect to iteration num
        for i in range(iteration_num):
            # In order to make prediction about logistic reg. linear model is calculated
            y_linear = self.theta_1 + np.dot(x, self.theta_2)
            # Sigmoid function output is obtained for gradient descent in Logistic Regression
            y_sigmoid = self.sigmoid(y_linear)
            # Get new weight and bias
            new_theta_1 = self.theta_1 - learning_rate / training_num * np.sum(y_sigmoid - y)
            new_theta_2 = self.theta_2 - learning_rate / training_num * np.dot(x.transpose(), (y_sigmoid - y))
            # Assign the new weight and bias
            self.theta_1 = new_theta_1
            self.theta_2 = new_theta_2

    # Prediction
    def prediction(self, x):
        # Linear model
        y_linear = self.theta_1 + np.dot(x, self.theta_2)
        # Logistic Regression outputs
        y_sigmoid = list(map(self.sigmoid, y_linear))
        # If output is above 0.5 it is assumed as it belongs to 1 otherwise 0
        y_pred = list(map(lambda x: 1 if x > 0.5 else 0, y_sigmoid))
        return y_pred

    # Sigmoid function
    def sigmoid(self, y_linear):
        return 1 / (1 + np.exp(- y_linear))

    # Accuracy calculation
    def accuracy(self, output_predicted, output_true):
        # Get comparison of predicted and true output
        comparison_list = [output_predicted[k] == output_true[k] for k in range(len(output_predicted))]
        # Get Percentage of correct predictions
        acc = comparison_list.count(True) / len(comparison_list)
        return acc



def dataset_preprocessing(csv_file="Breast_cancer_data.csv"):
    # Reading csv file
    dataset = pd.read_csv(csv_file)
    # Getting columns for x and y arrays
    # Assuming last column is output (y)
    dataset_array = dataset.to_numpy()
    # np.random.shuffle(dataset_array) # Uncomment this line if shuffling operation is needed
    x, y = dataset_array[:, :-1], dataset_array[:, -1]
    test_percentage = 0.2  # It can be decreased or increased to see how accuracy percentage varies
    test_size = int(test_percentage * len(x))  # test size
    return x, y, test_size


if __name__ == "__main__":
    # Define model
    logistic_regression = LogisticRegression()
    # Split to x, y and test_size
    x, y, test_size = dataset_preprocessing("Logistic_Regression.csv")
    # Split to train and test
    x_train, y_train, x_test, y_test = x[:-test_size], y[:-test_size], x[-test_size:], y[-test_size:]
    # Fit the model
    logistic_regression.fit(x_train, y_train, iteration_num=6000, learning_rate=1E-2)
    # Since 2 class is used 2 color is enough
    color_list = ["black", "red"]
    # Prediction
    label_list = logistic_regression.prediction(x)
    _, ax = plt.subplots(figsize=(8, 6))
    # Add labels
    for i, label in enumerate(label_list):
        ax.scatter(x[i][0], x[i][1], color=color_list[label_list[i]], linewidth=1)
    plt.grid()
    plt.show()
    y_pred = logistic_regression.prediction(x_test)
    acc = logistic_regression.accuracy(y_pred, y_test)
    print(f"Accuracy is {round(acc, 4)}")
