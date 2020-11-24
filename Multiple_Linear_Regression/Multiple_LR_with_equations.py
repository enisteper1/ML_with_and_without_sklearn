import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MultipleLinearRegression:

    def __init__(self):
        self.x = None # parameters
        self.y = None # outputs
        self.thetas = None # Will contain weights for each parameter category
        self.temporary_thetas = None # Will contain updated thetas

    def fit(self, x, y, iteration_num=8000, learning_rate = 1E-2, accepted_error=1E-4):
        self.x = x
        self.y = y
        self.thetas = np.ones(self.x.shape[1])
        self.temporary_thetas = np.ones(self.x.shape[1])
        # Fit with respect to iteration num
        for i in range(iteration_num):
            error = self.mse(self.prediction(self.x), self.y)
            if error <= accepted_error:
                break
            # Get new thetas
            for j in range(self.x.shape[1]):
                self.temporary_thetas[j] = self.thetas[j] - learning_rate * self.cost_func(j)
            # Update the thetas
            self.thetas = self.temporary_thetas

        return self.thetas

        # Mean Squared Error
    def mse(self, output, expected):
        error = np.sum((expected - output) ** 2) / len(output)
        return error

    def prediction(self, x):
        # multiply weights with parameters
        predicted_y = np.array([theta * x[:, i] for i, theta in enumerate(self.thetas)])
        # get transpose of the array to turn rows to columns
        predicted_y = predicted_y.transpose()
        # get sum of all columns and create 1 featured array
        predicted_y = np.array([np.sum(predicted_y[i]) for i in range(x.shape[0])])
        return predicted_y

    def accuracy(self, output_predicted, output_true):
        # Turn rows to columns
        # get numerator to squeeze values between 0 to 1
        numerator = np.sum(((output_true - output_predicted) ** 2), axis=0, dtype=np.float64)
        # get denominator to squeeze values between 0 to 1
        denominator = np.sum(((output_true - np.average(
            output_true, axis=0)) ** 2), axis=0, dtype=np.float64)
        # Get mean of 1 - squeezed error value
        accuracy = np.mean(1 - np.mean(numerator) / np.mean(denominator))

        return accuracy

    # This function is used to return partial derivative for every parameter
    def cost_func(self, current_param):
        # Obtain the prediction
        y_pred = self.prediction(self.x)
        # Obtain the result of partial derivative for each variable
        funct = np.sum((self.y - y_pred) * (- self.x[:, current_param])) / self.x.shape[0]

        return funct


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
    multiple_LR = MultipleLinearRegression()
    # Getting x, y and test_size
    x, y, test_size = dataset_preprocessing("Multiple_Regression_Data.csv")
    # Splitting to test and train arrays
    x_train, y_train, x_test, y_test = x[:-test_size], y[:-test_size], x[-test_size:], y[-test_size:]
    # fitting data with selected iteration number and learning rate
    multiple_LR.fit(x_train, y_train, iteration_num=5000, learning_rate=1E-2)
    # Used all the data with the aim of getting continuous line on plot
    predict = multiple_LR.prediction(x)
    plt.plot(x[:, 0], y, "o", color="b")
    plt.plot(x[:, 0], predict, "o", color="r")
    plt.grid()
    plt.show()
    # Prediction of test values
    test_prediction = multiple_LR.prediction(x_test)
    score = multiple_LR.accuracy(test_prediction, y_test)
    print(f"Accuracy is {score}")