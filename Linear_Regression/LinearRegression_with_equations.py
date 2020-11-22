import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In LinearRegression_with_sklearn.py csv type dataset creation operation is added
# First run the LinearRegression_with_sklearn.py to compare how line fits in same dataset

class Linear_Regression:

    def __init__(self):
        self.theta_1 = 1  # initial bias value
        self.theta_2 = 1  # initial weight value

    def fit(self, x_train, y_train, iteration_num=8000, learning_rate=0.01, accepted_error=1E-4):
        self.theta_1 = 1  # reinitialized bias value
        self.theta_2 = 1  # reinitialized weight value
        training_num = x_train.shape[0]

        # How many times data will be iterated
        for i in range(iteration_num):
            # Calculated Mean Squared Error to check if fitting satisfies accepted error
            error = self.mse(self.theta_1 + x_train * self.theta_2, y_train)
            if error <= accepted_error:
                break
            # Calculate all y values for corresponding bias and weight value
            # calculated_y = self.theta_1 + x * self.theta_2
            # Cost Function(MSE) -> J = 1/2 * n sum([(y[i] - (theta_1 + x[i] * theta_2)**2) for i in range(len(x))])
            # Differentiation with respect to weight dJ/dtheta_2 = 1/(2*n) * 2 * (y - (theta_1 + x * theta_2)(-x))
            # Differentiation with respect to bias dJ/dtheta_1 = 1/(2*n) * 2 * (y - (theta_1 + x * theta_2)(-1))
            # n -> number of samples, x -> input list, y -> output list

            # Since changing theta_2 before calculating theta_1  will avoid fitting well, temporary values are added
            new_theta_2 = self.theta_2 - learning_rate / training_num * np.sum(
                [(self.theta_1 + x_train[k] * self.theta_2 - y_train[k]) * x_train[k] for k in range(len(x_train))])

            new_theta_1 = self.theta_1 - learning_rate / training_num * np.sum(
                                        self.theta_1 + x_train * self.theta_2 - y_train)

            self.theta_2 = new_theta_2
            self.theta_1 = new_theta_1

        return self.theta_2, self.theta_1

    # Mean Squared Error
    def mse(self, output, expected):
        error = np.sum((expected - output)**2) / len(output)
        return error

    # Mean Absolute Error
    def mae(self,output, expected):
        error = np.sum(abs(expected - output)) / len(output)
        return error

    # Returns the slope with updated weight and bias
    def prediction(self, x):
        y = self.theta_1 + x * self.theta_2
        return y


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
    lin_reg = Linear_Regression()
    # Getting x, y and test_size
    x, y, test_size = dataset_preprocessing("Regression_Data.csv")
    # Splitting to test and train arrays
    x_train, y_train, x_test, y_test = x[:-test_size], y[:-test_size], x[-test_size:], y[-test_size:]
    # fitting data with selected iteration number and learning rate
    lin_reg.fit(x_train, y_train, iteration_num=10000, learning_rate=1E-2)
    # Predict the line
    predict = lin_reg.prediction(x_test)
    plt.plot(x, y, "o")
    plt.plot(x_test, predict)
    plt.grid()
    plt.show()


