import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# This Regression can be used for polynomial data
class PolynomialRegression:

    def __init__(self, degree):
        self.x = None  # Parameters
        self.y = None  # Outputs
        self.degree = degree  # Initialize the degree


    # Fit the model with given data
    def fit(self, x, y, iteration_num=5000, learning_rate=1E-4, accepted_error=1E-4):
        # Initialize the data
        self.x = x
        self.y = y
        self.thetas = np.zeros(self.degree + 1)  # Weights
        self.temporary_thetas = np.zeros(self.degree + 1)  # Temporary weights
        for i in range(iteration_num):
            # Get new error
            error = self.mse(self.prediction(x), y)
            if error <= accepted_error:
                break
            # Calculate new weights
            for j in range(len(self.thetas)):
                self.temporary_thetas[j] = self.thetas[j] - learning_rate * self.cost_function(j)
            # Update the weights
            self.thetas = self.temporary_thetas

    # Our cost function to update weights
    def cost_function(self,current_degree):
        # Obtain the prediction
        y_pred = self.prediction(self.x)
        # Obtain the result of partial derivative for each variable
        funct = np.sum((self.y - y_pred) * (- self.x[:, current_degree])) / self.x.shape[0]

        return funct

    # Prediction
    def prediction(self, x):
        # multiply weights with parameters and degree
        predicted_y = self.thetas * x
        # get transpose of the array to turn rows to columns
        #predicted_y = predicted_y.transpose()
        # get sum of all columns and create 1 featured array
        predicted_y = predicted_y.sum(axis=1)
        return predicted_y

    # Calculate accuracy
    def accuracy(self, output_predicted, output_true):
        # get numerator to squeeze values between 0 to 1
        numerator = np.sum(((output_true - output_predicted) ** 2), axis=0, dtype=np.float64)
        # get denominator to squeeze values between 0 to 1
        denominator = np.sum(((output_true - np.average(
            output_true, axis=0)) ** 2), axis=0, dtype=np.float64)
        # Get mean of 1 - squeezed error value
        accuracy = np.mean(1 - np.mean(numerator) / np.mean(denominator))

        return accuracy

    # Error calculation with Mean Squared Error
    def mse(self, output_predicted, output_true):
        error = np.sum((output_true - output_predicted)**2) / len(output_predicted)
        return error

    def fit_transform(self, x, degree):
        # Initialize the new x which will contain powered of x data
        # Define column and row number
        new_x = np.zeros((x.shape[0], degree + 1))
        # Set the powered new x data
        for i in range(degree+1):
            new_x[:, i] = x ** i
        return new_x


def dataset_preprocessing(csv_file="Breast_cancer_data.csv"):
    # Reading csv file
    dataset = pd.read_csv(csv_file)
    # Getting columns for x and y arrays
    # Assuming last column is output (y)
    dataset_array = dataset.to_numpy()
    # np.random.shuffle(dataset_array) # Uncomment this line if shuffling operation is needed
    x, y = dataset_array[:, 0], dataset_array[:, -1]
    test_percentage = 0.2  # It can be decreased or increased to see how accuracy percentage varies
    test_size = int(test_percentage * len(x))  # test size
    return x, y, test_size


if __name__ == "__main__":
    polynomial_regression = PolynomialRegression(degree=4)
    # Getting x, y
    x, y, _ = dataset_preprocessing("Polynomial_Regression_Data.csv")
    # Create powered columns
    x_new = polynomial_regression.fit_transform(x, polynomial_regression.degree)
    # Wont split the data because of data is too small
    # fitting data with selected iteration number and learning rate
    polynomial_regression.fit(x_new, y, iteration_num=10000, learning_rate=1)
    # Used all the data with the aim of getting continuous line on plot
    predict = polynomial_regression.prediction(x_new)
    plt.plot(x, y, "o", color="black")
    plt.plot(x, predict, color="red", linewidth=3)
    plt.title("Degree: 4")
    plt.grid()
    plt.show()
    # Prediction of test values
    prediction = polynomial_regression.prediction(x_new)
    score = polynomial_regression.accuracy(prediction, y)
    print(f"Accuracy is {score}")
