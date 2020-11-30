import numpy as np
import pandas as pd


class GaussianNaiveBayes:

    def __init__(self):
        self.class_list = None  # Will contain unique classes
        self.x = None  # Inputs
        self.y = None  # Outputs
        self.params_list = {}  # Will contain parameters for each class
        self.epsilon = 1E-5  # It will be used to avoid division by 0 error
        self.mean_of_classes = {}  # Mean of column of classes
        self.var_of_classes = {}  # Variance of classes
        self.prior_of_classes = {}  # Probability of class

    def fit(self, x, y):
        # Get each class
        self.class_list = np.unique(y)
        # hold x
        self.x = x
        # Get sample number
        sample_num = self.x.shape[0]
        # hold y
        self.y = y
        for i, each_class in enumerate(self.class_list):
            # Get all rows for each class
            each_class_rows = self.x[each_class == self.y]
            self.mean_of_classes[str(each_class)] = each_class_rows.mean(axis=0)
            self.var_of_classes[str(each_class)] = each_class_rows.var(axis=0)
            self.prior_of_classes[str(each_class)] = each_class_rows.shape[0] / sample_num
    # Prediction
    def prediction(self, x):
        # Container of predictions
        prediction_list = list()
        # Each row is processed one by one
        for each_x in x:
            # Posterior Container
            posterior_list = list()
            # Calculate Posterior for each class
            for i, each_class in enumerate(self.class_list):

                prior = np.log(self.prior_of_classes[str(each_class)])
                # Get maen, var, posterior of each class
                mean = self.mean_of_classes[str(each_class)]
                var = self.var_of_classes[str(each_class)]
                posterior = np.sum(np.log(self.calculate_likelihood(var, mean, each_x)))
                # Posterior probability of class given predictor
                posterior = posterior + prior
                posterior_list.append(posterior)
            output = self.class_list[np.argmax(posterior_list)]
            prediction_list.append(output)

        return np.array(prediction_list)

    # Posterior probability of target of predictor given class
    def calculate_likelihood(self, var_of_class, mean_of_class, sample):
        # Gaussian calculation for each sample
        eq1 = np.exp(- ((sample-mean_of_class)**2) / (2 * var_of_class + self.epsilon))
        eq2 = 1 / np.sqrt(2 * np.pi * var_of_class)
        return eq1 * eq2

    # Calculate accuracy
    def accuracy(self, output_predicted, output_true):
        acc = np.sum(output_true == output_predicted) / len(output_predicted)
        return acc


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
    # Initialization
    gnb = GaussianNaiveBayes()
    x_train, y_train, x_test, y_test = dataset_preprocessing("wine_data.csv")
    gnb.fit(x_train, y_train)
    # Prediction
    output_predicted = gnb.prediction(x_test)
    accuracy = gnb.accuracy(output_predicted, y_test)
    print(f"Accuracy is {round(accuracy,4)}")