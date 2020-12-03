from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Because of y is between 0-10 Multinomial Logistic Regression will be used automatically
    x, y = datasets.load_digits(return_X_y=True)
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2405)
    # Define model
    logistic_regression = LogisticRegression()
    # Fit
    logistic_regression.fit(x_train, y_train)
    # Create columns for csv
    params = [f"param{i}" for i in range(x.shape[1])]
    # Add output as last column
    params.append("output")
    # Turn row to column
    y_csv = y[:, np.newaxis]
    combined_xy = np.append(x, y_csv, axis=1)
    csv = pd.DataFrame(data=combined_xy, columns=params)
    csv.to_csv("Multinomial_Log_Reg.csv", index=False)
    # Show accuracy of model
    print(logistic_regression.score(x_test, y_test))


if __name__ == "__main__":
    main()