from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    # Create dataset
    x, y = datasets.make_regression(n_samples=150,n_features=1,bias=1,noise=15)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=False)
    # Turn row to column
    y = y[:, np.newaxis]
    combined = np.append(x, y, axis=1)
    # Write dataset to csv file
    df = pd.DataFrame(data=combined, columns=["input","output"])

    df.to_csv("Regression_Data.csv", index=False)
    # Define The model
    lin_reg = LinearRegression()
    # Fit with train data
    lin_reg.fit(x_train,y_train)
    # Predict the test data
    predicted_y = lin_reg.predict(x_test)
    # Plot true points
    plt.plot(x, y, "o")
    # Plot predicted line
    plt.plot(x_test, predicted_y)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
