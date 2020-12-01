from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Read csv
    data = pd.read_csv('Polynomial_Regression_Data.csv')
    # Get data
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # Initialize the model
    polynomial_regression = PolynomialFeatures(degree=4)
    # Expand the data with powers of x with respect to degree
    powered_x = polynomial_regression.fit_transform(x)
    # Polynomial model uses linear regression with powered inputs
    linear_regression = LinearRegression()
    linear_regression.fit(powered_x, y)
    predicted_y = linear_regression.predict(polynomial_regression.fit_transform(x))
    plt.scatter(x, y, color='black')
    plt.plot(x, predicted_y, color='red',linewidth=3)
    plt.title("Degree: 4")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()