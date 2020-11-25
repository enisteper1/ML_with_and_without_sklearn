from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # veri yukleme
    veriler = pd.read_csv('Polynomial_Regression_Data.csv')

    x = veriler.iloc[:, :-1]
    y = veriler.iloc[:, -1]
    X = x.values
    Y = y.values

    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(X)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(x_poly, y)
    plt.scatter(X, Y, color='red')
    plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='black',linewidth=2)
    plt.show()


if __name__ == "__main__":
    main()