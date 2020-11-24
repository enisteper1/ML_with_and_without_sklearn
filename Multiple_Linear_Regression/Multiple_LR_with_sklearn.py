from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Create Dataset
    x, y = datasets.make_regression(n_samples=200,n_features=5,bias=1,noise=15)
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=322)
    # Save to csv
    y_csv = y[:, np.newaxis]
    combined_to_csv = np.append(x, y_csv, axis=1)
    param_list = [f"param{i}" for i in range(x.shape[1])]
    param_list.append("output")
    # Create DataFrame
    df = pd.DataFrame(data=combined_to_csv, columns=param_list)
    # Save to csv
    df.to_csv("Multiple_Regression_Data.csv", index=False)
    # Initialize model
    multiple_LR = LinearRegression()
    # Fit
    multiple_LR.fit(x_train, y_train)
    # Prediction
    predicted_y = multiple_LR.predict(x_test)
    # Since it has multiple parameters only prediction and true outputs can be compared in plot
    # True outputs
    plt.plot(x_test[:, 0], y_test, "o", color="r")
    # Predicted outputs
    plt.plot(x_test[:, 0], predicted_y, "o", color="b")
    # Add grid
    plt.grid()
    # Show the plot
    plt.show()
    # To understand success rate of the model accuracy is calculated
    score = multiple_LR.score(x_test, y_test)
    print(f"Accuracy is  {score}%")

if __name__ == "__main__":
    main()