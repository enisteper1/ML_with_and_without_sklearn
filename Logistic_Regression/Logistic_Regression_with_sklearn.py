from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Create Dataset
    x, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2)
    # With the aim of avoiding overflow x values are squeezed with StandardScaler
    standard_scaler = StandardScaler()
    x = standard_scaler.fit_transform(x)
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2405)
    # Save the data to csv
    y_csv = y[:, np.newaxis]
    combined_to_csv = np.append(x, y_csv, axis=1)
    param_list = [f"param{i}" for i in range(x.shape[1])]
    param_list.append("output")
    # Create DataFrame
    df = pd.DataFrame(data=combined_to_csv, columns=param_list)
    # Save to csv
    df.to_csv("Logistic_Regression.csv", index=False)
    # Define the model
    logistic_regression = LogisticRegression()
    # fit
    logistic_regression.fit(x_train, y_train)
    color_list = ["black", "red"]
    # Predict
    label_list = logistic_regression.predict(x)
    _, ax = plt.subplots(figsize=(8, 6))
    # Add labels
    for i, label in enumerate(label_list):
        ax.scatter(x[i][0], x[i][1], color=color_list[label_list[i]], linewidth=1)

    plt.show()
    acc = logistic_regression.score(x_test, y_test)
    print(f"Accuracy is {round(acc, 4)}")


if __name__ == "__main__":
    main()