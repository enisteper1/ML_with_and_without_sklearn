import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    # Get data
    x, y = datasets.load_iris(return_X_y=True)
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2405)
    # Define Random Forest Classifier
    # It uses various subsets and train different decision trees
    random_forest_classifier = RandomForestClassifier(max_depth=10)
    # Fit
    random_forest_classifier.fit(x_train, y_train)
    # Create color list to differentiate every data
    color_list = ["r", "b", "g", "c", "y", "m", "w"] * 5
    # When predicting the model obtains outputs from each decision tree and decides most common output
    label_list = random_forest_classifier.predict(x)
    # Initialize the subplot
    _, ax = plt.subplots(figsize=(8, 6))
    # Add labels
    # 2 feature of x is used in plot
    for i, label in enumerate(label_list):
        ax.scatter(x[i][0], x[i][1], color=color_list[label_list[i]], linewidth=1)
    plt.grid()
    plt.show()
    acc = random_forest_classifier.score(x_test, y_test)
    print(f"Accuracy is {round(acc,4)}")


if __name__ == "__main__":
    main()