from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    # Load Data
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    # Split to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)
    # Build tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    # Show tree in text
    print(export_text(decision_tree, feature_names=iris["feature_names"]))
    figure, axes = plt.subplots(figsize=(12, 8))
    # Visualization of  tree
    tree.plot_tree(decision_tree,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)
    # Save as figure
    figure.savefig('Decision_tree_sklearn.png')
    # Plot
    plt.show()


if __name__ == "__main__":
    main()