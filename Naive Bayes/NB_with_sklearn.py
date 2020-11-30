import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB


def main():
    # Iris or breast cancer dataset can be used too
    x, y = datasets.load_wine(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2405)

    # Multinomial Naive Bayes
    MNB = MultinomialNB()
    MNB.fit(x_train, y_train)
    mnb_accuracy = MNB.score(x_test, y_test)
    print(f"MultinomialNB accuracy is {round(mnb_accuracy, 4)}")

    # Gaussian Naive Bayes
    GNB = GaussianNB()
    GNB.fit(x_train, y_train)
    gnb_accuracy = GNB.score(x_test, y_test)
    print(f"GaussianNB accuracy is {round(gnb_accuracy, 4)}")

    # Complement Naive Bayes
    CNB = ComplementNB()
    CNB.fit(x_train, y_train)
    cnb_accuracy = CNB.score(x_test, y_test)
    print(f"ComplementNB accuracy is {round(cnb_accuracy, 4)}")


if __name__ == "__main__":
    main()