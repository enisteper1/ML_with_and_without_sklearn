from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


cancer_dataset = datasets.load_breast_cancer()
X, y = cancer_dataset.data, cancer_dataset.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if __name__ == "__main__":
    max_percentage = 0
    for i in range(1,10):
        knn_sk = KNeighborsClassifier(n_neighbors=i)
        knn_sk.fit(x_train, y_train)
        score_sk = knn_sk.score(x_test,y_test)

        if score_sk > max_percentage:
            neighbor_num, max_percentage = i, score_sk
    print(f"Max percentage is {round(max_percentage, 4)} with {neighbor_num} neighbour numbers.")
    knn_sk = KNeighborsClassifier(n_neighbors=neighbor_num)
    knn_sk.fit(x_train, y_train)
    print(f"Predicted label: {knn_sk.predict(x_test)[0]}\nExpected label: {int(y_test[0])}")


