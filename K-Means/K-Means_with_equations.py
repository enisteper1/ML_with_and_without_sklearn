import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

class KMeans:
    # Get cluster number and initialize variables
    def __init__(self, cluster_num=3):
        self.cluster_num = cluster_num
        # List of clusters
        self.cluster_list = None
        # List of centroids
        self.centroid_list = None
        # Samples
        self.data = None
        # Outputs
        self.y_train = None

    def fit(self,data, iteration_num):
        self.data = data
        # First initialize random centroids
        self.centroid_list = [data[i] for i in range(self.cluster_num)]
        for n in range(iteration_num):
            # Secondly create empty clusters for initialized cluster number
            self.cluster_list = [list() for i in range(self.cluster_num)]
            # Cluster all samples
            for h,sample in enumerate(data):
                #print(h)
                # Calculate the distances between centroids and sample
                distance_list = [self.euclidean_distance(sample, centroid) for centroid in self.centroid_list]
                # Then get minimum distanced centroid idx
                sample = [sample[0], sample[1]]
                centroid_idx = distance_list.index(min(distance_list))
                self.cluster_list[centroid_idx].append(sample)
            # Creating temporary centroid holder to check difference between old and new centroids
            temp_centroid = self.centroid_list
            self.centroid_list = []
            for cluster in self.cluster_list:
                try:
                    cluster = np.array(cluster)
                    # self.centroid_list.append(np.mean(data[cluster], axis=0))
                    self.centroid_list.append(np.mean(cluster, axis=0))
                except:
                    self.centroid_list.append([1,1])
                    pass

        _, ax = plt.subplots(figsize=(8, 6))
        color = ["r", "b", "g", "c", "y", "m", "w"]*5
        for i, cluster in enumerate(self.cluster_list):
            for sample in cluster:
                ax.scatter(sample[0],sample[1],color=color[i], linewidth=1)

        for point in self.centroid_list:
            ax.scatter(point[0],point[1], marker="x", color='black', linewidth=2)

        plt.show()


    def euclidean_distance(self, p1, p2):

        # Calculate sum of distances for all dimensions of p1 and p2
        distance = sum((p1[h] - p2[h]) ** 2 for h in range(len(p1)))
        # Get square root to finish euclidean distance formula
        distance = sqrt(distance)

        return distance


def dataset_preprocessing(csv_file="Breast_cancer_data.csv"):
    # Reading csv file
    dataset = pd.read_csv(csv_file)
    # Getting columns for x and y arrays
    # Assuming last column is output (y)
    dataset_array = dataset.to_numpy()
    # np.random.shuffle(dataset_array)  # Uncomment this this row if dataset labels are sequential
    x, y = dataset_array[:, :-1], dataset_array[:, -1]
    y = np.array(y, dtype=np.int64)
    return x, y


if __name__ == "__main__":
    x, y = dataset_preprocessing("KMEANS_Data.csv")
    kmeans = KMeans(cluster_num=3)
    kmeans.fit(x,500)
    plt.show()
