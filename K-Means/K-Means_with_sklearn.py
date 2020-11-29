from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    # Create the dataset
    x, y = datasets.make_blobs(centers=3, n_samples=400, n_features=2)
    # Save to csv
    y_csv = y[:, np.newaxis]
    combined_to_csv = np.append(x, y_csv, axis=1)
    param_list = [f"param{i}" for i in range(x.shape[1])]
    param_list.append("output")
    # Create DataFrame
    df = pd.DataFrame(data=combined_to_csv, columns=param_list)
    # Save to csv
    df.to_csv("KMEANS_Data.csv", index=False)
    # Initialize and fit the model
    kmeans = KMeans(n_clusters=3).fit(x)
    # Get centers of the clusters
    centers = kmeans.cluster_centers_
    # Find labels
    label_list = pairwise_distances_argmin(x, centers)
    # Create color list to differentiate every data
    color_list = ["r", "b", "g", "c", "y", "m", "w"] * 5
    # Initialize the subplot
    _, ax = plt.subplots(figsize=(8, 6))
    # Add labels
    for i,label in enumerate(label_list):
        ax.scatter(x[i][0], x[i][1], color=color_list[label_list[i]], linewidth=1)
    # Add centers
    for h,center in enumerate(centers):
        ax.scatter(center[0], center[1], marker="x", color="black", linewidth=2)
    # Show
    plt.show()

if __name__ == "__main__":
    main()