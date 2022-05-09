import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac
import numpy as np
from feature_extraction.extractors import AutoencoderExtractor

INPUT_LEN = 19


def dendrogram_each_feature(data, feature_extraction=False):
    linkage_matrices = {}

    for feature, X in data.items():
        if feature_extraction:
            extractor = AutoencoderExtractor(feature, input_len=INPUT_LEN)
            Y = extractor.extract_features(X)
            linkage_matrix = hac.linkage(Y, method="ward")
        else:
            linkage_matrix = hac.linkage(X, method="ward")

        plt.figure(figsize=(16, 8))
        hac.dendrogram(linkage_matrix)
        plt.title(feature)
        plt.show()
        linkage_matrices[feature] = linkage_matrix

    return linkage_matrices


def cluster_each_feature(data, linkage_matrices, number_of_clusters):
    data_clustered = data.copy()

    for feature, X in data_clustered.items():
        cluster_labels = hac.fcluster(
            linkage_matrices[feature], number_of_clusters[feature], criterion="maxclust"
        )
        x_clustered = np.column_stack((X, cluster_labels))
        data_clustered[feature] = x_clustered

    return data_clustered


def dendrogram_features_combined(data, title, feature_extraction=False):
    if feature_extraction:
        y_all = []
        for feature, X in data.items():
            extractor = AutoencoderExtractor(feature, input_len=INPUT_LEN)
            y_feature = extractor.extract_features(X)
            y_all.append(y_feature)

        y_all = np.hstack(y_all)
    else:
        y_all = np.hstack(list(data.values()))
    plt.figure(figsize=(16, 8))
    linkage_matrix = hac.linkage(y_all, method="ward")
    hac.dendrogram(linkage_matrix)
    plt.title(title)
    plt.show()

    return linkage_matrix, y_all


def cluster_combined_features(data, linkage_matrix, number_of_clusters):
    data_clustered = np.copy(data)
    cluster_labels = hac.fcluster(
        linkage_matrix, number_of_clusters, criterion="maxclust"
    )
    data_clustered = np.column_stack((data_clustered, cluster_labels))

    return data_clustered
