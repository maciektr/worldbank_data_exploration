import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac
import numpy as np
import os
from feature_extraction.extractors import AutoencoderExtractor

INPUT_LEN = 19


def get_project_dir():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    repo_name = "worldbank_data_exploration"
    repo_folder = curr_path.split(repo_name)[0]
    return os.path.join(repo_folder, repo_name)


def dendrogram_each_feature(data, feature_extraction=False, shape=None, combined_image_name=None):
    linkage_matrices = {}
    i = 1
    project_dir = get_project_dir()
    images_dir = os.path.join(project_dir, 'images')

    if shape is not None:
        plt.figure(figsize=(24, 16))
        plt.subplots_adjust(hspace=0.3)
        plt.rcParams['figure.constrained_layout.use'] = True

    for feature, X in data.items():
        image_name = feature

        if feature_extraction:
            extractor = AutoencoderExtractor(feature, input_len=INPUT_LEN, root=project_dir)
            Y = extractor.extract_features(X)
            linkage_matrix = hac.linkage(Y, method="ward")
            image_name = image_name + ' (with feature extraction)'
        else:
            linkage_matrix = hac.linkage(X, method="ward")

        if shape is None:
            plt.figure(figsize=(16, 8))
            hac.dendrogram(linkage_matrix)
            plt.title(feature)
            plt.savefig(os.path.join(images_dir, image_name))
            plt.show()
            linkage_matrices[feature] = linkage_matrix
        else:
            plt.subplot(shape[0], shape[1], i)
            hac.dendrogram(linkage_matrix)
            plt.title(feature)
            linkage_matrices[feature] = linkage_matrix
            i += 1

    if shape is not None:
        plt.savefig(os.path.join(images_dir, combined_image_name))

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
    project_dir = get_project_dir()
    images_dir = os.path.join(project_dir, 'images')

    if feature_extraction:
        y_all = []
        for feature, X in data.items():
            extractor = AutoencoderExtractor(feature, input_len=INPUT_LEN, root=project_dir)
            y_feature = extractor.extract_features(X)
            y_all.append(y_feature)

        y_all = np.hstack(y_all)
    else:
        y_all = np.hstack(list(data.values()))
    plt.figure(figsize=(16, 8))
    linkage_matrix = hac.linkage(y_all, method="ward")
    hac.dendrogram(linkage_matrix)
    plt.title(title)
    plt.savefig(os.path.join(os.path.join(images_dir, 'combined'), title))
    plt.show()

    return linkage_matrix, y_all


def cluster_combined_features(data, linkage_matrix, number_of_clusters):
    data_clustered = np.copy(data)
    cluster_labels = hac.fcluster(
        linkage_matrix, number_of_clusters, criterion="maxclust"
    )
    data_clustered = np.column_stack((data_clustered, cluster_labels))

    return data_clustered
