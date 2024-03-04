import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(similarity_matrix, num_clusters):
    # Step 1: 计算拉普拉斯矩阵
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    # Step 2: 特征分解
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

    # Step 3: 选择特征向量
    # 选择前 num_clusters 个最小的非零特征值对应的特征向量
    indices = np.argsort(eigenvalues)[:num_clusters]
    selected_eigenvectors = eigenvectors[:, indices]

    # Step 4: 聚类分析
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(selected_eigenvectors)
    labels = kmeans.labels_

    return labels

# Example usage:
# similarity_matrix = your_similarity_matrix
# num_clusters = desired_number_of_clusters
# labels = spectral_clustering(similarity_matrix, num_clusters)
