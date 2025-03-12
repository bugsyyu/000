"""
Clustering utilities for identifying outlier nodes and groups.
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict, Any, Optional

def detect_outliers_dbscan(
    points: np.ndarray,
    eps: float = 50.0,  # Distance in km
    min_samples: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect outlier points using DBSCAN clustering.

    Args:
        points: Array of shape (n_points, 2) containing x, y coordinates
        eps: Maximum distance between two samples to be considered as neighbors
        min_samples: Minimum number of samples in a neighborhood for a point to be a core point

    Returns:
        Tuple of (labels, core_points, outlier_points)
    """
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # Get core samples and outliers
    core_sample_indices = np.zeros_like(labels, dtype=bool)
    core_sample_indices[dbscan.core_sample_indices_] = True

    # Outliers have label -1
    outlier_indices = labels == -1

    # Subset the original points
    core_points = points[core_sample_indices]
    outlier_points = points[outlier_indices]

    return labels, core_points, outlier_points

def find_optimal_clusters(
    points: np.ndarray,
    max_clusters: int = 10
) -> Tuple[int, np.ndarray]:
    """
    Find the optimal number of clusters using silhouette score.

    Args:
        points: Array of shape (n_points, 2) containing x, y coordinates
        max_clusters: Maximum number of clusters to consider

    Returns:
        Tuple of (optimal_n_clusters, cluster_labels)
    """
    if len(points) <= 1:
        return 1, np.zeros(len(points), dtype=int)

    # Limit max_clusters based on the number of points
    max_possible_clusters = min(max_clusters, len(points) - 1)

    if max_possible_clusters <= 1:
        return 1, np.zeros(len(points), dtype=int)

    best_score = -1
    best_n_clusters = 2  # Default to 2 clusters
    best_labels = None

    # Try different numbers of clusters
    for n_clusters in range(2, max_possible_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)

        # Only compute silhouette score if we have at least 2 clusters with data
        if len(np.unique(labels)) > 1:
            score = silhouette_score(points, labels)

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels

    # If no good cluster was found, default to 1 cluster
    if best_labels is None:
        return 1, np.zeros(len(points), dtype=int)

    return best_n_clusters, best_labels

def identify_isolated_clusters(
    points: np.ndarray,
    distance_threshold: float = 100.0  # Distance in km
) -> List[int]:
    """
    Identify clusters that are isolated from others.

    Args:
        points: Array of shape (n_points, 2) containing x, y coordinates
        distance_threshold: Threshold distance to consider a cluster isolated

    Returns:
        List of indices of isolated clusters
    """
    # Find the optimal number of clusters
    n_clusters, labels = find_optimal_clusters(points)

    if n_clusters <= 1:
        return []

    # Calculate cluster centers
    cluster_centers = []
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)

    cluster_centers = np.array(cluster_centers)

    # Calculate pairwise distances between cluster centers
    isolated_clusters = []
    for i in range(n_clusters):
        center = cluster_centers[i]

        # Calculate distances to all other centers
        distances = []
        for j in range(n_clusters):
            if i != j:
                dist = np.linalg.norm(center - cluster_centers[j])
                distances.append(dist)

        # If the minimum distance is greater than the threshold, it's isolated
        if min(distances) > distance_threshold:
            isolated_clusters.append(i)

    return isolated_clusters

def suggest_intermediate_nodes(
    points: np.ndarray,
    n_points: int = 9,  # Number of frontline or airport points
    max_intermediate_nodes: int = 15,
    isolation_threshold: float = 100.0
) -> np.ndarray:
    """
    Suggest intermediate nodes based on clustering results.

    Args:
        points: Array of shape (n_points*2, 2) containing frontline and airport coordinates
        n_points: Number of frontline/airport points (assumed to be equal)
        max_intermediate_nodes: Maximum number of intermediate nodes to suggest
        isolation_threshold: Threshold to consider a cluster isolated

    Returns:
        Array of shape (m, 2) containing suggested intermediate node coordinates
    """
    # Separate frontline and airport points
    frontline_points = points[:n_points]
    airport_points = points[n_points:2*n_points]

    # Combine all points for clustering
    all_points = np.vstack([frontline_points, airport_points])

    # Identify isolated clusters
    n_clusters, labels = find_optimal_clusters(all_points)
    isolated_cluster_indices = identify_isolated_clusters(all_points, isolation_threshold)

    # Generate intermediate nodes
    intermediate_nodes = []

    # 1. Add nodes between isolated clusters and main clusters
    if isolated_cluster_indices:
        main_cluster_indices = [i for i in range(n_clusters) if i not in isolated_cluster_indices]

        # Calculate cluster centers
        cluster_centers = []
        for i in range(n_clusters):
            cluster_points = all_points[labels == i]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)

        # For each isolated cluster, add a node halfway to the closest main cluster
        for iso_idx in isolated_cluster_indices:
            iso_center = cluster_centers[iso_idx]

            # Find closest main cluster
            min_dist = float('inf')
            closest_center = None

            for main_idx in main_cluster_indices:
                main_center = cluster_centers[main_idx]
                dist = np.linalg.norm(iso_center - main_center)

                if dist < min_dist:
                    min_dist = dist
                    closest_center = main_center

            if closest_center is not None:
                # Add a point halfway between
                halfway_point = (iso_center + closest_center) / 2
                intermediate_nodes.append(halfway_point)

    # 2. Add nodes at strategic locations within clusters
    for i in range(n_clusters):
        cluster_points = all_points[labels == i]

        # Skip very small clusters
        if len(cluster_points) < 2:
            continue

        # Add a node at the centroid
        centroid = np.mean(cluster_points, axis=0)
        intermediate_nodes.append(centroid)

        # If it's a larger cluster, add more nodes
        if len(cluster_points) >= 4:
            # Add nodes at the geometric mean of frontline and airport points in this cluster
            frontline_in_cluster = [p for p, l in zip(frontline_points, labels[:n_points]) if l == i]
            airports_in_cluster = [p for p, l in zip(airport_points, labels[n_points:]) if l == i]

            if frontline_in_cluster and airports_in_cluster:
                frontline_center = np.mean(frontline_in_cluster, axis=0)
                airport_center = np.mean(airports_in_cluster, axis=0)

                # Add a point between frontline and airport centers
                halfway_point = (frontline_center + airport_center) / 2
                intermediate_nodes.append(halfway_point)

    # Limit the number of intermediate nodes
    intermediate_nodes = np.array(intermediate_nodes)
    if len(intermediate_nodes) > max_intermediate_nodes:
        # If we have too many, use KMeans to reduce
        kmeans = KMeans(n_clusters=max_intermediate_nodes, random_state=42, n_init=10)
        kmeans.fit(intermediate_nodes)
        intermediate_nodes = kmeans.cluster_centers_

    return intermediate_nodes