"""
Utilities for the reinforcement learning environments.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import (
    distance_point_to_point,
    does_line_cross_adi_zone,
    get_danger_zone_penalty,
    is_line_segment_valid
)

def find_shortest_paths(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    start_indices: List[int],
    end_indices: List[int]
) -> Tuple[List[List[int]], List[bool]]:
    """
    Find shortest paths between multiple start and end points.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        start_indices: List of start node indices
        end_indices: List of end node indices

    Returns:
        Tuple of (paths, success_flags)
    """
    # Create a graph from nodes and edges
    G = nx.Graph()

    # 修复: 首先添加所有节点，包括没有边的节点
    # 这样确保所有的起始点和终点都存在于图中
    for i in range(len(nodes)):
        G.add_node(i)

    # Add weighted edges
    for i, j in edges:
        weight = distance_point_to_point(nodes[i], nodes[j])
        G.add_edge(i, j, weight=weight)

    paths = []
    success_flags = []

    # Find shortest path for each start-end pair
    for start_idx in start_indices:
        for end_idx in end_indices:
            try:
                # 检查起点和终点是否有连通路径
                if nx.has_path(G, source=start_idx, target=end_idx):
                    path = nx.shortest_path(G, source=start_idx, target=end_idx, weight='weight')
                    paths.append(path)
                    success_flags.append(True)
                else:
                    # 没有路径
                    paths.append([])
                    success_flags.append(False)
            except nx.NetworkXNoPath:
                # No path exists
                paths.append([])
                success_flags.append(False)
            except nx.NodeNotFound as e:
                # 节点不存在 (这不应该发生，因为我们已经确保添加了所有节点)
                print(f"Warning: Node not found in graph: {e}")
                paths.append([])
                success_flags.append(False)

    return paths, success_flags


def check_adi_zone_traversal(
    path: List[int],
    nodes: np.ndarray,
    adi_zones: List[Dict]
) -> List[bool]:
    """
    Check if a path correctly traverses ADI zones.

    Args:
        path: List of node indices representing a path
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        adi_zones: List of ADI zone parameters

    Returns:
        List of booleans indicating whether each ADI zone is correctly traversed
    """
    if len(path) < 2:
        return [False] * len(adi_zones)

    # For each ADI zone, check if the path properly traverses it
    zone_traversed = [False] * len(adi_zones)

    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]

        start_point = tuple(nodes[start_idx])
        end_point = tuple(nodes[end_idx])
        line = (start_point, end_point)

        for j, zone in enumerate(adi_zones):
            crosses_inner, crosses_outer = does_line_cross_adi_zone(line, zone)

            # If it crosses the inner zone, invalid path
            if crosses_inner:
                zone_traversed[j] = False

            # If it crosses the outer zone, mark as traversed
            if crosses_outer:
                zone_traversed[j] = True

    return zone_traversed

def calculate_connectivity_score(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    frontline_indices: List[int],
    airport_indices: List[int]
) -> Tuple[float, int, List[List[int]]]:
    """
    Calculate connectivity score between frontline and airport nodes.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        frontline_indices: List of frontline node indices
        airport_indices: List of airport node indices

    Returns:
        Tuple of (connectivity_score, num_connected_pairs, paths)
    """
    paths, success_flags = find_shortest_paths(nodes, edges, frontline_indices, airport_indices)

    # Count successful paths
    num_connected_pairs = sum(success_flags)

    # Calculate connectivity score (0.0 to 1.0)
    total_pairs = len(frontline_indices) * len(airport_indices)
    connectivity_score = num_connected_pairs / total_pairs if total_pairs > 0 else 0.0

    return connectivity_score, num_connected_pairs, paths

def check_path_angles(path: List[int], nodes: np.ndarray, min_angle_deg: float = 80.0) -> List[float]:
    """
    Check if a path has any angles less than the specified minimum.

    Args:
        path: List of node indices representing a path
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        min_angle_deg: Minimum allowed angle in degrees

    Returns:
        List of angles at each internal node
    """
    if len(path) < 3:
        return []

    angles = []

    for i in range(1, len(path) - 1):
        prev_idx = path[i - 1]
        curr_idx = path[i]
        next_idx = path[i + 1]

        prev_point = tuple(nodes[prev_idx])
        curr_point = tuple(nodes[curr_idx])
        next_point = tuple(nodes[next_idx])

        # Calculate vectors
        v1 = np.array(curr_point) - np.array(prev_point)
        v2 = np.array(next_point) - np.array(curr_point)

        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Calculate angle in radians
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)

    return angles

def evaluate_network(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    frontline_indices: List[int],
    airport_indices: List[int],
    adi_zones: List[Dict],
    danger_zones: List[Dict],
    min_angle_deg: float = 80.0
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the airspace network.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        frontline_indices: List of frontline node indices
        airport_indices: List of airport node indices
        adi_zones: List of ADI zone parameters
        danger_zones: List of danger zone parameters
        min_angle_deg: Minimum allowed angle in degrees

    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate connectivity
    connectivity_score, num_connected_pairs, paths = calculate_connectivity_score(
        nodes, edges, frontline_indices, airport_indices
    )

    # Check ADI zone traversal
    valid_adi_traversals = 0
    adi_traversal_metrics = []

    for path in paths:
        if not path:  # Skip empty paths
            continue

        zone_traversals = check_adi_zone_traversal(path, nodes, adi_zones)
        adi_traversal_metrics.append(zone_traversals)

        # Count paths that traverse at least one ADI zone correctly
        if any(zone_traversals):
            valid_adi_traversals += 1

    # Check path angles
    valid_angle_paths = 0
    angle_metrics = []

    for path in paths:
        if not path or len(path) < 3:  # Skip short paths
            continue

        angles = check_path_angles(path, nodes, min_angle_deg)
        angle_metrics.append(angles)

        # Count paths with all angles valid
        if all(angle >= min_angle_deg for angle in angles):
            valid_angle_paths += 1

    # Calculate danger zone penalties
    total_danger_penalty = 0.0

    for i, j in edges:
        start_point = tuple(nodes[i])
        end_point = tuple(nodes[j])
        line = (start_point, end_point)

        penalty = get_danger_zone_penalty(line, danger_zones)
        total_danger_penalty += penalty

    # Calculate network complexity (number of edges relative to nodes)
    complexity = len(edges) / len(nodes) if len(nodes) > 0 else 0.0

    # Calculate average path length
    total_length = 0.0
    count = 0

    for path in paths:
        if not path:  # Skip empty paths
            continue

        path_length = 0.0
        for i in range(len(path) - 1):
            start_idx = path[i]
            end_idx = path[i + 1]

            start_point = tuple(nodes[start_idx])
            end_point = tuple(nodes[end_idx])

            segment_length = distance_point_to_point(start_point, end_point)
            path_length += segment_length

        total_length += path_length
        count += 1

    avg_path_length = total_length / count if count > 0 else 0.0

    # Compile the evaluation results
    results = {
        'connectivity_score': connectivity_score,
        'num_connected_pairs': num_connected_pairs,
        'total_pairs': len(frontline_indices) * len(airport_indices),
        'valid_adi_traversals': valid_adi_traversals,
        'valid_angle_paths': valid_angle_paths,
        'total_danger_penalty': total_danger_penalty,
        'network_complexity': complexity,
        'avg_path_length': avg_path_length,
        'paths': paths,
        'adi_traversal_metrics': adi_traversal_metrics,
        'angle_metrics': angle_metrics
    }

    return results


def extract_features_for_gnn(
        nodes: np.ndarray,
        node_types: List[int],
        edges: List[Tuple[int, int]],
        adi_zones: List[Dict],
        danger_zones: List[Dict]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features for graph neural network.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        node_types: List of node types (0: frontline, 1: airport, 2: common, 3: outlier)
        edges: List of edges as tuples of node indices
        adi_zones: List of ADI zone parameters
        danger_zones: List of danger zone parameters

    Returns:
        Tuple of (node_features, edge_indices, edge_features)
    """
    num_nodes = len(nodes)

    # 确定节点类型的最大值，用于one-hot编码
    max_node_type = max(node_types)
    unique_node_types = max_node_type + 1  # 0到max_node_type的数量

    # 修复: 调整节点特征维度，确保one-hot编码与节点类型匹配
    # node_features: [x, y, type_onehot[unique_node_types], distances_to_adi_zones[len(adi_zones)]]
    node_features = np.zeros((num_nodes, 2 + unique_node_types + len(adi_zones)))

    # Copy coordinates
    node_features[:, 0:2] = nodes

    # One-hot encode node types
    for i, node_type in enumerate(node_types):
        node_features[i, 2 + node_type] = 1.0

    # Calculate distances to ADI zones
    for i in range(num_nodes):
        node_point = tuple(nodes[i])

        for j, zone in enumerate(adi_zones):
            center = (zone['center_x'], zone['center_y'])
            distance = distance_point_to_point(node_point, center)

            # Normalize by outer radius
            normalized_distance = distance / zone['epsilon']
            node_features[i, 2 + unique_node_types + j] = normalized_distance

    # Edge indices and features
    if not edges:
        # Return empty arrays if no edges
        edge_indices = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, 1 + len(adi_zones) + 1))
        return node_features, edge_indices, edge_features

    # Edge indices: [2, num_edges]
    edge_indices = np.array([[e[0], e[1]] for e in edges], dtype=np.int64).T

    # Edge features: [length, crosses_adi_outer[len(adi_zones)], danger_penalty]
    edge_features = np.zeros((len(edges), 1 + len(adi_zones) + 1))

    for e_idx, (i, j) in enumerate(edges):
        start_point = tuple(nodes[i])
        end_point = tuple(nodes[j])
        line = (start_point, end_point)

        # Calculate length
        length = distance_point_to_point(start_point, end_point)
        edge_features[e_idx, 0] = length

        # Check if it crosses ADI zones' outer circles
        for z_idx, zone in enumerate(adi_zones):
            _, crosses_outer = does_line_cross_adi_zone(line, zone)
            if crosses_outer:
                edge_features[e_idx, 1 + z_idx] = 1.0

        # Calculate danger penalty
        danger_penalty = get_danger_zone_penalty(line, danger_zones)
        edge_features[e_idx, -1] = danger_penalty

    return node_features, edge_indices, edge_features