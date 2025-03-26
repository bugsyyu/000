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
    end_indices: List[int],
    node_types: Optional[List[int]] = None
) -> Tuple[List[List[int]], List[bool]]:
    """
    Find shortest paths between multiple start and end points, subject to special constraints.

    - If node_types is None, we fall back to the original logic (no special filtering).
    - If node_types is provided, then for each front_i in start_indices and airport_j in end_indices,
      we only allow traveling through:
         1) The start node front_i itself
         2) The end node airport_j itself
         3) Any node whose type >= 2 (common or outlier), i.e. not another frontline/airport
      This ensures we do NOT pass through other frontlines or other airports as intermediate nodes.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        start_indices: List of start node indices (frontlines)
        end_indices: List of end node indices (airports)
        node_types: Optional list of node types (0: frontline, 1: airport, 2+: intermediate).
                    If provided, will enforce that paths cannot go through other frontlines/airports.

    Returns:
        Tuple of (paths, success_flags)
          - paths: in the same order of the nested loops for front in start_indices, end in end_indices
          - success_flags: True/False for each pair
    """
    # If node_types is None, we do the original approach (old logic).
    if node_types is None:
        # -- Original logic: single global graph, try Nx shortest_path for each pair --
        G = nx.Graph()
        # First ensure all nodes are added so that isolated nodes are recognized
        for i in range(len(nodes)):
            G.add_node(i)

        # Add weighted edges
        for i, j in edges:
            weight = distance_point_to_point(nodes[i], nodes[j])
            G.add_edge(i, j, weight=weight)

        paths = []
        success_flags = []

        for start_idx in start_indices:
            for end_idx in end_indices:
                try:
                    if nx.has_path(G, source=start_idx, target=end_idx):
                        path = nx.shortest_path(G, source=start_idx, target=end_idx, weight='weight')
                        paths.append(path)
                        success_flags.append(True)
                    else:
                        paths.append([])
                        success_flags.append(False)
                except nx.NetworkXNoPath:
                    paths.append([])
                    success_flags.append(False)
                except nx.NodeNotFound as e:
                    # If any node not found in graph, treat as no path
                    paths.append([])
                    success_flags.append(False)

        return paths, success_flags

    # ----------------------------------------------------------
    # New logic: for each (frontline, airport) pair, build a subgraph
    # that excludes any other frontline or airport except that pair.
    # ----------------------------------------------------------
    all_paths = []
    success_flags = []

    # Precompute adjacency info so we don't rebuild from scratch
    adjacency_list = {}
    for i in range(len(nodes)):
        adjacency_list[i] = []
    for (i, j) in edges:
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    for f in start_indices:
        for a in end_indices:
            # Create a subgraph G_sub with:
            #   - node f, node a
            #   - all nodes of type >= 2
            # Then add edges among them
            valid_nodes = set()
            valid_nodes.add(f)
            valid_nodes.add(a)
            for idx, t in enumerate(node_types):
                if t >= 2:
                    valid_nodes.add(idx)

            G_sub = nx.Graph()
            # Add valid nodes
            for vn in valid_nodes:
                G_sub.add_node(vn)

            # Add edges among valid nodes
            for vn in valid_nodes:
                for neighbor in adjacency_list[vn]:
                    if neighbor in valid_nodes:
                        # Add edge with weight
                        dist = distance_point_to_point(nodes[vn], nodes[neighbor])
                        G_sub.add_edge(vn, neighbor, weight=dist)

            # Check path
            if nx.has_path(G_sub, f, a):
                path = nx.shortest_path(G_sub, f, a, weight='weight')
                all_paths.append(path)
                success_flags.append(True)
            else:
                all_paths.append([])
                success_flags.append(False)

    return all_paths, success_flags


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

    zone_traversed = [False] * len(adi_zones)

    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]

        start_point = tuple(nodes[start_idx])
        end_point = tuple(nodes[end_idx])
        line = (start_point, end_point)

        for j, zone in enumerate(adi_zones):
            crosses_inner, crosses_outer = does_line_cross_adi_zone(line, zone)
            # If crosses inner zone, that typically means invalid path
            # but we only mark zone_traversed false. The path might still exist for the overall net
            if crosses_inner:
                zone_traversed[j] = False

            if crosses_outer:
                zone_traversed[j] = True

    return zone_traversed

def calculate_connectivity_score(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    frontline_indices: List[int],
    airport_indices: List[int],
    node_types: Optional[List[int]] = None
) -> Tuple[float, int, List[List[int]]]:
    """
    Calculate connectivity score between frontline and airport nodes.
    IMPORTANT: If node_types is provided, we forbid passing through other frontlines/airports
               except the pair in question.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        frontline_indices: List of frontline node indices
        airport_indices: List of airport node indices
        node_types: Optional list of node types. If provided, means we must not
                    pass through other frontlines/airports for the path.

    Returns:
        Tuple of (connectivity_score, num_connected_pairs, paths)
          - connectivity_score: ratio of connected pairs over total pairs
          - num_connected_pairs: how many pairs actually connected
          - paths: the actual paths used for each pair (flattened list in the order of the nested loops)
    """
    paths, success_flags = find_shortest_paths(
        nodes, edges, frontline_indices, airport_indices, node_types=node_types
    )

    # Count successful paths
    num_connected_pairs = sum(success_flags)

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

        v1 = np.array(curr_point) - np.array(prev_point)
        v2 = np.array(next_point) - np.array(curr_point)

        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)

    return angles

def evaluate_network(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    frontline_indices: List[int],
    airport_indices: List[int],
    node_types: List[int],
    adi_zones: List[Dict],
    danger_zones: List[Dict],
    min_angle_deg: float = 80.0
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the airspace network.

    New: We require node_types to enforce the "no-other-frontline/airport-in-between" rule
         in connectivity checks.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        frontline_indices: List of frontline node indices
        airport_indices: List of airport node indices
        node_types: List of node types (0: frontline, 1: airport, 2: common, 3: outlier)
        adi_zones: List of ADI zone parameters
        danger_zones: List of danger zone parameters
        min_angle_deg: Minimum allowed angle in degrees

    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate connectivity (with new rule of no-other-frontline/airport bridging)
    connectivity_score, num_connected_pairs, paths = calculate_connectivity_score(
        nodes, edges, frontline_indices, airport_indices, node_types=node_types
    )

    # Check ADI zone traversal
    valid_adi_traversals = 0
    adi_traversal_metrics = []

    for path in paths:
        if not path:
            continue

        zone_traversals = check_adi_zone_traversal(path, nodes, adi_zones)
        adi_traversal_metrics.append(zone_traversals)

        if any(zone_traversals):
            valid_adi_traversals += 1

    # Check path angles
    valid_angle_paths = 0
    angle_metrics = []

    for path in paths:
        if not path or len(path) < 3:
            continue

        angles = check_path_angles(path, nodes, min_angle_deg)
        angle_metrics.append(angles)

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

    # Network complexity
    complexity = len(edges) / len(nodes) if len(nodes) > 0 else 0.0

    # Average path length
    total_length = 0.0
    count = 0
    for path in paths:
        if not path:
            continue

        path_length = 0.0
        for k in range(len(path) - 1):
            start_idx = path[k]
            end_idx = path[k + 1]

            segment_length = distance_point_to_point(nodes[start_idx], nodes[end_idx])
            path_length += segment_length

        total_length += path_length
        count += 1

    avg_path_length = total_length / count if count > 0 else 0.0

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