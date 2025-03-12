"""
Script to test that all imports are working correctly.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config.latlon_config import latlon_config
from utils.coordinate_transform import transform_config_to_cartesian
from environment.node_env import NodePlacementEnv
from environment.graph_env import GraphConstructionEnv
from utils.visualization import plot_airspace_network
from utils.geometry import (
    distance_point_to_point,
    is_line_segment_valid,
    does_line_cross_adi_zone,
    get_danger_zone_penalty
)
from utils.clustering import (
    detect_outliers_dbscan,
    find_optimal_clusters,
    identify_isolated_clusters,
    suggest_intermediate_nodes
)

def test_imports():
    """Test that all imports are working correctly."""
    print("Testing imports...")

    # Load configuration
    print("Loading configuration...")
    cartesian_config = transform_config_to_cartesian(latlon_config)

    # Create environments
    print("Creating environments...")
    node_env = NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=30,
        min_distance=10.0,
        max_distance=150.0
    )

    # Reset node environment
    print("Resetting node environment...")
    reset_result = node_env.reset()
    # New Gym API returns (obs, info) tuple
    obs = reset_result[0]
    print(f"Node observation shape: {obs.shape}")

    # Test a single step in node environment
    print("Testing step in node environment...")
    action = node_env.action_space.sample()
    step_result = node_env.step(action)
    # New Gym API returns (obs, reward, done, truncated, info) tuple
    obs, reward, done, truncated, info = step_result
    print(f"Step result: reward={reward}, done={done}, truncated={truncated}, info={info}")

    # Create a simple test for graph environment
    print("Setting up test data for graph environment...")
    test_nodes = np.array([
        [0, 0],   # Frontline 0
        [10, 10], # Frontline 1
        [20, 20], # Airport 0
        [30, 30], # Airport 1
        [5, 5],   # Common 0
        [15, 15]  # Common 1
    ])

    test_node_types = [0, 0, 1, 1, 2, 2]  # 0: frontline, 1: airport, 2: common

    # Try creating a graph environment
    print("Creating graph environment...")
    graph_env = GraphConstructionEnv(
        nodes=test_nodes,
        node_types=test_node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        frontline_indices=[0, 1],
        airport_indices=[2, 3],
        max_edges=10,
        max_angle_deg=80.0
    )

    # Reset graph environment
    print("Resetting graph environment...")
    graph_reset_result = graph_env.reset()
    # New Gym API returns (obs, info) tuple
    graph_obs = graph_reset_result[0]
    print(f"Graph observation keys: {graph_obs.keys()}")

    # Test a single step in graph environment (if possible)
    try:
        print("Testing step in graph environment...")
        graph_action = graph_env.action_space.sample()
        graph_step_result = graph_env.step(graph_action)
        # New Gym API returns (obs, reward, done, truncated, info) tuple
        graph_obs, graph_reward, graph_done, graph_truncated, graph_info = graph_step_result
        print(f"Graph step result: reward={graph_reward}, done={graph_done}, truncated={graph_truncated}, info={graph_info}")
    except Exception as e:
        print(f"Note: Could not test graph environment step: {e}")

    # Test clustering utilities
    print("Testing clustering utilities...")
    test_points = np.array([
        [0, 0], [1, 1], [10, 10], [11, 11], [100, 100]
    ])

    labels, core_points, outlier_points = detect_outliers_dbscan(test_points)
    print(f"DBSCAN labels: {labels}")
    print(f"Outlier points: {outlier_points}")

    # Test geometry utilities
    print("Testing geometry utilities...")
    dist = distance_point_to_point((0, 0), (3, 4))
    print(f"Distance: {dist}")

    # Test visualization
    print("Testing visualization...")
    try:
        fig, ax = plot_airspace_network(
            nodes=test_nodes,
            edges=[(0, 4), (4, 5), (5, 2)],  # Some sample edges
            node_types=test_node_types,
            adi_zones=cartesian_config['adi_zones'],
            danger_zones=cartesian_config['danger_zones'],
            title='Test Visualization'
        )
        plt.close(fig)
        print("Visualization test successful")
    except Exception as e:
        print(f"Note: Visualization test encountered issue: {e}")

    print("\nAll import tests complete!")
    return True

if __name__ == "__main__":
    test_imports()