"""
Simplified training script for quick testing of the airspace network planning system,
with logs now directed to both console and a dedicated file.
"""

import logging
import os
import argparse
import numpy as np
import torch
import random
import time
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

from config.latlon_config import latlon_config
from utils.coordinate_transform import transform_config_to_cartesian
from environment.node_env import NodePlacementEnv
from environment.graph_env import GraphConstructionEnv
from utils.visualization import plot_airspace_network
from environment.utils import evaluate_network

logger = logging.getLogger(__name__)

def create_nodes_manually(cartesian_config: Dict[str, Any], num_intermediate: int = 10) -> Tuple[np.ndarray, List[int]]:
    """
    Create nodes manually without training.

    Args:
        cartesian_config: Configuration with Cartesian coordinates
        num_intermediate: Number of intermediate nodes to create

    Returns:
        Tuple of (nodes, node_types)
    """
    # Extract fixed nodes (frontline and airports)
    frontline_points = np.array([
        [point['x'], point['y']] for point in cartesian_config['frontline']
    ])
    airport_points = np.array([
        [airport['x'], airport['y']] for airport in cartesian_config['airports']
    ])

    # Combine fixed nodes
    fixed_nodes = np.vstack([frontline_points, airport_points])

    # Create intermediate nodes by placing them strategically
    intermediate_nodes = []

    # Calculate the bounds
    min_x = np.min(fixed_nodes[:, 0])
    max_x = np.max(fixed_nodes[:, 0])
    min_y = np.min(fixed_nodes[:, 1])
    max_y = np.max(fixed_nodes[:, 1])
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Add some nodes near ADI zones
    for adi_zone in cartesian_config['adi_zones']:
        center_x = adi_zone['center_x']
        center_y = adi_zone['center_y']
        outer_radius = adi_zone['epsilon']

        # Add nodes around outer edge
        for i in range(3):  # Add 3 nodes per ADI zone
            angle = random.uniform(0, 2 * np.pi)
            distance = outer_radius * random.uniform(0.9, 1.1)  # Close to outer edge

            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)
            intermediate_nodes.append([x, y])

    # Add some random nodes
    remaining = num_intermediate - len(intermediate_nodes)
    for _ in range(remaining):
        # Place near random fixed node
        rand_idx = random.randrange(len(fixed_nodes))
        base_x, base_y = fixed_nodes[rand_idx]

        # Add random offset (20-50km)
        offset_dist = random.uniform(20, 50)
        angle = random.uniform(0, 2 * np.pi)
        x = base_x + offset_dist * np.cos(angle)
        y = base_y + offset_dist * np.sin(angle)
        intermediate_nodes.append([x, y])

    # Combine all nodes
    all_nodes = np.vstack([fixed_nodes, np.array(intermediate_nodes)])

    # Create node types list
    # 0: frontline, 1: airport, 2: common intermediate, 3: outlier intermediate
    node_types = [0]*len(frontline_points) + [1]*len(airport_points)

    # Randomly assign intermediate nodes as common or outlier
    for _ in range(len(intermediate_nodes)):
        # 70% common, 30% outlier
        if random.random() < 0.7:
            node_types.append(2)  # Common
        else:
            node_types.append(3)  # Outlier

    return all_nodes, node_types

def create_edges_manually(nodes: np.ndarray, node_types: List[int], cartesian_config: Dict[str, Any], max_edges: int = 50) -> List[Tuple[int, int]]:
    """
    Create edges manually without training.

    Args:
        nodes: Node coordinates
        node_types: Node types
        cartesian_config: Configuration with Cartesian coordinates
        max_edges: Maximum number of edges to create

    Returns:
        List of edges as (node1_idx, node2_idx)
    """
    # Extract frontline and airport indices
    frontline_indices = [i for i, t in enumerate(node_types) if t == 0]
    airport_indices = [i for i, t in enumerate(node_types) if t == 1]
    intermediate_indices = [i for i, t in enumerate(node_types) if t >= 2]

    # Create edges by connecting nodes
    edges = []

    # Connect frontline points to nearest intermediate nodes
    for front_idx in frontline_indices:
        front_point = nodes[front_idx]

        # Find closest intermediate node
        min_dist = float('inf')
        closest_idx = None
        for inter_idx in intermediate_indices:
            inter_point = nodes[inter_idx]
            dist = np.sqrt(np.sum((front_point - inter_point) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_idx = inter_idx
        if closest_idx is not None:
            edges.append((front_idx, closest_idx))

    # Connect airport points to nearest intermediate nodes
    for airport_idx in airport_indices:
        airport_point = nodes[airport_idx]

        # Find closest intermediate node
        min_dist = float('inf')
        closest_idx = None
        for inter_idx in intermediate_indices:
            inter_point = nodes[inter_idx]
            dist = np.sqrt(np.sum((airport_point - inter_point) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_idx = inter_idx
        if closest_idx is not None:
            edges.append((airport_idx, closest_idx))

    # Connect intermediate nodes to form a network
    # Use a simple strategy: connect each intermediate node to its K nearest neighbors
    K = 2  # Number of nearest neighbors to connect

    for i, idx1 in enumerate(intermediate_indices):
        point1 = nodes[idx1]
        distances = []

        # Calculate distances to all other intermediate nodes
        for j, idx2 in enumerate(intermediate_indices):
            if i == j:
                continue
            point2 = nodes[idx2]
            dist = np.sqrt(np.sum((point1 - point2) ** 2))
            distances.append((dist, idx2))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Connect to K nearest neighbors
        for k in range(min(K, len(distances))):
            _, idx2 = distances[k]
            # Create edge (if not already exists)
            if (idx1, idx2) not in edges and (idx2, idx1) not in edges:
                edges.append((idx1, idx2))

                # Break if we've reached max edges
                if len(edges) >= max_edges:
                    break

        # Break if we've reached max edges
        if len(edges) >= max_edges:
            break
    return edges

def test_environment(nodes: np.ndarray, node_types: List[int], cartesian_config: Dict[str, Any], edges: List[Tuple[int, int]], max_steps: int = 10):
    """
    Test the graph construction environment with manual nodes and edges.

    Args:
        nodes: Node coordinates
        node_types: Node types
        cartesian_config: Configuration with Cartesian coordinates
        edges: Edges to start with
        max_steps: Maximum steps to run
    """
    logger.info("Testing GraphConstructionEnv environment...")

    # Extract frontline and airport indices
    frontline_indices = [i for i, t in enumerate(node_types) if t == 0]
    airport_indices = [i for i, t in enumerate(node_types) if t == 1]

    # Create environment
    env = GraphConstructionEnv(
        nodes=nodes,
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        frontline_indices=frontline_indices,
        airport_indices=airport_indices,
        max_edges=100,
        max_angle_deg=80.0,
        log_level=3
    )

    # Reset environment
    reset_result = env.reset()
    obs = reset_result[0]
    logger.info("Observation keys: %s", obs.keys())
    # print(f"Node features shape: {obs['node_features'].shape}")
    # print(f"Valid mask shape: {obs['valid_mask'].shape}")

    # Run a few steps
    for i in range(max_steps):
        # Sample a random action
        action = env.action_space.sample()
        logger.info("Step %d: Taking action %s", i+1, action)
        # Take step
        obs, reward, done, _, info = env.step(action)
        logger.info("  Reward: %.3f, Done=%s, Info=%s", reward, done, info)
        if done:
            logger.info("Environment done at step %d.", i+1)
            break

    # Get the final network
    _, _, final_edges = env.get_network()
    logger.info("Final edges count: %d", len(final_edges))
    return final_edges

def main():
    """
    Main function to run the quick testing script.
    """
    parser = argparse.ArgumentParser(description='Quick test for the airspace network planning system.')
    parser.add_argument('--output_dir', type=str, default='./quick_output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_intermediate', type=int, default=20, help='Number of intermediate nodes')
    parser.add_argument('--max_edges', type=int, default=50, help='Maximum number of edges')
    parser.add_argument('--test_env', action='store_true', help='Test environment directly')
    parser.add_argument('--log_level', type=int, default=2, help='Logging verbosity level (1=minimal,2=info,3=debug)')

    args = parser.parse_args()

    # Determine logging level
    # level = logging.DEBUG if args.log_level >= 3 else (logging.INFO if args.log_level == 2 else logging.WARNING)
    level = logging.INFO
    # Ensure output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up log file path
    log_file = os.path.join(args.output_dir, 'train_quick.log')

    # Configure logging to file + console
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Starting quick test of airspace network planning...")

    # Transform latitude-longitude config to Cartesian coordinates
    cartesian_config = transform_config_to_cartesian(latlon_config)
    logger.info("Transforming coordinates...")

    # Create nodes manually
    logger.info("Creating nodes manually...")
    nodes, node_types = create_nodes_manually(
        cartesian_config=cartesian_config,
        num_intermediate=args.num_intermediate
    )
    logger.info("Created %d nodes (%d intermediate nodes)", len(nodes), args.num_intermediate)

    print(f"Created {len(nodes)} nodes ({args.num_intermediate} intermediate nodes)")

    # Save nodes
    np.savez(
        os.path.join(args.output_dir, 'quick_nodes.npz'),
        nodes=nodes,
        node_types=node_types
    )

    # Visualize nodes
    fig, ax = plot_airspace_network(
        nodes=nodes,
        edges=[],
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        title='Quick Node Placement'
    )
    plt.savefig(os.path.join(args.output_dir, 'quick_nodes.png'))
    plt.close(fig)

    logger.info("Creating edges manually...")
    edges = create_edges_manually(
        nodes=nodes,
        node_types=node_types,
        cartesian_config=cartesian_config,
        max_edges=args.max_edges
    )
    logger.info("Created %d edges", len(edges))

    # Test environment if requested
    if args.test_env:
        final_edges = test_environment(
            nodes=nodes,
            node_types=node_types,
            cartesian_config=cartesian_config,
            edges=edges,
            max_steps=10
        )
        # Use the edges from environment test
        if final_edges:
            edges = final_edges

    # Save edges
    np.savez(
        os.path.join(args.output_dir, 'quick_edges.npz'),
        edges=np.array(edges)
    )

    # Save the complete network
    np.savez(
        os.path.join(args.output_dir, 'quick_network.npz'),
        nodes=nodes,
        node_types=node_types,
        edges=np.array(edges)
    )

    # Visualize network
    fig, ax = plot_airspace_network(
        nodes=nodes,
        edges=edges,
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        title='Quick Network Planning'
    )
    plt.savefig(os.path.join(args.output_dir, 'quick_network.png'))
    plt.close(fig)

    logger.info("Quick test completed. Results saved to %s", args.output_dir)
    logger.info("To evaluate the full network, run: python evaluate.py --input_file %s/quick_network.npz --output_dir %s/evaluation",
                args.output_dir, args.output_dir)


if __name__ == '__main__':
    main()
