"""
Evaluation script for the airspace network planning system.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import json

from config.latlon_config import latlon_config
from utils.coordinate_transform import transform_config_to_cartesian, cartesian_to_lat_lon
from environment.node_env import NodePlacementEnv
from environment.graph_env import GraphConstructionEnv
from utils.visualization import plot_airspace_network, plot_paths_between_points
from environment.utils import evaluate_network, find_shortest_paths


def evaluate_network_plan(
        nodes: np.ndarray,
        node_types: List[int],
        edges: List[Tuple[int, int]],
        cartesian_config: Dict[str, Any],
        output_dir: str
):
    """
    Evaluate a network plan.

    Args:
        nodes: Node coordinates
        node_types: Node types
        edges: Edges as tuples of node indices
        cartesian_config: Configuration with Cartesian coordinates
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract frontline and airport indices
    frontline_indices = [i for i, t in enumerate(node_types) if t == 0]
    airport_indices = [i for i, t in enumerate(node_types) if t == 1]

    # Get network evaluation
    evaluation = evaluate_network(
        nodes=nodes,
        edges=edges,
        frontline_indices=frontline_indices,
        airport_indices=airport_indices,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        min_angle_deg=80.0
    )

    # Save evaluation results
    with open(os.path.join(output_dir, 'network_evaluation.txt'), 'w') as f:
        for key, value in evaluation.items():
            if key not in ['paths', 'adi_traversal_metrics', 'angle_metrics']:
                f.write(f"{key}: {value}\n")

    # Visualize the network
    fig, ax = plot_airspace_network(
        nodes=nodes,
        edges=edges,
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        title='Airspace Network Plan'
    )

    plt.savefig(os.path.join(output_dir, 'network_plan.png'))
    plt.close(fig)

    # Visualize all paths
    # Group paths by ADI zone traversal patterns
    path_groups = {}

    for i, path in enumerate(evaluation['paths']):
        if not path:  # Skip empty paths
            continue

        start_idx = frontline_indices[i // len(airport_indices)]
        end_idx = airport_indices[i % len(airport_indices)]

        # Get ADI traversal pattern as a tuple of booleans
        adi_pattern = tuple(evaluation['adi_traversal_metrics'][i]) if i < len(
            evaluation['adi_traversal_metrics']) else tuple([False] * len(cartesian_config['adi_zones']))

        if adi_pattern not in path_groups:
            path_groups[adi_pattern] = {
                'paths': [],
                'start_indices': [],
                'end_indices': []
            }

        path_groups[adi_pattern]['paths'].append(path)
        path_groups[adi_pattern]['start_indices'].append(start_idx)
        path_groups[adi_pattern]['end_indices'].append(end_idx)

    # Visualize each group
    for i, (adi_pattern, group) in enumerate(path_groups.items()):
        if len(group['paths']) > 10:
            # If too many paths in a group, sample 10
            indices = np.random.choice(len(group['paths']), 10, replace=False)
            sample_paths = [group['paths'][j] for j in indices]
            sample_start_indices = [group['start_indices'][j] for j in indices]
            sample_end_indices = [group['end_indices'][j] for j in indices]
        else:
            sample_paths = group['paths']
            sample_start_indices = group['start_indices']
            sample_end_indices = group['end_indices']

        fig, ax = plot_paths_between_points(
            nodes=nodes,
            edges=edges,
            paths=sample_paths,
            start_indices=sample_start_indices,
            end_indices=sample_end_indices,
            adi_zones=cartesian_config['adi_zones'],
            danger_zones=cartesian_config['danger_zones'],
            title=f'Paths for ADI Traversal Pattern {i + 1}'
        )

        plt.savefig(os.path.join(output_dir, f'paths_group_{i + 1}.png'))
        plt.close(fig)

    # Convert to geographic coordinates
    geo_nodes = []
    ref_lat = cartesian_config['reference']['lat']
    ref_lon = cartesian_config['reference']['lon']

    for node in nodes:
        lat, lon = cartesian_to_lat_lon(node[0], node[1], ref_lat, ref_lon)
        geo_nodes.append((lat, lon))

    # Save geojson format - 修改这部分以处理NumPy类型
    geojson = {
        'type': 'FeatureCollection',
        'features': []
    }

    # Add nodes
    for i, (lat, lon) in enumerate(geo_nodes):
        node_type = int(node_types[i])  # 转换为标准 Python int
        node_type_str = ['frontline', 'airport', 'common', 'outlier'][node_type]

        geojson['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [float(lon), float(lat)]  # 转换为标准 Python float
            },
            'properties': {
                'id': int(i),  # 转换为标准 Python int
                'type': node_type_str
            }
        })

    # Add edges
    for i, j in edges:
        geojson['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [
                    [float(geo_nodes[i][1]), float(geo_nodes[i][0])],  # 转换为标准 Python float
                    [float(geo_nodes[j][1]), float(geo_nodes[j][0])]  # 转换为标准 Python float
                ]
            },
            'properties': {
                'from_node': int(i),  # 转换为标准 Python int
                'to_node': int(j)  # 转换为标准 Python int
            }
        })

    # Add ADI zones
    for i, zone in enumerate(cartesian_config['adi_zones']):
        geojson['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [float(zone['original_lon']), float(zone['original_lat'])]  # 转换为标准 Python float
            },
            'properties': {
                'type': 'adi_zone',
                'id': int(i),  # 转换为标准 Python int
                'inner_radius': float(zone['radius']),  # 转换为标准 Python float
                'outer_radius': float(zone['epsilon']),  # 转换为标准 Python float
                'weapon_type': str(zone['weapon_type'])  # 确保字符串类型
            }
        })

    # 使用自定义编码器保存 geojson
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Save geojson
    with open(os.path.join(output_dir, 'network_plan.geojson'), 'w') as f:
        json.dump(geojson, f, indent=2, cls=NumpyEncoder)  # 使用自定义编码器


def main():
    """
    Main function to run the evaluation process.
    """
    parser = argparse.ArgumentParser(description='Evaluate the airspace network planning system.')
    parser.add_argument('--input_file', type=str, required=True, help='Input file with network data')
    parser.add_argument('--output_dir', type=str, default='./evaluation', help='Output directory')

    args = parser.parse_args()

    # Load network data
    data = np.load(args.input_file)
    nodes = data['nodes']
    node_types = data['node_types']
    edges = data['edges']

    if edges.ndim == 2 and edges.shape[1] == 2:
        # Convert to list of tuples
        edges = [tuple(edge) for edge in edges]

    # Transform latitude-longitude config to Cartesian coordinates
    cartesian_config = transform_config_to_cartesian(latlon_config)

    # Evaluate the network
    evaluate_network_plan(nodes, node_types, edges, cartesian_config, args.output_dir)

    print(f"Evaluation results saved to {args.output_dir}")


if __name__ == '__main__':
    main()