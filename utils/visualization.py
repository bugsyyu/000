"""
Visualization utilities for the airspace network planning system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx

def plot_airspace_network(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    node_types: List[int],
    adi_zones: List[Dict],
    danger_zones: List[Dict],
    connected_pairs: List[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Airspace Network"
):
    """
    Plot the airspace network with all relevant components.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        node_types: List of node types (0: frontline, 1: airport, 2: common, 3: outlier)
        adi_zones: List of ADI zone parameters
        danger_zones: List of danger zone parameters
        connected_pairs: List of directly connected frontline-airport pairs
        figsize: Figure size
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ADI zones
    for zone in adi_zones:
        center = (zone['center_x'], zone['center_y'])
        inner_radius = zone['radius']
        outer_radius = zone['epsilon']

        # Outer circle (lighter)
        outer_circle = Circle(center, outer_radius, fill=True, alpha=0.2, color='blue')
        ax.add_patch(outer_circle)

        # Inner circle (darker)
        inner_circle = Circle(center, inner_radius, fill=True, alpha=0.5, color='red')
        ax.add_patch(inner_circle)

    # Plot danger zones
    for zone in danger_zones:
        center = (zone['center_x'], zone['center_y'])
        radius = zone['radius']
        threat_level = zone['threat_level']

        # Color based on threat level (red intensity)
        color = (threat_level, 0, 0, 0.3)
        circle = Circle(center, radius, fill=True, color=color)
        ax.add_patch(circle)

    # Create a graph for edges
    G = nx.Graph()

    # Plot nodes
    node_colors = []
    node_sizes = []

    for i, node_type in enumerate(node_types):
        if node_type == 0:  # Frontline
            node_colors.append('green')
            node_sizes.append(100)
        elif node_type == 1:  # Airport
            node_colors.append('blue')
            node_sizes.append(100)
        elif node_type == 2:  # Common
            node_colors.append('gray')
            node_sizes.append(50)
        elif node_type == 3:  # Outlier
            node_colors.append('purple')
            node_sizes.append(50)

        G.add_node(i, pos=(nodes[i][0], nodes[i][1]))

    # Add edges to the graph
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, ax=ax)

    # Highlight connected pairs if provided
    if connected_pairs:
        # Create a subgraph of connected pairs
        H = nx.Graph()
        for start, end in connected_pairs:
            H.add_edge(start, end)
            H.nodes[start]['pos'] = pos[start]
            H.nodes[end]['pos'] = pos[end]

        # Draw the connected pairs with a different color
        pos_H = nx.get_node_attributes(H, 'pos')
        nx.draw_networkx_edges(H, pos_H, width=2.0, edge_color='red', ax=ax)

    # Add labels for nodes
    node_labels = {}
    for i in range(len(nodes)):
        if node_types[i] in [0, 1]:  # Only label frontline and airport nodes
            node_labels[i] = str(i)

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

    # Set plot limits with some padding
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]

    padding = 20  # km
    plt.xlim(min(x_coords) - padding, max(x_coords) + padding)
    plt.ylim(min(y_coords) - padding, max(y_coords) + padding)

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')

    # Add a legend
    frontline = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Frontline')
    airport = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Airport')
    common = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Common Node')
    outlier = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Outlier Node')
    inner_adi = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', alpha=0.5, markersize=10, label='Inner ADI (No-Fly)')
    outer_adi = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', alpha=0.2, markersize=10, label='Outer ADI (Recognition)')
    danger = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', alpha=0.3, markersize=10, label='Danger Zone')

    plt.legend(handles=[frontline, airport, common, outlier, inner_adi, outer_adi, danger], loc='upper right')

    plt.tight_layout()

    return fig, ax

def plot_training_progress(
    episode_rewards: List[float],
    episode_lengths: List[int],
    episode_successes: List[bool],
    moving_avg_window: int = 10,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Training Progress"
):
    """
    Plot training progress metrics.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths (number of steps)
        episode_successes: List of episode success flags
        moving_avg_window: Window size for moving average
        figsize: Figure size
        title: Plot title
    """
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Plot rewards
    axs[0].plot(episode_rewards, alpha=0.6, label='Episode Reward')

    # Moving average for rewards
    if len(episode_rewards) >= moving_avg_window:
        moving_avg = [np.mean(episode_rewards[max(0, i-moving_avg_window):i+1])
                     for i in range(len(episode_rewards))]
        axs[0].plot(moving_avg, color='red', label=f'Moving Avg ({moving_avg_window})')

    axs[0].set_ylabel('Reward')
    axs[0].set_title(f'{title} - Rewards')
    axs[0].legend()
    axs[0].grid(True)

    # Plot episode lengths
    axs[1].plot(episode_lengths, alpha=0.6, label='Episode Length')

    # Moving average for episode lengths
    if len(episode_lengths) >= moving_avg_window:
        moving_avg = [np.mean(episode_lengths[max(0, i-moving_avg_window):i+1])
                     for i in range(len(episode_lengths))]
        axs[1].plot(moving_avg, color='red', label=f'Moving Avg ({moving_avg_window})')

    axs[1].set_ylabel('Steps')
    axs[1].set_title(f'{title} - Episode Lengths')
    axs[1].legend()
    axs[1].grid(True)

    # Plot success rate
    success_rate = [sum(episode_successes[:i+1]) / (i+1) for i in range(len(episode_successes))]
    axs[2].plot(success_rate, label='Success Rate')

    # Moving average for success rate
    if len(success_rate) >= moving_avg_window:
        moving_avg = [np.mean(success_rate[max(0, i-moving_avg_window):i+1])
                     for i in range(len(success_rate))]
        axs[2].plot(moving_avg, color='red', label=f'Moving Avg ({moving_avg_window})')

    axs[2].set_ylabel('Success Rate')
    axs[2].set_xlabel('Episodes')
    axs[2].set_title(f'{title} - Success Rate')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

    return fig, axs

def plot_paths_between_points(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    paths: List[List[int]],
    start_indices: List[int],
    end_indices: List[int],
    adi_zones: List[Dict],
    danger_zones: List[Dict],
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Paths Between Points"
):
    """
    Plot paths between specific points in the network.

    Args:
        nodes: Array of shape (n_nodes, 2) containing node coordinates
        edges: List of edges as tuples of node indices
        paths: List of paths, where each path is a list of node indices
        start_indices: List of start node indices
        end_indices: List of end node indices
        adi_zones: List of ADI zone parameters
        danger_zones: List of danger zone parameters
        figsize: Figure size
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ADI zones
    for zone in adi_zones:
        center = (zone['center_x'], zone['center_y'])
        inner_radius = zone['radius']
        outer_radius = zone['epsilon']

        # Outer circle (lighter)
        outer_circle = Circle(center, outer_radius, fill=True, alpha=0.2, color='blue')
        ax.add_patch(outer_circle)

        # Inner circle (darker)
        inner_circle = Circle(center, inner_radius, fill=True, alpha=0.5, color='red')
        ax.add_patch(inner_circle)

    # Plot danger zones
    for zone in danger_zones:
        center = (zone['center_x'], zone['center_y'])
        radius = zone['radius']
        threat_level = zone['threat_level']

        # Color based on threat level (red intensity)
        color = (threat_level, 0, 0, 0.3)
        circle = Circle(center, radius, fill=True, color=color)
        ax.add_patch(circle)

    # Create a graph for all edges
    G = nx.Graph()

    # Add all nodes and edges
    for i in range(len(nodes)):
        G.add_node(i, pos=(nodes[i][0], nodes[i][1]))

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw all nodes and edges with light color
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=30, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, ax=ax)

    # Highlight paths
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))

    for i, path in enumerate(paths):
        # Create a path graph
        P = nx.Graph()
        for j in range(len(path) - 1):
            P.add_edge(path[j], path[j+1])
            P.nodes[path[j]]['pos'] = pos[path[j]]
            P.nodes[path[j+1]]['pos'] = pos[path[j+1]]

        pos_P = nx.get_node_attributes(P, 'pos')

        # Draw the path with a unique color
        nx.draw_networkx_edges(P, pos_P, width=2.0, edge_color=[colors[i]], ax=ax)

    # Highlight start and end nodes
    start_nodes = [nodes[i] for i in start_indices]
    end_nodes = [nodes[i] for i in end_indices]

    ax.scatter([n[0] for n in start_nodes], [n[1] for n in start_nodes],
              color='green', s=100, label='Start Points')
    ax.scatter([n[0] for n in end_nodes], [n[1] for n in end_nodes],
              color='blue', s=100, label='End Points')

    # Add labels for start and end nodes
    for i, idx in enumerate(start_indices):
        ax.annotate(f'S{i}', (nodes[idx][0], nodes[idx][1]),
                   xytext=(5, 5), textcoords='offset points')

    for i, idx in enumerate(end_indices):
        ax.annotate(f'E{i}', (nodes[idx][0], nodes[idx][1]),
                   xytext=(5, 5), textcoords='offset points')

    # Set plot limits with some padding
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]

    padding = 20  # km
    plt.xlim(min(x_coords) - padding, max(x_coords) + padding)
    plt.ylim(min(y_coords) - padding, max(y_coords) + padding)

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')

    # Add a legend
    start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start Points')
    end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='End Points')
    inner_adi = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', alpha=0.5, markersize=10, label='Inner ADI (No-Fly)')
    outer_adi = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', alpha=0.2, markersize=10, label='Outer ADI (Recognition)')
    danger = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', alpha=0.3, markersize=10, label='Danger Zone')

    plt.legend(handles=[start_point, end_point, inner_adi, outer_adi, danger], loc='upper right')

    plt.tight_layout()

    return fig, ax