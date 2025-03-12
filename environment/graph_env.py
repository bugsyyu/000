"""
Graph construction environment for reinforcement learning.
"""

import numpy as np
import gym
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
import copy
import networkx as nx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import (
    distance_point_to_point,
    is_line_segment_valid,
    does_line_cross_adi_zone,
    get_danger_zone_penalty
)
from environment.utils import (
    calculate_connectivity_score,
    evaluate_network,
    extract_features_for_gnn
)

class GraphConstructionEnv(gym.Env):
    """
    Environment for constructing a graph/network by adding edges between nodes.

    State space:
        - Node features (position, type, etc.)
        - Current graph adjacency structure
        - ADI zone parameters

    Action space:
        - Discrete: Select an edge to add from the set of valid potential edges
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        nodes: np.ndarray,
        node_types: List[int],
        adi_zones: List[Dict],
        danger_zones: List[Dict],
        frontline_indices: List[int],
        airport_indices: List[int],
        max_edges: int = 100,
        max_angle_deg: float = 80.0,
        render_mode: str = None
    ):
        super(GraphConstructionEnv, self).__init__()

        self.nodes = np.array(nodes)
        self.node_types = node_types
        self.adi_zones = adi_zones
        self.danger_zones = danger_zones
        self.frontline_indices = frontline_indices
        self.airport_indices = airport_indices
        self.max_edges = max_edges
        self.max_angle_deg = max_angle_deg
        self.render_mode = render_mode

        self.num_nodes = len(nodes)
        self.num_frontline = len(frontline_indices)
        self.num_airports = len(airport_indices)

        # Initialize empty graph
        self.edges = []
        self.valid_potential_edges = []
        self.valid_mask = None

        # Define action space
        # We'll dynamically update the action space as edges are added
        self.action_space = spaces.Discrete(1)  # Placeholder, updated in reset

        # Define observation space
        # We'll use a Dict space with:
        # - 'node_features': [num_nodes, feature_dim]
        # - 'edge_indices': [2, num_edges]
        # - 'edge_features': [num_edges, edge_feature_dim]
        # - 'valid_mask': [num_potential_edges]

        # 修复: 使node_feature_dim匹配extract_features_for_gnn中的实际维度
        # 修改前: node_feature_dim = 2 + 4 + len(self.adi_zones)
        # 查看node_types中的类型取值范围，确保one-hot编码维度正确
        unique_node_types = len(set(node_types))
        node_feature_dim = 2 + unique_node_types + len(self.adi_zones)  # x, y, type_onehot, distances_to_adi

        edge_feature_dim = 1 + len(self.adi_zones) + 1  # length, crosses_adi_outer[3], danger

        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_nodes, node_feature_dim),
                dtype=np.float32
            ),
            'edge_indices': spaces.Box(
                low=0,
                high=self.num_nodes - 1,
                shape=(2, self.max_edges),  # Padded with zeros if fewer edges
                dtype=np.int64
            ),
            'edge_features': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.max_edges, edge_feature_dim),
                dtype=np.float32
            ),
            'valid_mask': spaces.Box(
                low=0,
                high=1,
                shape=(self.num_nodes * self.num_nodes,),  # Upper bound, will be smaller in practice
                dtype=bool
            )
        })

        # Define rewards
        self.reward_edge_added = 1.0
        self.reward_edge_invalid = -1.0
        self.reward_connectivity_improvement = 5.0
        self.reward_adi_traversal = 2.0
        self.reward_danger_penalty_factor = -1.0
        self.reward_completion = 20.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and reset info dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Clear edges
        self.edges = []

        # Initialize valid potential edges
        self._update_valid_potential_edges()

        # Update action space
        self.action_space = spaces.Discrete(len(self.valid_potential_edges))

        # Get current observation
        obs = self._get_observation()

        # Return the observation and empty info dict to comply with SB3 interface
        return obs, {}

    def step(self, action):
        """
        Take a step in the environment by adding an edge.

        Args:
            action: Index into valid_potential_edges list

        Returns:
            (observation, reward, done, truncated, info)
        """
        # Check if action is valid
        if action >= len(self.valid_potential_edges):
            obs = self._get_observation()
            return obs, self.reward_edge_invalid, False, False, {'reason': 'Invalid action index'}

        # Get the edge to add
        edge = self.valid_potential_edges[action]
        i, j = edge

        # Extract node coordinates
        start_point = tuple(self.nodes[i])
        end_point = tuple(self.nodes[j])

        # Check if the edge is valid
        is_valid, reason, penalty = is_line_segment_valid(
            start_point,
            end_point,
            self.adi_zones,
            self.edges,
            self.nodes,
            self.max_angle_deg
        )

        if not is_valid:
            # This shouldn't happen since we precompute valid edges,
            # but we'll handle it gracefully
            obs = self._get_observation()
            return obs, self.reward_edge_invalid, False, False, {'reason': reason}

        # Add the edge
        self.edges.append(edge)

        # Update valid potential edges
        self._update_valid_potential_edges()

        # Update action space
        self.action_space = spaces.Discrete(len(self.valid_potential_edges))

        # Calculate reward
        reward = self._calculate_reward(edge)

        # Check if we're done (all frontline-airport pairs are connected)
        connectivity_score, num_connected_pairs, _ = calculate_connectivity_score(
            self.nodes, self.edges, self.frontline_indices, self.airport_indices
        )

        total_pairs = self.num_frontline * self.num_airports
        done = (num_connected_pairs == total_pairs) or (len(self.edges) >= self.max_edges)

        # Add completion bonus if all pairs are connected
        if num_connected_pairs == total_pairs:
            reward += self.reward_completion

        # Get new observation
        obs = self._get_observation()

        return obs, reward, done, False, {
            'num_connected_pairs': num_connected_pairs,
            'total_pairs': total_pairs,
            'connectivity_score': connectivity_score,
            'num_edges': len(self.edges)
        }

    def render(self, mode='human'):
        """
        Render the environment.
        """
        # Rendering is handled externally with visualization.py
        pass

    def close(self):
        """
        Close the environment.
        """
        pass

    def get_network(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        """
        Get the current network.

        Returns:
            Tuple of (nodes, node_types, edges)
        """
        return self.nodes, self.node_types, self.edges

    def get_network_evaluation(self) -> Dict[str, Any]:
        """
        Get a comprehensive evaluation of the current network.

        Returns:
            Dictionary of evaluation metrics
        """
        return evaluate_network(
            self.nodes,
            self.edges,
            self.frontline_indices,
            self.airport_indices,
            self.adi_zones,
            self.danger_zones,
            self.max_angle_deg
        )

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation.

        Returns:
            Observation dictionary
        """
        # Extract features for graph neural network
        node_features, edge_indices, edge_features = extract_features_for_gnn(
            self.nodes,
            self.node_types,
            self.edges,
            self.adi_zones,
            self.danger_zones
        )

        # Pad edge_indices and edge_features if needed
        num_edges = edge_indices.shape[1] if edge_indices.ndim > 1 else 0

        if num_edges < self.max_edges:
            # Pad edge_indices
            padded_edge_indices = np.zeros((2, self.max_edges), dtype=np.int64)
            if num_edges > 0:
                padded_edge_indices[:, :num_edges] = edge_indices

            # Pad edge_features
            edge_feature_dim = edge_features.shape[1] if edge_features.ndim > 1 else (1 + len(self.adi_zones) + 1)
            padded_edge_features = np.zeros((self.max_edges, edge_feature_dim), dtype=np.float32)
            if num_edges > 0:
                padded_edge_features[:num_edges] = edge_features
        else:
            padded_edge_indices = edge_indices
            padded_edge_features = edge_features

        # Create valid action mask
        valid_mask = np.zeros(len(self.valid_potential_edges), dtype=bool)
        valid_mask[:] = True

        # Create the observation dictionary
        obs = {
            'node_features': node_features.astype(np.float32),
            'edge_indices': padded_edge_indices,
            'edge_features': padded_edge_features.astype(np.float32),
            'valid_mask': valid_mask
        }

        return obs

    def _update_valid_potential_edges(self):
        """
        Update the list of valid potential edges.
        """
        valid_edges = []

        # Check all possible edges
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):  # Only consider i < j to avoid duplicates
                # Skip if edge already exists
                if (i, j) in self.edges or (j, i) in self.edges:
                    continue

                # Extract node coordinates
                start_point = tuple(self.nodes[i])
                end_point = tuple(self.nodes[j])

                # Check if the edge is valid
                is_valid, _, _ = is_line_segment_valid(
                    start_point,
                    end_point,
                    self.adi_zones,
                    self.edges,
                    self.nodes,
                    self.max_angle_deg
                )

                if is_valid:
                    valid_edges.append((i, j))

        self.valid_potential_edges = valid_edges

    def _calculate_reward(self, edge: Tuple[int, int]) -> float:
        """
        Calculate the reward for adding an edge.

        Args:
            edge: The edge that was added (i, j)

        Returns:
            Reward value
        """
        i, j = edge

        # Basic reward for adding a valid edge
        reward = self.reward_edge_added

        # Extract node coordinates
        start_point = tuple(self.nodes[i])
        end_point = tuple(self.nodes[j])
        line = (start_point, end_point)

        # Check if the edge traverses an ADI zone's outer ring
        crosses_adi_outer = False

        for zone in self.adi_zones:
            _, crosses_outer = does_line_cross_adi_zone(line, zone)
            if crosses_outer:
                crosses_adi_outer = True
                break

        # Reward for traversing an ADI zone's outer ring
        if crosses_adi_outer:
            reward += self.reward_adi_traversal

        # Penalty for crossing danger zones
        danger_penalty = get_danger_zone_penalty(line, self.danger_zones)
        reward += self.reward_danger_penalty_factor * danger_penalty

        # Evaluate connectivity before and after adding the edge
        edges_before = [e for e in self.edges if e != edge]  # Remove the edge temporarily

        connectivity_before, num_connected_before, _ = calculate_connectivity_score(
            self.nodes, edges_before, self.frontline_indices, self.airport_indices
        )

        connectivity_after, num_connected_after, _ = calculate_connectivity_score(
            self.nodes, self.edges, self.frontline_indices, self.airport_indices
        )

        # Reward for improving connectivity
        connectivity_improvement = num_connected_after - num_connected_before
        reward += self.reward_connectivity_improvement * connectivity_improvement

        return reward