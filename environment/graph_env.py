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
    does_line_cross_circle,
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

    We fix the action space dimension at environment init:
    - action_space = Discrete(max_possible_edges)
      (or just num_nodes^2 to be safe)
    Each action index corresponds to "which valid edge index to pick"
    in the current valid_potential_edges list. If the chosen action >= len(valid_potential_edges),
    we treat it as invalid.
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

        # Keep track of existing edges
        self.edges = []
        self.valid_potential_edges = []
        self.valid_mask = None

        # ---------- Key fix: do NOT dynamically set action_space dimension ----------
        # We'll pick a stable upper bound. For safety we use total possible edges = n*(n-1)//2
        # If you prefer, you can also use self.num_nodes * self.num_nodes
        self.max_possible_edges = self.num_nodes * (self.num_nodes - 1) // 2
        if self.max_possible_edges < 1:
            self.max_possible_edges = 1  # fallback to 1 if there's only 1 node
        self.action_space = spaces.Discrete(self.max_possible_edges)

        # ----- Construct observation space -----
        # Node feature dim: x, y + onehot(node_type) + distance_to_each_ADI
        unique_node_types = max(node_types) + 1
        node_feature_dim = 2 + unique_node_types + len(self.adi_zones)
        # Edge feature dim: length + crosses_adi(= len(adi_zones)?) + danger
        edge_feature_dim = 1 + len(self.adi_zones) + 1

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
                shape=(2, self.max_edges),
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
                shape=(self.num_nodes * self.num_nodes,),
                dtype=bool
            )
        })

        # Rewards
        self.reward_edge_added = 1.0
        self.reward_edge_invalid = -1.0
        self.reward_connectivity_improvement = 5.0
        self.reward_adi_traversal = 2.0
        self.reward_danger_penalty_factor = -1.0
        self.reward_completion = 20.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.edges = []
        self._update_valid_potential_edges()

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        """
        action is an integer in [0, self.max_possible_edges-1].
        If action >= len(valid_potential_edges), treat as invalid.
        Otherwise, pick that edge from valid_potential_edges.
        """
        # 1) Check if action is in range
        if action >= len(self.valid_potential_edges):
            # Invalid action => negative reward, not done
            obs = self._get_observation()
            return obs, self.reward_edge_invalid, False, False, {'reason': 'action index out of range'}

        # 2) Retrieve the chosen edge
        edge = self.valid_potential_edges[action]
        i, j = edge

        start_point = tuple(self.nodes[i])
        end_point = tuple(self.nodes[j])

        # 3) Check if the chosen edge is valid (should be if it's in valid_potential_edges, but double-check)
        is_valid, reason, penalty = is_line_segment_valid(
            start_point,
            end_point,
            self.adi_zones,
            self.edges,
            self.nodes,
            self.node_types,
            self.airport_indices,
            self.max_angle_deg
        )
        if not is_valid:
            # theoretically不应该出现，但以防万一
            obs = self._get_observation()
            return obs, self.reward_edge_invalid, False, False, {'reason': reason}

        # 4) Add edge
        self.edges.append(edge)

        # 5) Update valid potential edges
        self._update_valid_potential_edges()

        # 6) Calculate reward for adding the edge
        reward = self._calculate_reward(edge)

        # 7) Check connectivity
        connectivity_score, num_connected_pairs, _ = calculate_connectivity_score(
            self.nodes,
            self.edges,
            self.frontline_indices,
            self.airport_indices,
            node_types=self.node_types
        )
        total_pairs = self.num_frontline * self.num_airports

        # Done conditions:
        done = False
        # a) all pairs connected
        if num_connected_pairs == total_pairs:
            reward += self.reward_completion
            done = True
        # b) we hit the maximum edges
        elif len(self.edges) >= self.max_edges:
            done = True
        # c) no valid edges remain => can't add more edges => if not connected => done
        elif len(self.valid_potential_edges) == 0:
            done = True

        obs = self._get_observation()
        info = {
            'num_connected_pairs': num_connected_pairs,
            'total_pairs': total_pairs,
            'connectivity_score': connectivity_score,
            'num_edges': len(self.edges)
        }

        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_network(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        """
        Return the final (nodes, node_types, edges).
        """
        return self.nodes, self.node_types, self.edges

    def get_network_evaluation(self) -> Dict[str, Any]:
        """
        Evaluate the network with new ADI / front-airport logic.
        """
        return evaluate_network(
            self.nodes,
            self.edges,
            self.frontline_indices,
            self.airport_indices,
            self.node_types,
            self.adi_zones,
            self.danger_zones,
            self.max_angle_deg
        )

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Build the GNN-based observation dict.
        """
        node_features, edge_indices, edge_features = extract_features_for_gnn(
            self.nodes,
            self.node_types,
            self.edges,
            self.adi_zones,
            self.danger_zones
        )

        num_edges = edge_indices.shape[1] if edge_indices.ndim > 1 else 0

        # Pad edge_indices/features to self.max_edges
        if num_edges < self.max_edges:
            padded_edge_indices = np.zeros((2, self.max_edges), dtype=np.int64)
            if num_edges > 0:
                padded_edge_indices[:, :num_edges] = edge_indices

            feat_dim = edge_features.shape[1] if edge_features.ndim > 1 else (1 + len(self.adi_zones) + 1)
            padded_edge_features = np.zeros((self.max_edges, feat_dim), dtype=np.float32)
            if num_edges > 0:
                padded_edge_features[:num_edges] = edge_features
        else:
            padded_edge_indices = edge_indices
            padded_edge_features = edge_features

        # Build valid_mask
        valid_mask = np.zeros(self.num_nodes * self.num_nodes, dtype=bool)
        for (i, j) in self.valid_potential_edges:
            idx = i * self.num_nodes + j
            valid_mask[idx] = True

        obs = {
            'node_features': node_features.astype(np.float32),
            'edge_indices': padded_edge_indices,
            'edge_features': padded_edge_features.astype(np.float32),
            'valid_mask': valid_mask
        }
        return obs

    def _update_valid_potential_edges(self):
        """
        Recompute the list of all edges that can still be added without violating constraints.
        (Does not set action_space. We keep action_space fixed.)
        """
        valid_edges = []
        existing = set(self.edges) | set((b, a) for (a, b) in self.edges)

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if (i, j) in existing:
                    continue
                # frontlines can't connect frontlines, airports can't connect airports
                if (i in self.frontline_indices and j in self.frontline_indices) or \
                   (i in self.airport_indices and j in self.airport_indices):
                    continue

                start_point = tuple(self.nodes[i])
                end_point = tuple(self.nodes[j])

                # 20km no-fly around airports
                skip_edge = False
                for airport_idx in self.airport_indices:
                    if i == airport_idx or j == airport_idx:
                        continue  # The line can start/end at the airport itself
                    airport_point = tuple(self.nodes[airport_idx])
                    if does_line_cross_circle((start_point, end_point), (airport_point, 20.0)):
                        skip_edge = True
                        break
                if skip_edge:
                    continue

                # run geometry check
                is_valid_seg, _, _ = is_line_segment_valid(
                    start_point,
                    end_point,
                    self.adi_zones,
                    self.edges,
                    self.nodes,
                    self.node_types,
                    self.airport_indices,
                    self.max_angle_deg
                )
                if is_valid_seg:
                    valid_edges.append((i, j))

        self.valid_potential_edges = valid_edges

    def _calculate_reward(self, edge: Tuple[int, int]) -> float:
        """
        Reward function after adding an edge.
        """
        i, j = edge
        reward = self.reward_edge_added

        start_point = tuple(self.nodes[i])
        end_point = tuple(self.nodes[j])
        line = (start_point, end_point)

        # Check if crosses ADI outer
        crosses_adi_outer = False
        for zone in self.adi_zones:
            _, crosses_outer = does_line_cross_adi_zone(line, zone)
            if crosses_outer:
                crosses_adi_outer = True
                break
        if crosses_adi_outer:
            reward += self.reward_adi_traversal

        # Danger penalty
        danger_penalty = get_danger_zone_penalty(line, self.danger_zones)
        reward += self.reward_danger_penalty_factor * danger_penalty

        # Connectivity improvement
        edges_before = [e for e in self.edges if e != edge]
        conn_before, num_conn_before, _ = calculate_connectivity_score(
            self.nodes,
            edges_before,
            self.frontline_indices,
            self.airport_indices,
            node_types=self.node_types
        )
        conn_after, num_conn_after, _ = calculate_connectivity_score(
            self.nodes,
            self.edges,
            self.frontline_indices,
            self.airport_indices,
            node_types=self.node_types
        )

        improvement = num_conn_after - num_conn_before
        reward += self.reward_connectivity_improvement * improvement

        return reward
