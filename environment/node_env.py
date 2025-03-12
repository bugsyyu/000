"""
Node placement environment for reinforcement learning.
"""

import numpy as np
import gym
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
import copy

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import is_point_in_circle
from utils.clustering import detect_outliers_dbscan, suggest_intermediate_nodes

class NodePlacementEnv(gym.Env):
    """
    Environment for placing intermediate nodes in the airspace network.

    State space:
        - Fixed nodes (frontline and airport points)
        - Currently placed intermediate nodes
        - ADI zone parameters
        - Number of remaining nodes to place

    Action space:
        - Continuous: (x, y, node_type)
        - node_type: 0 for common node, 1 for outlier node
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        cartesian_config: Dict[str, Any],
        max_nodes: int = 30,
        min_distance: float = 10.0,  # Minimum distance between nodes in km
        max_distance: float = 150.0,  # Maximum distance for valid node placement
        render_mode: str = None
    ):
        super(NodePlacementEnv, self).__init__()

        self.cartesian_config = cartesian_config
        self.max_nodes = max_nodes
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.render_mode = render_mode

        # Extract relevant data from config
        self.frontline_points = np.array([
            [point['x'], point['y']] for point in cartesian_config['frontline']
        ])

        self.airport_points = np.array([
            [airport['x'], airport['y']] for airport in cartesian_config['airports']
        ])

        self.adi_zones = cartesian_config['adi_zones']
        self.danger_zones = cartesian_config['danger_zones']

        # Combine frontline and airport points as fixed nodes
        self.fixed_nodes = np.vstack([self.frontline_points, self.airport_points])
        self.num_fixed_nodes = len(self.fixed_nodes)

        # Compute spatial bounds
        all_x = np.concatenate([self.frontline_points[:, 0], self.airport_points[:, 0]])
        all_y = np.concatenate([self.frontline_points[:, 1], self.airport_points[:, 1]])

        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # Add some padding to bounds
        padding = 50.0  # km
        self.x_min, self.x_max = x_min - padding, x_max + padding
        self.y_min, self.y_max = y_min - padding, y_max + padding

        # Define action space
        # (x, y, node_type)
        self.action_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, 0]),
            high=np.array([self.x_max, self.y_max, 1]),
            dtype=np.float32
        )

        # Define observation space
        # We include:
        # - Fixed nodes: [x, y, is_frontline] * num_fixed_nodes
        # - Current intermediate nodes: [x, y, is_outlier] * max_nodes (padded with zeros)
        # - ADI zones: [center_x, center_y, inner_radius, outer_radius] * num_adi_zones
        # - Remaining nodes counter: [count]

        num_adi_zones = len(self.adi_zones)

        # Calculate total observation size
        obs_size = (self.num_fixed_nodes * 3) + (self.max_nodes * 3) + (num_adi_zones * 4) + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Initialize state
        self.intermediate_nodes = []
        self.intermediate_node_types = []
        self.remaining_nodes = self.max_nodes

        # For clustering
        self._suggested_nodes = None

        # Define rewards
        self.reward_node_valid = 1.0
        self.reward_node_invalid = -1.0
        self.reward_node_near_adi = 2.0
        self.reward_outlier_valid = 3.0
        self.reward_efficiency = 0.5  # For placing nodes in efficient locations

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

        # Clear intermediate nodes
        self.intermediate_nodes = []
        self.intermediate_node_types = []
        self.remaining_nodes = self.max_nodes

        # Detect outliers and suggest initial nodes
        self._suggested_nodes = self._suggest_initial_nodes()

        # Get current observation
        obs = self._get_observation()

        # Return obs and empty info dict to comply with gymnasium/sb3 interface
        return obs, {}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: (x, y, node_type) where node_type is 0 for common, 1 for outlier

        Returns:
            (observation, reward, done, truncated, info)
        """
        # Extract action components
        x, y, node_type_float = action
        node_type = int(round(node_type_float))  # Convert to binary 0 or 1

        # Check if we've placed all available nodes
        if self.remaining_nodes <= 0:
            obs = self._get_observation()
            return obs, 0.0, True, False, {'reason': 'No more nodes available'}

        # Check if node placement is valid
        is_valid, validity_reward, reason = self._check_node_validity(x, y)

        # If valid, add the node
        if is_valid:
            self.intermediate_nodes.append([x, y])
            self.intermediate_node_types.append(node_type)
            self.remaining_nodes -= 1

            # Extra reward logic
            reward = validity_reward

            # Reward for placing nodes near suggested positions
            if self._suggested_nodes is not None and len(self._suggested_nodes) > 0:
                for suggested_node in self._suggested_nodes:
                    dist = np.sqrt((x - suggested_node[0])**2 + (y - suggested_node[1])**2)
                    if dist < 20.0:  # km
                        reward += self.reward_efficiency
                        break

            # Additional reward for valid outlier node placement
            if node_type == 1:
                # Check if it's really an outlier region
                is_outlier_region = self._is_outlier_region(x, y)
                if is_outlier_region:
                    reward += self.reward_outlier_valid

            # Check if we're done
            done = self.remaining_nodes <= 0

            # Get new observation
            obs = self._get_observation()

            return obs, reward, done, False, {'reason': 'Valid placement', 'is_outlier': node_type == 1}
        else:
            # Invalid placement, return negative reward and same state
            obs = self._get_observation()
            return obs, validity_reward, False, False, {'reason': reason}

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

    def get_full_nodes(self) -> Tuple[np.ndarray, List[int]]:
        """
        Get all nodes (fixed + intermediate) and their types.

        Returns:
            Tuple of (nodes, node_types)
        """
        # Combine fixed and intermediate nodes
        all_nodes = np.vstack([
            self.fixed_nodes,
            np.array(self.intermediate_nodes) if self.intermediate_nodes else np.zeros((0, 2))
        ])

        # Create node types list
        # 0: frontline, 1: airport, 2: common intermediate, 3: outlier intermediate
        node_types = []

        # Add fixed node types
        num_frontline = len(self.frontline_points)
        num_airports = len(self.airport_points)

        node_types.extend([0] * num_frontline)  # Frontline points
        node_types.extend([1] * num_airports)  # Airport points

        # Add intermediate node types
        for node_type in self.intermediate_node_types:
            node_types.append(2 + node_type)  # 2 for common, 3 for outlier

        return all_nodes, node_types

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.

        Returns:
            Observation array
        """
        # Prepare fixed nodes section
        fixed_nodes_obs = np.zeros((self.num_fixed_nodes, 3))

        # Add frontline points
        num_frontline = len(self.frontline_points)
        fixed_nodes_obs[:num_frontline, :2] = self.frontline_points
        fixed_nodes_obs[:num_frontline, 2] = 1.0  # is_frontline flag

        # Add airport points
        fixed_nodes_obs[num_frontline:, :2] = self.airport_points
        fixed_nodes_obs[num_frontline:, 2] = 0.0  # not frontline

        # Prepare intermediate nodes section (padded with zeros)
        intermediate_obs = np.zeros((self.max_nodes, 3))

        num_intermediate = len(self.intermediate_nodes)
        if num_intermediate > 0:
            intermediate_obs[:num_intermediate, :2] = np.array(self.intermediate_nodes)
            intermediate_obs[:num_intermediate, 2] = np.array(self.intermediate_node_types)

        # Prepare ADI zones section
        adi_obs = np.zeros((len(self.adi_zones), 4))

        for i, zone in enumerate(self.adi_zones):
            adi_obs[i, 0] = zone['center_x']
            adi_obs[i, 1] = zone['center_y']
            adi_obs[i, 2] = zone['radius']  # Inner radius
            adi_obs[i, 3] = zone['epsilon']  # Outer radius

        # Prepare remaining nodes counter
        counter_obs = np.array([self.remaining_nodes])

        # Combine all observation components
        obs = np.concatenate([
            fixed_nodes_obs.flatten(),
            intermediate_obs.flatten(),
            adi_obs.flatten(),
            counter_obs
        ])

        return obs

    def _check_node_validity(self, x: float, y: float) -> Tuple[bool, float, str]:
        """
        Check if a node placement is valid.

        Args:
            x: X-coordinate of the node
            y: Y-coordinate of the node

        Returns:
            Tuple of (is_valid, reward, reason)
        """
        # Check if node is within bounds
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False, self.reward_node_invalid, "Out of bounds"

        # Check if node is inside any ADI inner zone (not allowed)
        for zone in self.adi_zones:
            center = (zone['center_x'], zone['center_y'])
            inner_radius = zone['radius']

            if is_point_in_circle((x, y), (center, inner_radius)):
                return False, self.reward_node_invalid, "Inside ADI inner zone"

        # Check if node is too close to existing nodes
        for node in self.fixed_nodes:
            dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
            if dist < self.min_distance:
                return False, self.reward_node_invalid, "Too close to fixed node"

        for node in self.intermediate_nodes:
            dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
            if dist < self.min_distance:
                return False, self.reward_node_invalid, "Too close to intermediate node"

        # Check if node is too far from any existing node
        min_dist_to_any = float('inf')
        for node in np.vstack([self.fixed_nodes, np.array(self.intermediate_nodes) if self.intermediate_nodes else np.zeros((0, 2))]):
            dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
            min_dist_to_any = min(min_dist_to_any, dist)

        if min_dist_to_any > self.max_distance:
            return False, self.reward_node_invalid, "Too far from any node"

        # Check if node is near ADI outer zone (preferred)
        near_adi_outer = False
        for zone in self.adi_zones:
            center = (zone['center_x'], zone['center_y'])
            outer_radius = zone['epsilon']
            inner_radius = zone['radius']

            dist_to_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

            # Check if the node is near the outer edge
            if abs(dist_to_center - outer_radius) < 20.0 and dist_to_center > inner_radius:
                near_adi_outer = True
                break

        # Basic reward for valid placement
        reward = self.reward_node_valid

        # Extra reward for placing near ADI outer zone
        if near_adi_outer:
            reward += self.reward_node_near_adi

        return True, reward, "Valid placement"

    def _is_outlier_region(self, x: float, y: float) -> bool:
        """
        Check if a point is in an outlier region.

        Args:
            x: X-coordinate
            y: Y-coordinate

        Returns:
            True if the point is in an outlier region, False otherwise
        """
        # Get all nodes
        all_nodes = np.vstack([
            self.fixed_nodes,
            np.array(self.intermediate_nodes) if self.intermediate_nodes else np.zeros((0, 2))
        ])

        # Calculate distances to all nodes
        distances = []
        for node in all_nodes:
            dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
            distances.append(dist)

        # If the closest node is far away, it's an outlier region
        if min(distances) > 50.0:  # km
            return True

        # If the average distance is high, it's an outlier region
        if np.mean(distances) > 100.0:  # km
            return True

        return False

    def _suggest_initial_nodes(self) -> np.ndarray:
        """
        Suggest initial node placements based on clustering.

        Returns:
            Array of suggested node coordinates
        """
        # Combine frontline and airport points for clustering
        combined_points = np.vstack([self.frontline_points, self.airport_points])

        # Get suggested intermediate nodes
        suggested_nodes = suggest_intermediate_nodes(
            combined_points,
            n_points=len(self.frontline_points),
            max_intermediate_nodes=self.max_nodes,
            isolation_threshold=100.0
        )

        return suggested_nodes