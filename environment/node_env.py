"""
Node placement environment for reinforcement learning, supporting log_level for debug prints.

Key points:
  - We introduce log_level to control output verbosity:
    log_level=1: minimal logs
    log_level=2: print info each episode (or each invalid action)
    log_level=3: print step-level debug info
  - We unify debug_invalid_placement with log_level:
    Only if debug_invalid_placement=True AND log_level >= 2, we print invalid placement reasons.

修复要点：
  - 增加对连续非法放置次数的计数，一旦过多，就提前 done=True，避免陷入无限非法动作循环。
  - 使环境不会在大量回合内都只得到 "Too close to intermediate node" 而无从收敛。
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
          node_type: 0 for common node, 1 for outlier node

    We add log_level and debug_invalid_placement to unify output control:
      - debug_invalid_placement: master switch to enable/disable printing invalid reasons at all
      - log_level: if debug_invalid_placement=True, only print reasons if log_level>=2
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        cartesian_config: Dict[str, Any],
        max_nodes: int = 30,
        min_distance: float = 10.0,   # Minimum distance (km) between new node and existing node
        max_distance: float = 150.0,  # Maximum distance for valid node placement
        render_mode: str = None,
        debug_invalid_placement: bool = True,
        log_level: int = 1,
        # ---------------- 新增可配置参数 ----------------
        max_invalid_placements_per_episode: int = 50
    ):
        super(NodePlacementEnv, self).__init__()

        self.cartesian_config = cartesian_config
        self.max_nodes = max_nodes
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.render_mode = render_mode

        # Debug/Logging control
        self.debug_invalid_placement = debug_invalid_placement
        self.log_level = log_level

        # 新增：当单回合非法放置次数超过此阈值，则提前结束
        self.max_invalid_placements_per_episode = max_invalid_placements_per_episode
        self.invalid_placement_count = 0  # 计数器

        # Extract data from cartesian_config
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
        padding = 50.0
        self.x_min, self.x_max = x_min - padding, x_max + padding
        self.y_min, self.y_max = y_min - padding, y_max + padding

        # Action space = (x, y, node_type)
        # node_type is in [0,1], but we keep the box continuous in [0,1]
        self.action_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, 0]),
            high=np.array([self.x_max, self.y_max, 1]),
            dtype=np.float32
        )

        # We'll flatten our observation:
        # - fixed_nodes (n_fixed * 3): [x, y, is_frontline?]
        # - intermediate_nodes (max_nodes * 3): [x, y, is_outlier?]
        # - adi_zones (#adi_zones * 4): [center_x, center_y, inner_radius, outer_radius]
        # - 1 extra for remaining_nodes
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

        # Rewards
        self.reward_node_valid = 1.0
        self.reward_node_invalid = -1.0
        self.reward_node_near_adi = 2.0
        self.reward_outlier_valid = 3.0
        self.reward_efficiency = 0.5

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

        # Reset非法动作计数器
        self.invalid_placement_count = 0

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
        node_type = int(round(node_type_float))

        # Check if we've placed all available nodes
        if self.remaining_nodes <= 0:
            obs = self._get_observation()
            return obs, 0.0, True, False, {'reason': 'No more nodes available'}

        # Check if node placement is valid
        is_valid, validity_reward, reason = self._check_node_validity(x, y)

        done = False
        truncated = False
        info_extra = {}

        if is_valid:
            # Valid placement
            self.intermediate_nodes.append([x, y])
            self.intermediate_node_types.append(node_type)
            self.remaining_nodes -= 1

            # Additional logic
            reward = validity_reward

            # Reward for placing nodes near suggested positions
            if self._suggested_nodes is not None and len(self._suggested_nodes) > 0:
                for suggested_node in self._suggested_nodes:
                    dist = np.sqrt((x - suggested_node[0])**2 + (y - suggested_node[1])**2)
                    if dist < 20.0:
                        reward += self.reward_efficiency
                        break

            # Additional reward for valid outlier node placement
            if node_type == 1:
                if self._is_outlier_region(x, y):
                    reward += self.reward_outlier_valid

            # If we've used up all placements, we are done
            if self.remaining_nodes <= 0:
                done = True

            obs = self._get_observation()
            info_extra = {'reason': 'Valid placement', 'is_outlier': (node_type == 1)}

        else:
            # Invalid
            # Only print reason if debug_invalid_placement=True and log_level > 2
            if self.debug_invalid_placement and self.log_level > 2:
                print(f"[NodePlacementEnv] Invalid node (x={x:.2f}, y={y:.2f}, type={node_type}): {reason}")

            self.invalid_placement_count += 1  # 累计非法动作
            reward = validity_reward  # 通常为负值

            # 如果超过阈值，则结束本回合
            if self.invalid_placement_count >= self.max_invalid_placements_per_episode:
                done = True
                info_extra = {
                    'reason': f'Max invalid placements reached: {self.invalid_placement_count}'
                }

            obs = self._get_observation()

        return obs, reward, done, truncated, info_extra

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
            np.array(self.intermediate_nodes) if self.intermediate_nodes else np.zeros((0,2))
        ])

        # Create node types list
        # 0: frontline, 1: airport, 2: common intermediate, 3: outlier intermediate
        node_types = []

        # Add fixed node types
        num_frontline = len(self.frontline_points)
        num_airports = len(self.airport_points)
        node_types.extend([0]*num_frontline)  # 0=frontline
        node_types.extend([1]*num_airports)   # 1=airport

        # Add intermediate node types
        for nt in self.intermediate_node_types:
            node_types.append(2+nt)  # 2=common,3=outlier

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
        # 1) Bounds
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False, self.reward_node_invalid, "Out of bounds"

        # 2) Inside ADI inner zone?
        for zone in self.adi_zones:
            center = (zone['center_x'], zone['center_y'])
            inner_radius = zone['radius']
            if is_point_in_circle((x,y), (center, inner_radius)):
                return False, self.reward_node_invalid, "Inside ADI inner zone"

        # 3) Too close to existing nodes
        for node in self.fixed_nodes:
            dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
            if dist < self.min_distance:
                return False, self.reward_node_invalid, "Too close to fixed node"
        for node in self.intermediate_nodes:
            dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
            if dist < self.min_distance:
                return False, self.reward_node_invalid, "Too close to intermediate node"

        # 4) Too far from any node
        all_current_nodes = np.vstack([self.fixed_nodes, np.array(self.intermediate_nodes) if self.intermediate_nodes else np.zeros((0,2))])
        min_dist_to_any = float('inf')
        for node in all_current_nodes:
            dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
            if dist < min_dist_to_any:
                min_dist_to_any = dist
        if min_dist_to_any > self.max_distance:
            return False, self.reward_node_invalid, "Too far from any node"

        # 5) near ADI outer => extra reward
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
            np.array(self.intermediate_nodes) if self.intermediate_nodes else np.zeros((0,2))
        ])
        dists = [np.sqrt((x - n[0])**2 + (y - n[1])**2) for n in all_nodes]
        if min(dists) > 50.0:
            return True
        if np.mean(dists) > 100.0:
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
