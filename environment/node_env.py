"""
Node placement environment (simplified) with improved "strategic" node placement logic.

这里在 reset() 时基于借鉴的“改进节点放置算法”一次性确定所有中间节点，
然后在 step() 中直接返回 done=True，不再进行深度强化学习式的逐步搜索。
这样可快速完成节点放置，并避免反复采样造成的大量无效动作。

具体逻辑写在 _place_nodes_geometrically() 中，
参考了此前给出的“place_strategic_nodes”思路，并做了适配与简化。
"""

import logging
import numpy as np
import gym
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
import math

# Add parent directory to path (so we can import from ../utils if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import is_point_in_circle, distance_point_to_point

logger = logging.getLogger(__name__)

class NodePlacementEnv(gym.Env):
    """
    Environment for placing intermediate nodes with an improved deterministic strategy
    (rather than random RL exploration).

    Use-cases:
      - If you prefer a guaranteed geometric approach to place nodes between frontlines and airports,
        near ADI outer rings, and in some random fallback, you can do so here.
      - We unify everything into the reset() method => once environment is reset, all nodes are placed.
      - Then step() just returns done=True, effectively skipping multi-step RL.

    Node Types:
      0: frontline
      1: airport
      2: normal intermediate
      3: special intermediate

    Basic usage remains the same as the old RL environment. But effectively
    the "training" on this env will finish quickly since there's no multi-step flow.
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        cartesian_config: Dict[str, Any],
        max_nodes: int = 30,
        min_distance: float = 10.0,
        max_distance: float = 150.0,
        render_mode: str = None,
        debug_invalid_placement: bool = True,
        log_level: int = 1,
        max_invalid_placements_per_episode: int = 50
    ):
        super(NodePlacementEnv, self).__init__()

        self.cartesian_config = cartesian_config
        self.max_nodes = max_nodes
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.render_mode = render_mode

        self.debug_invalid_placement = debug_invalid_placement
        self.log_level = log_level

        self.max_invalid_placements_per_episode = max_invalid_placements_per_episode

        # Extract fixed (frontline + airport) nodes
        self.frontline_points = np.array([
            [point['x'], point['y']] for point in cartesian_config['frontline']
        ])
        self.airport_points = np.array([
            [airport['x'], airport['y']] for airport in cartesian_config['airports']
        ])
        self.fixed_nodes = np.vstack([self.frontline_points, self.airport_points])
        self.num_fixed_nodes = len(self.fixed_nodes)

        # For reference only (e.g. bounding the random fallback):
        all_x = np.concatenate([self.frontline_points[:, 0], self.airport_points[:, 0]])
        all_y = np.concatenate([self.frontline_points[:, 1], self.airport_points[:, 1]])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        padding = 50.0
        self.x_min, self.x_max = x_min - padding, x_max + padding
        self.y_min, self.y_max = y_min - padding, y_max + padding

        self.adi_zones = cartesian_config['adi_zones']
        self.danger_zones = cartesian_config['danger_zones']

        # Action space: still a 3D Box for (x,y,node_type), but effectively unused
        self.action_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, 0]),
            high=np.array([self.x_max, self.y_max, 1]),
            dtype=np.float32
        )

        # We keep the same shape for observation as the old approach, for SB3 compatibility
        num_adi_zones = len(self.adi_zones)
        obs_size = (self.num_fixed_nodes * 3) + (max_nodes * 3) + (num_adi_zones * 4) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.intermediate_nodes = []
        self.intermediate_node_types = []
        self.remaining_nodes = self.max_nodes

        if self.log_level >= 1:
            logger.info(
                "[NodePlacementEnv] Using a strategic deterministic placement approach. "
                "max_nodes=%d, min_distance=%.1f, max_distance=%.1f",
                self.max_nodes, self.min_distance, self.max_distance
            )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)

        # Perform the strategic/deterministic node placement
        self._place_nodes_geometrically()

        # We'll consider that we've placed all nodes at once
        self.remaining_nodes = 0

        obs = self._get_observation()
        if self.log_level >= 2:
            logger.info(
                "[NodePlacementEnv] Reset => placed %d intermediate nodes. (limit=%d)",
                len(self.intermediate_nodes), self.max_nodes
            )
        return obs, {}

    def step(self, action):
        """
        We skip step-by-step logic: immediately done.
        """
        obs = self._get_observation()
        reward = 0.0
        done = True
        truncated = False
        info = {'reason': 'Nodes placed in reset()'}

        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_full_nodes(self) -> Tuple[np.ndarray, List[int]]:
        """
        Return (nodes, node_types).
          0: frontline
          1: airport
          2: normal intermediate
          3: special intermediate
        """
        all_nodes = np.vstack([
            self.fixed_nodes,
            np.array(self.intermediate_nodes) if self.intermediate_nodes else np.zeros((0, 2))
        ])

        # Build node types
        num_frontline = len(self.frontline_points)
        num_airports = len(self.airport_points)
        node_types = [0] * num_frontline + [1] * num_airports
        for t in self.intermediate_node_types:
            node_types.append(t)

        return all_nodes, node_types

    def _get_observation(self) -> np.ndarray:
        """
        Flatten everything to match the old observation shape.
        """
        fixed_obs = np.zeros((self.num_fixed_nodes, 3))
        num_frontline = len(self.frontline_points)
        # Mark frontline
        fixed_obs[:num_frontline, :2] = self.frontline_points
        fixed_obs[:num_frontline, 2] = 1.0
        # Mark airports
        fixed_obs[num_frontline:, :2] = self.airport_points
        fixed_obs[num_frontline:, 2] = 0.0

        interm_obs = np.zeros((self.max_nodes, 3))
        n_interm = len(self.intermediate_nodes)
        if n_interm > 0:
            interm_obs[:n_interm, :2] = np.array(self.intermediate_nodes)
            interm_obs[:n_interm, 2] = np.array(self.intermediate_node_types)

        adi_obs = np.zeros((len(self.adi_zones), 4))
        for i, zone in enumerate(self.adi_zones):
            adi_obs[i, 0] = zone['center_x']
            adi_obs[i, 1] = zone['center_y']
            adi_obs[i, 2] = zone['radius']
            adi_obs[i, 3] = zone['epsilon']

        counter_obs = np.array([self.remaining_nodes])

        obs = np.concatenate([
            fixed_obs.flatten(),
            interm_obs.flatten(),
            adi_obs.flatten(),
            counter_obs
        ]).astype(np.float32)
        return obs

    def _place_nodes_geometrically(self):
        """
        Core logic for strategic deterministic placement.

        参考了 `place_strategic_nodes()` 的思路:
          1) 在 ADI 外圈附近放置一批节点。
          2) 在前沿点->机场连线之间均匀布点。
          3) 若仍不足，随机补足若干节点，保证距离不小于min_distance、不大于max_distance。
        """

        # Start fresh
        self.intermediate_nodes = []
        self.intermediate_node_types = []

        # Some references
        frontline_points = self.frontline_points
        airport_points = self.airport_points
        fixed_nodes = self.fixed_nodes
        adi_zones = self.adi_zones

        num_frontline = len(frontline_points)
        num_airports = len(airport_points)

        # 1) Place around ADI outer ring
        if self.log_level >= 2:
            logger.info("[NodePlacementEnv] Step1: Place nodes near ADI outer ring.")
        for zone_idx, zone in enumerate(adi_zones):
            center = (zone['center_x'], zone['center_y'])
            outer_radius = zone['epsilon']
            inner_radius = zone['radius']

            # We attempt ~6 nodes around each zone (tweak if needed)
            num_points = min(6, max(1, self.max_nodes // max(1, len(adi_zones))))
            angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)

            for angle in angles:
                # place ~5% outside the outer radius
                dist = outer_radius * 1.05
                x = center[0] + dist * math.cos(angle)
                y = center[1] + dist * math.sin(angle)

                if not self._valid_node(x, y, fixed_nodes, self.intermediate_nodes):
                    continue

                self.intermediate_nodes.append([x, y])
                # normal intermediate => type=2
                self.intermediate_node_types.append(2)

                if len(self.intermediate_nodes) >= self.max_nodes:
                    break
            if len(self.intermediate_nodes) >= self.max_nodes:
                break

        # 2) Place nodes along frontline->airport
        if len(self.intermediate_nodes) < self.max_nodes:
            if self.log_level >= 2:
                logger.info("[NodePlacementEnv] Step2: Place nodes between frontline & airport pairs.")

            remaining = self.max_nodes - len(self.intermediate_nodes)
            total_pairs = num_frontline * num_airports
            if total_pairs > 0:
                nodes_per_pair = max(1, remaining // total_pairs)
            else:
                nodes_per_pair = 1  # fallback

            for f_idx in range(num_frontline):
                for a_idx in range(num_airports):
                    front = frontline_points[f_idx]
                    airp = airport_points[a_idx]
                    dx = airp[0] - front[0]
                    dy = airp[1] - front[1]

                    for i in range(nodes_per_pair):
                        if len(self.intermediate_nodes) >= self.max_nodes:
                            break
                        ratio = (i + 1) / (nodes_per_pair + 1)
                        x = front[0] + dx * ratio
                        y = front[1] + dy * ratio

                        if not self._valid_node(x, y, fixed_nodes, self.intermediate_nodes):
                            continue

                        # Let's pick type=3 in the middle, else 2
                        if nodes_per_pair > 1 and i == nodes_per_pair // 2:
                            node_type = 3
                        else:
                            node_type = 2

                        self.intermediate_nodes.append([x, y])
                        self.intermediate_node_types.append(node_type)

                    if len(self.intermediate_nodes) >= self.max_nodes:
                        break
                if len(self.intermediate_nodes) >= self.max_nodes:
                    break

        # 3) random fallback if still not enough
        if len(self.intermediate_nodes) < self.max_nodes:
            if self.log_level >= 2:
                logger.info("[NodePlacementEnv] Step3: random fallback for the rest.")
            attempts = 0
            max_attempts = 1000
            while len(self.intermediate_nodes) < self.max_nodes and attempts < max_attempts:
                attempts += 1
                x = np.random.uniform(self.x_min, self.x_max)
                y = np.random.uniform(self.y_min, self.y_max)

                if not self._valid_node(x, y, fixed_nodes, self.intermediate_nodes):
                    continue

                # also check if too far from everything
                if not self._within_max_range(x, y, fixed_nodes, self.intermediate_nodes):
                    continue

                self.intermediate_nodes.append([x, y])
                self.intermediate_node_types.append(2)

        if self.log_level >= 2:
            logger.info("[NodePlacementEnv] Placed total %d intermediate nodes (limit=%d).",
                        len(self.intermediate_nodes), self.max_nodes)

    def _valid_node(self, x: float, y: float,
                    fixed_nodes: np.ndarray,
                    placed_nodes: List[List[float]]) -> bool:
        """
        Check if a candidate node is valid:
          - Not inside any ADI inner circle
          - Not within min_distance of existing nodes
        """
        # 1) not in ADI's inner zone
        for z in self.adi_zones:
            if is_point_in_circle((x, y), ((z['center_x'], z['center_y']), z['radius'])):
                return False
        # 2) not too close to any existing node
        for node in fixed_nodes:
            if distance_point_to_point((x, y), tuple(node)) < self.min_distance:
                return False
        for node in placed_nodes:
            if distance_point_to_point((x, y), tuple(node)) < self.min_distance:
                return False
        return True

    def _within_max_range(self, x: float, y: float,
                          fixed_nodes: np.ndarray,
                          placed_nodes: List[List[float]]) -> bool:
        """
        Check if the candidate node is not too far from EVERY existing node
        (i.e. it must be within self.max_distance to at least one node).
        """
        found_close = False
        # check fixed
        for node in fixed_nodes:
            if distance_point_to_point((x, y), tuple(node)) <= self.max_distance:
                found_close = True
                break
        # check placed
        if not found_close:
            for node in placed_nodes:
                if distance_point_to_point((x, y), tuple(node)) <= self.max_distance:
                    found_close = True
                    break
        return found_close
