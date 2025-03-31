"""
Node placement environment (simplified) with improved "strategic" node placement logic.

本文件原先的做法是：在 reset() 时基于借鉴的“改进节点放置算法”一次性确定所有中间节点，
然后在 step() 中直接返回 done=True，不再进行深度强化学习式的逐步搜索。
这样可快速完成节点放置，并避免反复采样造成的大量无效动作。

原先的做法是通过 KMeans 将前沿点和机场点各聚为若干类，然后对聚类中心进行合并求几何中心。
此方法比较粗糙，容易产生未充分利用的节点或过多冗余节点，且无法灵活应对多条前沿与机场之间
通过节点灵活调度的需要。

这里我们设计一个全新的、简单但更有效的算法来放置节点，以便后续连线阶段可以有更好的连接潜力：

新算法（“前沿-机场对 midpoints + 去重 + 过滤”）：

1) 收集所有前沿点(frontline_points)和机场(airport_points)。
2) 对于每一个前沿点 f 和机场点 a，计算它们的中点 mid = (f + a)/2。
   - 如果该 mid 处在ADI内圈、机场禁飞圈(20km)、危险区等禁区，则忽略。
   - 如果该 mid 与已有节点（固定或已放置）距离小于 min_distance，则忽略。
3) 将所有合法 midpoints 收集起来，如此会得到 (N_frontline * N_airport) 个候选点，实际可能少于此数量因为某些点位无效或被过滤。
4) 如果最终候选数超过 max_nodes，则用一次 KMeans(n_clusters=max_nodes) 将候选点压缩成 max_nodes 个中心。
   这里的 KMeans 仅作为一个简单降维手段；若不想依赖外部库或想使用其他压缩策略也可。
5) 将这些中心点（或未超限时的所有 midpoints）作为中间节点，node_type=2。
6) 全部一次性放置到 intermediate_nodes 里完成。

注意：此方法非常直观：基本思路是让每条“前沿-机场”直线的中部形成一个可被后续连线利用的节点，如果该节点在禁区就跳过。如果数量太多，则用简单聚类合并一下。

这样可以比过去那种对前沿/机场分别聚类再两两合并求中心，更容易保证节点对潜在多条连接有价值，并能兼顾到多对前沿-机场可能形成的庞大连线需求。
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

from utils.geometry import (
    is_point_in_circle,
    distance_point_to_point
)

# ========== 依赖KMeans用于必要时降到 max_nodes ==========
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)

class NodePlacementEnv(gym.Env):
    """
    Environment for placing intermediate nodes with an improved deterministic strategy.

    Node Types:
      0: frontline
      1: airport
      2: normal intermediate
      3: special intermediate (本版本同样不会区分特殊节点, 一律用2)

    当前设计：一旦reset()时就一次性放置所有中间节点，然后在step()里直接返回done=True，
    不执行多步操作。这样可以简化流程并与后续GraphConstructionEnv对接。
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

        # Zones
        self.adi_zones = cartesian_config['adi_zones']
        self.danger_zones = cartesian_config['danger_zones']

        # Action space: 仍然保留3D，但实际上不使用
        self.action_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, 0]),
            high=np.array([self.x_max, self.y_max, 1]),
            dtype=np.float32
        )

        # 保持与之前相同的 observation shape 以兼容性
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
                "[NodePlacementEnv] Using new midpoint-based approach for node placement. "
                "max_nodes=%d, min_distance=%.1f, max_distance=%.1f",
                self.max_nodes, self.min_distance, self.max_distance
            )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)

        # 这里进行一次性放置节点
        self._place_nodes_midpoint_based()

        # 我们已经放置完节点，不需要多余step
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
        直接done，不进行多步放置。
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
          3: special intermediate (统一=2)
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
        Flatten everything to match old observation shape.
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

    def _place_nodes_midpoint_based(self):
        """
        使用“前沿-机场对”中点的思路放置中间节点：
          1) 遍历所有 (frontline, airport) 对，求中点
          2) 如果该中点不在ADI内圈/机场禁飞圈/危险区内，且与已有节点距离>=min_distance，则保留
          3) 如果候选节点数量 > max_nodes，用KMeans聚成 max_nodes个
          4) 放入 self.intermediate_nodes，node_type=2
        """

        self.intermediate_nodes = []
        self.intermediate_node_types = []

        if len(self.frontline_points) == 0 or len(self.airport_points) == 0:
            if self.log_level >= 2:
                logger.info("[NodePlacementEnv] No frontline or airport points => no intermediate nodes.")
            return

        candidate_points = []
        # 1) 生成候选中点
        for f_pt in self.frontline_points:
            for a_pt in self.airport_points:
                mid_x = 0.5 * (f_pt[0] + a_pt[0])
                mid_y = 0.5 * (f_pt[1] + a_pt[1])
                if not self._valid_node(mid_x, mid_y, self.fixed_nodes, candidate_points):
                    continue
                # 先暂时收集，不立即加到candidate_points里，因为要统一判距离
                candidate_points.append([mid_x, mid_y])

        if len(candidate_points) == 0:
            # 没有可用的中点
            if self.log_level >= 2:
                logger.info("[NodePlacementEnv] No valid midpoints found.")
            return

        # 2) 去重(防止坐标几乎完全相同)
        unique_candidates = self._deduplicate_points(candidate_points)

        # 3) 如果数量超限，用 KMeans 压缩到 max_nodes
        if len(unique_candidates) > self.max_nodes:
            kmeans = KMeans(n_clusters=self.max_nodes, random_state=42)
            kmeans.fit(unique_candidates)
            final_centers = kmeans.cluster_centers_
        else:
            final_centers = np.array(unique_candidates)

        # 4) 将压缩或原始的节点纳入 intermediate_nodes
        #    但仍需对每个中心做一次 _valid_node 检查（可能因相互距离等因素）
        placed_count = 0
        placed_nodes = []

        for pt in final_centers:
            x, y = pt
            if not self._valid_node(x, y, self.fixed_nodes, placed_nodes):
                continue
            placed_nodes.append([x, y])
            placed_count += 1
            if placed_count >= self.max_nodes:
                break

        # 记录
        self.intermediate_nodes = placed_nodes
        self.intermediate_node_types = [2]*len(self.intermediate_nodes)

        if self.log_level >= 2:
            logger.info("[NodePlacementEnv] _place_nodes_midpoint_based => final placed %d nodes (limit=%d).",
                        len(self.intermediate_nodes), self.max_nodes)

    def _deduplicate_points(self, points: List[List[float]], eps: float = 1.0) -> List[List[float]]:
        """
        对候选点进行简单去重，如果两点距离<eps则视为同一个点，仅保留一个。
        这里 eps=1.0km，可自行调整。
        """
        unique_points = []
        for p in points:
            if not unique_points:
                unique_points.append(p)
                continue
            # check if p is near any
            too_close = False
            for q in unique_points:
                if distance_point_to_point(p, q) < eps:
                    too_close = True
                    break
            if not too_close:
                unique_points.append(p)
        return unique_points

    def _valid_node(self, x: float, y: float,
                    fixed_nodes: np.ndarray,
                    placed_nodes: List[List[float]]) -> bool:
        """
        Check if a candidate node is valid:
          1) 不在任何ADI内圈
          2) 不在任何机场20km禁飞区
          3) 不在任何危险区圆内
          4) 与所有已存在节点(含固定与中间)的距离 >= self.min_distance
          5) 距离至少与已有节点中任意一个 <= self.max_distance(否则没意义，离所有节点都太远)
        """

        # 1) 不在ADI内圈
        for z in self.adi_zones:
            if is_point_in_circle((x, y), ((z['center_x'], z['center_y']), z['radius'])):
                return False

        # 2) 不在机场20km禁飞区
        for a in self.cartesian_config['airports']:
            dist_airport = distance_point_to_point((x, y), (a['x'], a['y']))
            if dist_airport < 20.0:
                return False

        # 3) 不在危险区圆内
        for d in self.danger_zones:
            dist_danger = distance_point_to_point((x, y), (d['center_x'], d['center_y']))
            if dist_danger <= d['radius']:
                return False

        # 4) min_distance
        for node in fixed_nodes:
            if distance_point_to_point((x, y), tuple(node)) < self.min_distance:
                return False
        for node in placed_nodes:
            if distance_point_to_point((x, y), tuple(node)) < self.min_distance:
                return False

        # 5) 需要离“至少一个已存在节点”不超过 self.max_distance (如果都太远就没什么意义)
        #    - 也可以认为只要与所有点均>max_distance就无意义
        all_too_far = True
        for node in fixed_nodes:
            if distance_point_to_point((x, y), tuple(node)) <= self.max_distance:
                all_too_far = False
                break
        if all_too_far:
            for node in placed_nodes:
                if distance_point_to_point((x, y), tuple(node)) <= self.max_distance:
                    all_too_far = False
                    break
        if all_too_far:
            return False

        return True
