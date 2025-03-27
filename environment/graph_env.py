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
    extract_features_for_gnn,
    find_shortest_paths  # ← 新增 import，用于打印每对的连通情况
)

class GraphConstructionEnv(gym.Env):
    """
    Environment for constructing a graph/network by adding edges between nodes.

    修正要点：
    1. 我们在 __init__ 中一次性固定离散动作空间的最大值（为所有可能边的上限），
       而不再在每个 step 或 reset 中动态修改 action_space。
    2. 当 done = True 后，打印每个前沿点-机场配对的连通性，为了方便直观查看最终是否全部连通。
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

        # 当前已添加的边
        self.edges = []
        self.valid_potential_edges = []

        # ----- Fix：一次性确定 action_space 的最大可能取值，不再动态变化 -----
        # 令上限为 n*(n-1)//2（所有无向边总数量），确保至少为1
        self.max_possible_edges = self.num_nodes * (self.num_nodes - 1) // 2
        if self.max_possible_edges < 1:
            self.max_possible_edges = 1
        self.action_space = spaces.Discrete(self.max_possible_edges)

        # ----- 构建 observation_space -----
        unique_node_types = max(node_types) + 1
        node_feature_dim = 2 + unique_node_types + len(self.adi_zones)
        edge_feature_dim = 1 + len(self.adi_zones) + 1  # length + crossesADI + danger

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

        # 各种奖励参数
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
        action 是 [0, self.max_possible_edges-1] 区间内的整数。
        如果 action >= len(self.valid_potential_edges)，视为非法动作。
        """
        # 1) 如果 action 超出当前可用边数量，则视为非法动作
        if action >= len(self.valid_potential_edges):
            obs = self._get_observation()
            return obs, self.reward_edge_invalid, False, False, {'reason': 'action index out of range'}

        # 2) 取到要添加的边
        edge = self.valid_potential_edges[action]
        i, j = edge

        start_point = tuple(self.nodes[i])
        end_point = tuple(self.nodes[j])

        # 3) 再次校验该边是否真的有效（通常都有效）
        is_valid, reason, _ = is_line_segment_valid(
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
            # 理论上不会进入这里，但以防万一
            obs = self._get_observation()
            return obs, self.reward_edge_invalid, False, False, {'reason': reason}

        # 4) 添加边
        self.edges.append(edge)

        # 5) 更新可用边列表
        self._update_valid_potential_edges()

        # 6) 计算该动作的奖励
        reward = self._calculate_reward(edge)

        # 7) 检查连通性
        connectivity_score, num_connected_pairs, _ = calculate_connectivity_score(
            self.nodes,
            self.edges,
            self.frontline_indices,
            self.airport_indices,
            node_types=self.node_types
        )
        total_pairs = self.num_frontline * self.num_airports

        # 判断是否结束
        done = False
        if num_connected_pairs == total_pairs:
            # 全部连通
            reward += self.reward_completion
            done = True
        elif len(self.edges) >= self.max_edges:
            # 达到边数上限
            done = True
        elif len(self.valid_potential_edges) == 0:
            # 没有可添加的边了
            done = True

        obs = self._get_observation()
        info = {
            'num_connected_pairs': num_connected_pairs,
            'total_pairs': total_pairs,
            'connectivity_score': connectivity_score,
            'num_edges': len(self.edges)
        }

        # 如果 done=True，打印每个前沿点-机场对是否连通，方便最直观判断
        if done:
            all_paths, success_flags = find_shortest_paths(
                self.nodes,
                self.edges,
                self.frontline_indices,
                self.airport_indices,
                node_types=self.node_types
            )
            print("[GraphConstructionEnv] DONE! Pairwise connectivity details:")
            idx = 0
            for f in self.frontline_indices:
                for a in self.airport_indices:
                    status = "CONNECTED" if success_flags[idx] else "NOT CONNECTED"
                    print(f"  Pair(frontline={f}, airport={a}): {status}")
                    idx += 1
            if num_connected_pairs == total_pairs:
                print("=> All pairs are connected! ✅")
            else:
                print(f"=> Not all pairs connected. (connected {num_connected_pairs}/{total_pairs}) ❌")

        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_network(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        """
        返回 (nodes, node_types, edges)
        """
        return self.nodes, self.node_types, self.edges

    def get_network_evaluation(self) -> Dict[str, Any]:
        """
        使用 evaluate_network 函数对当前网络进行综合评估。
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
        生成 GNN 风格的 observation.
        """
        node_features, edge_indices, edge_features = extract_features_for_gnn(
            self.nodes,
            self.node_types,
            self.edges,
            self.adi_zones,
            self.danger_zones
        )

        num_edges = edge_indices.shape[1] if edge_indices.ndim > 1 else 0

        # 将 edge_indices, edge_features 填充至 self.max_edges 大小
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

        # 构建 valid_mask
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
        重新计算哪些边还可添加。不会修改 action_space，只更新 self.valid_potential_edges。
        """
        valid_edges = []
        existing = set(self.edges) | set((b, a) for (a, b) in self.edges)

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if (i, j) in existing:
                    continue

                # 前沿点不能直接连前沿点，机场不能直接连机场
                if (i in self.frontline_indices and j in self.frontline_indices) or \
                   (i in self.airport_indices and j in self.airport_indices):
                    continue

                start_point = tuple(self.nodes[i])
                end_point = tuple(self.nodes[j])

                # 检查机场 20km 禁飞区（若端点不是该机场本身）
                skip_edge = False
                for airport_idx in self.airport_indices:
                    if i == airport_idx or j == airport_idx:
                        continue
                    airport_point = tuple(self.nodes[airport_idx])
                    if does_line_cross_circle((start_point, end_point), (airport_point, 20.0)):
                        skip_edge = True
                        break
                if skip_edge:
                    continue

                # 几何合法性检测
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
        给新添加的边计算奖励。
        """
        i, j = edge
        reward = self.reward_edge_added

        start_point = tuple(self.nodes[i])
        end_point = tuple(self.nodes[j])
        line = (start_point, end_point)

        # 若穿过外圈 ADI，给予额外奖励
        crosses_adi_outer = False
        for zone in self.adi_zones:
            _, crosses_outer = does_line_cross_adi_zone(line, zone)
            if crosses_outer:
                crosses_adi_outer = True
                break
        if crosses_adi_outer:
            reward += self.reward_adi_traversal

        # 危险区惩罚
        danger_penalty = get_danger_zone_penalty(line, self.danger_zones)
        reward += self.reward_danger_penalty_factor * danger_penalty

        # 连通性提升奖励
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
