"""
Graph construction environment for reinforcement learning.

本版本在原有基础上，借鉴了“阈值 + 星型汇聚”的思路，对可选连线做了进一步筛选，
并在奖励函数中额外加入了星型汇聚奖励。 同时，新增“不能与已有边交叉”的硬性约束，
在 is_line_segment_valid(...) 函数中已实现对任何交叉情况的禁止。

只要新线段和已有线段在非公共端点处发生几何相交，即视为不合法。

【更新说明】为避免在同一回合中无限次尝试无效动作，导致回合不会结束的问题：
  1) 新增了 max_steps_per_episode (默认 500) 参数，用于限制每回合最大步数；
  2) 新增了 max_invalid_actions_per_episode (默认 50) 参数，如果连续/累计无效动作次数超过此值，则强制结束回合。
  3) 当以上两种条件任意一种触发时，done=True，并重置回合。

这样可以防止在某些极端情况下因智能体不断选择无效动作而导致回合一直不结束。
"""

import logging
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
    find_shortest_paths
)

logger = logging.getLogger(__name__)

def _distance_point_to_line(pt: Tuple[float, float],
                            f: Tuple[float, float],
                            a: Tuple[float, float]) -> float:
    """
    计算点 pt 到直线(f->a)的最短距离。
    """
    fx, fy = f
    ax, ay = a
    px, py = pt

    fax = ax - fx
    fay = ay - fy

    fpx = px - fx
    fpy = py - fy

    cross_val = abs(fpx * fay - fpy * fax)
    denom = np.sqrt(fax**2 + fay**2) + 1e-9

    return cross_val / denom

class GraphConstructionEnv(gym.Env):
    """
    Environment for constructing a graph/network by adding edges between nodes.

    Features:
      - Star-based feasibility: For each (frontline, airport) pair, only certain "close-to-line" nodes are considered.
      - Keeps original ADI inner circle constraint, outer circle angle constraint, airport no-fly zone, etc.
      - Additional constraint: newly added edges cannot intersect any existing edge (except at a shared endpoint).
      - Additional star-hub reward: connecting to "hot" (popular) nodes yields extra reward.
      - To prevent infinite episodes:
          * max_steps_per_episode (default=500)
          * max_invalid_actions_per_episode (default=50)

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
            render_mode: str = None,
            log_level: int = 1,
            star_alpha: float = 1.2,  # 用于计算距离阈值
            max_steps_per_episode: int = 1000,
            max_invalid_actions_per_episode: int = 250
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

        self.log_level = log_level
        self.num_nodes = len(nodes)
        self.num_frontline = len(frontline_indices)
        self.num_airports = len(airport_indices)

        # 额外限制：每回合最大步数 & 最大无效动作次数
        self.max_steps_per_episode = max_steps_per_episode
        self.max_invalid_actions_per_episode = max_invalid_actions_per_episode

        # 回合步数 & 无效动作计数
        self.current_step_count = 0
        self.invalid_action_count = 0

        # Record edges
        self.edges = []
        self.valid_potential_edges = []

        # Step1: 预计算星型可行性
        self.star_alpha = star_alpha
        self.node_feasible_pairs = [set() for _ in range(self.num_nodes)]
        self._precompute_feasible_sets_for_star()

        # node_pair_count[i] = 节点i可服务多少(F,A)对 => 用于星型奖励
        self.node_pair_count = [len(self.node_feasible_pairs[i]) for i in range(self.num_nodes)]

        # Define the action space (Discrete) by the maximum possible edges
        self.max_possible_edges = self.num_nodes * (self.num_nodes - 1) // 2
        if self.max_possible_edges < 1:
            self.max_possible_edges = 1
        self.action_space = spaces.Discrete(self.max_possible_edges)

        # Construct observation_space
        unique_node_types = max(node_types) + 1 if len(node_types) > 0 else 1
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

        # reward params
        self.reward_edge_added = 1.0
        self.reward_edge_invalid = -1.0
        self.reward_connectivity_improvement = 5.0
        self.reward_adi_traversal = 2.0
        self.reward_danger_penalty_factor = -1.0
        self.reward_completion = 20.0
        # star-like hub bonus
        self.star_node_bonus_coef = 0.2

        # [MODIFICATION] New synergy coefficient: how much bonus we get if a newly added edge serves multiple pairs
        self.synergy_coefficient = 0.3

        # [MODIFICATION] Factor controlling reward for improving (reducing) average path length
        # if path length after is less than before => negative difference => multiplied by negative => positive reward
        self.reward_path_length_factor = -0.02

        if self.log_level >= 1:
            logger.info("[GraphConstructionEnv] Initialized with %d nodes, max_edges=%d",
                        self.num_nodes, self.max_edges)

    def _precompute_feasible_sets_for_star(self):
        """
        For each frontline-airport pair, find "close-to-line" nodes.
        """
        frontline_coords = [(self.nodes[f][0], self.nodes[f][1]) for f in self.frontline_indices]
        airport_coords = [(self.nodes[a][0], self.nodes[a][1]) for a in self.airport_indices]

        for f_i, f_idx in enumerate(self.frontline_indices):
            f_pt = frontline_coords[f_i]
            for a_j, a_idx in enumerate(self.airport_indices):
                a_pt = airport_coords[a_j]

                # dist from each node to line (f_pt->a_pt)
                dvals = []
                for n_i in range(self.num_nodes):
                    pt = (self.nodes[n_i][0], self.nodes[n_i][1])
                    dist_line = _distance_point_to_line(pt, f_pt, a_pt)
                    dvals.append(dist_line)

                avg_d = float(np.mean(dvals))
                threshold = self.star_alpha * avg_d

                # pick feasible nodes
                for n_i, dd in enumerate(dvals):
                    if dd <= threshold:
                        self.node_feasible_pairs[n_i].add((f_idx, a_idx))

                # also mark that F,A themselves are feasible for that pair
                self.node_feasible_pairs[f_idx].add((f_idx, a_idx))
                self.node_feasible_pairs[a_idx].add((f_idx, a_idx))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.edges = []
        self._update_valid_potential_edges()

        # 重置计数器
        self.current_step_count = 0
        self.invalid_action_count = 0

        obs = self._get_observation()
        if self.log_level >= 2:
            logger.info("[GraphConstructionEnv] Reset. Potential edges count=%d",
                        len(self.valid_potential_edges))
        return obs, {}

    def step(self, action: int):
        # 先检查是否达到最大步数限制
        self.current_step_count += 1
        if self.current_step_count > self.max_steps_per_episode:
            # 超过回合最大步数 => done
            obs = self._get_observation()
            info = {
                'reason': 'max_steps_per_episode reached',
                'num_edges': len(self.edges)
            }
            return obs, 0.0, True, False, info

        # 判断action是否越界
        if action >= len(self.valid_potential_edges):
            # 无效动作
            obs = self._get_observation()
            self.invalid_action_count += 1
            if self.log_level >= 3:
                logger.debug("[GraphConstructionEnv] Action out of range => invalid edge action=%d", action)

            # 如果无效动作过多，也强制结束
            if self.invalid_action_count >= self.max_invalid_actions_per_episode:
                info = {
                    'reason': 'exceed max invalid actions',
                    'num_edges': len(self.edges)
                }
                return obs, self.reward_edge_invalid, True, False, info
            else:
                return obs, self.reward_edge_invalid, False, False, {'reason': 'action index out of range'}

        # pick the edge
        edge = self.valid_potential_edges[action]
        i, j = edge

        start_pt = tuple(self.nodes[i])
        end_pt = tuple(self.nodes[j])

        # double-check validity
        is_valid, reason, _ = is_line_segment_valid(
            start_pt,
            end_pt,
            self.adi_zones,
            self.edges,
            self.nodes,
            self.node_types,
            self.airport_indices,
            self.max_angle_deg
        )
        if not is_valid:
            # 动作不合法
            obs = self._get_observation()
            if self.log_level >= 3:
                logger.debug("[GraphConstructionEnv] Chosen edge invalid => %s", reason)

            self.invalid_action_count += 1
            if self.invalid_action_count >= self.max_invalid_actions_per_episode:
                info = {
                    'reason': 'exceed max invalid actions',
                    'num_edges': len(self.edges)
                }
                return obs, self.reward_edge_invalid, True, False, info
            else:
                return obs, self.reward_edge_invalid, False, False, {'reason': reason}

        # 动作合法 => 执行连线
        self.edges.append(edge)

        self._update_valid_potential_edges()

        reward = self._calculate_reward(edge)

        # check connectivity
        connectivity_score, num_connected_pairs, _ = calculate_connectivity_score(
            self.nodes,
            self.edges,
            self.frontline_indices,
            self.airport_indices,
            node_types=self.node_types
        )
        total_pairs = self.num_frontline * self.num_airports

        done = False
        if num_connected_pairs == total_pairs:
            # 全部连通
            reward += self.reward_completion
            done = True
        elif len(self.edges) >= self.max_edges:
            # 达到可连线数量上限
            done = True
        elif len(self.valid_potential_edges) == 0:
            # 没有可选连线了
            done = True

        # 如果已经到达回合步数上限，也结束
        if not done and self.current_step_count >= self.max_steps_per_episode:
            done = True

        obs = self._get_observation()
        info = {
            'num_connected_pairs': num_connected_pairs,
            'total_pairs': total_pairs,
            'connectivity_score': connectivity_score,
            'num_edges': len(self.edges)
        }

        # 如果 done=True，需要输出连接情况
        if done:
            all_paths, success_flags = find_shortest_paths(
                self.nodes,
                self.edges,
                self.frontline_indices,
                self.airport_indices,
                node_types=self.node_types
            )
            if self.log_level >= 2:
                logger.info("[GraphConstructionEnv] DONE! Pairwise connectivity details:")
            idx = 0
            for f_ in self.frontline_indices:
                for a_ in self.airport_indices:
                    status_ = "CONNECTED" if success_flags[idx] else "NOT CONNECTED"
                    if self.log_level >= 2:
                        logger.info("   Pair(frontline=%d, airport=%d): %s", f_, a_, status_)
                    idx += 1
            if num_connected_pairs == total_pairs and self.log_level >= 2:
                logger.info("=> All pairs are connected! [OK]")
            elif self.log_level >= 2:
                logger.info("=> Not all pairs connected. (connected %d/%d) [X]",
                            num_connected_pairs, total_pairs)

        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_network(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        """
        Return (nodes, node_types, edges).
        """
        return self.nodes, self.node_types, self.edges

    def get_network_evaluation(self) -> Dict[str, Any]:
        """
        Evaluate the final network with respect to angles, ADI crossing, danger zone penalty, etc.
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
        node_features, edge_indices, edge_features = extract_features_for_gnn(
            self.nodes,
            self.node_types,
            self.edges,
            self.adi_zones,
            self.danger_zones
        )
        num_edges = edge_indices.shape[1] if edge_indices.ndim > 1 else 0

        # pad edges
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

        valid_mask = np.zeros(self.num_nodes * self.num_nodes, dtype=bool)
        for (i, j) in self.valid_potential_edges:
            idx = i * self.num_nodes + j
            valid_mask[idx] = True

        return {
            'node_features': node_features.astype(np.float32),
            'edge_indices': padded_edge_indices,
            'edge_features': padded_edge_features.astype(np.float32),
            'valid_mask': valid_mask
        }

    def _update_valid_potential_edges(self):
        valid_edges = []
        existing = set(self.edges) | set((b, a) for (a, b) in self.edges)

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if (i, j) in existing:
                    continue

                # frontline-frontline or airport-airport => not allowed
                if (i in self.frontline_indices and j in self.frontline_indices) or \
                   (i in self.airport_indices and j in self.airport_indices):
                    continue

                # star feasibility check
                feasible_intersect = self.node_feasible_pairs[i].intersection(self.node_feasible_pairs[j])
                if len(feasible_intersect) == 0:
                    continue

                valid_edges.append((i, j))

        self.valid_potential_edges = valid_edges

    def _calculate_reward(self, edge: Tuple[int, int]) -> float:
        """
        Compute the reward for adding a particular edge. This includes:
          - A base reward for adding an edge
          - A bonus if the edge crosses ADI outer zone (reward_adi_traversal)
          - A penalty if crossing danger zone(s)
          - Connectivity improvement (based on how many new pairs got connected)
          - Star/hub bonus for connecting to popular nodes
          - [MODIFICATION] Synergy bonus for how many frontline-airport pairs use this new edge
          - [MODIFICATION] Improvement in average path length for connected pairs (before vs. after).
        """
        i, j = edge
        reward = self.reward_edge_added
        start_point = tuple(self.nodes[i])
        end_point = tuple(self.nodes[j])
        line = (start_point, end_point)

        # ADI outer circle crossing => small reward
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

        # connectivity improvement
        edges_before = [e for e in self.edges if e != edge]
        conn_before, num_conn_before, paths_before = calculate_connectivity_score(
            self.nodes,
            edges_before,
            self.frontline_indices,
            self.airport_indices,
            node_types=self.node_types
        )
        conn_after, num_conn_after, paths_after = calculate_connectivity_score(
            self.nodes,
            self.edges,
            self.frontline_indices,
            self.airport_indices,
            node_types=self.node_types
        )
        improvement = num_conn_after - num_conn_before
        reward += self.reward_connectivity_improvement * improvement

        # star/hub bonus: extra for connecting to popular nodes
        hotness_i = self.node_pair_count[i]
        hotness_j = self.node_pair_count[j]
        star_bonus = self.star_node_bonus_coef * float(hotness_i + hotness_j) / 10.0
        reward += star_bonus

        # [MODIFICATION]: synergy bonus => how many pairs actually use this new edge in their shortest paths?
        synergy_bonus = 0.0
        if improvement > 0:
            # We'll check how many of the newly connected pairs (or all connected pairs) pass through this edge
            synergy_usage_count = self._count_pairs_using_edge(edge, paths_after)
            synergy_bonus = synergy_usage_count * self.synergy_coefficient
        reward += synergy_bonus

        # [MODIFICATION]: difference in average path length for connected pairs
        avg_len_before = self._compute_avg_path_length(paths_before)
        avg_len_after = self._compute_avg_path_length(paths_after)
        path_length_diff = (avg_len_after - avg_len_before)
        reward += self.reward_path_length_factor * path_length_diff

        return reward

    # [MODIFICATION] Helper function to see how many pairs used the newly added edge in their shortest paths
    def _count_pairs_using_edge(
        self,
        new_edge: Tuple[int, int],
        all_paths: List[List[int]]
    ) -> int:
        """
        Count how many (frontline-airport) pairs' shortest path includes the new_edge.
        all_paths is in the same order as the nested loops (frontline_indices x airport_indices).
        """
        # We consider (i, j) or (j, i) as the same
        a, b = new_edge
        usage_count = 0
        for path in all_paths:
            if not path or len(path) < 2:
                continue
            # Walk the path in segments
            for k in range(len(path) - 1):
                if (path[k] == a and path[k + 1] == b) or (path[k] == b and path[k + 1] == a):
                    usage_count += 1
                    break  # no need to check further segments for this path
        return usage_count

    # [MODIFICATION] Helper to compute average path length for all successfully connected pairs
    def _compute_avg_path_length(self, paths: List[List[int]]) -> float:
        """
        Compute the average path length across all provided shortest paths.
        Only considers those that are non-empty. If none are non-empty, returns 0.
        """
        total_length = 0.0
        count = 0
        for path in paths:
            if len(path) > 1:
                path_dist = 0.0
                for k in range(len(path) - 1):
                    p1 = self.nodes[path[k]]
                    p2 = self.nodes[path[k+1]]
                    path_dist += distance_point_to_point(p1, p2)
                total_length += path_dist
                count += 1
        if count == 0:
            return 0.0
        return total_length / float(count)
