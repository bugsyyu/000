"""
Graph construction policy models for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Type
import gym
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from models.common import MLP, GraphNeuralNetwork, DictObservationProcessor

class GraphConstructionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for graph construction observations.

    Processes the dictionary observation and applies a GNN to extract node and graph features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 3,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(GraphConstructionExtractor, self).__init__(observation_space, features_dim)

        # Observation processor
        self.observation_processor = DictObservationProcessor(observation_space)

        # Get dimensions from observation space
        node_feature_dim = observation_space['node_features'].shape[1]
        edge_feature_dim = observation_space['edge_features'].shape[1]

        # Graph neural network
        self.gnn = GraphNeuralNetwork(
            node_dim=node_feature_dim,
            edge_dim=edge_feature_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            aggr='mean',
            activation=activation,
            layer_norm=True,
            dropout=0.1
        )

        # MLP to combine node and graph features
        self.combined_net = MLP(
            input_dim=gnn_hidden_dim,
            hidden_dims=[features_dim, features_dim],
            output_dim=features_dim,
            activation=activation,
            layer_norm=True
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Dictionary of observation tensors

        Returns:
            Extracted features of shape [batch_size, features_dim]
        """
        # Process observations
        node_features, edge_indices, edge_features = self.observation_processor(observations)

        # Apply GNN
        node_embeddings, graph_embedding = self.gnn(node_features, edge_indices, edge_features)

        # Use graph embedding as the features
        features = self.combined_net(graph_embedding)

        # Ensure we have a batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)

        return features

class GraphConstructionActorHead(nn.Module):
    """
    Actor head for graph construction policy.

    Takes node embeddings and outputs edge selection probabilities.
    """

    def __init__(
        self,
        node_embedding_dim: int,
        hidden_dim: int = 128,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(GraphConstructionActorHead, self).__init__()

        # Edge scoring MLP
        self.edge_scorer = MLP(
            input_dim=node_embedding_dim * 2,  # Concatenated node features
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=1,
            activation=activation,
            layer_norm=True
        )

    def forward(
            self,
            node_embeddings: torch.Tensor,
            valid_edges: List[Tuple[int, int]],
            valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the actor head.

        Args:
            node_embeddings: Node embedding tensor of shape [num_nodes, embedding_dim]
            valid_edges: List of valid edge indices
            valid_mask: Binary mask indicating valid actions

        Returns:
            Edge selection logits of shape [num_valid_edges]
        """
        # 修改为使用完整掩码而不是边的列表
        num_nodes = node_embeddings.shape[0]
        edge_scores = torch.zeros(num_nodes * num_nodes, device=node_embeddings.device)

        # 仅为有效操作计算分数
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # 避免自环
                    idx = i * num_nodes + j
                    if valid_mask[idx]:
                        # 拼接节点嵌入
                        edge_emb = torch.cat([node_embeddings[i], node_embeddings[j]], dim=0)
                        # 计算单个边的分数
                        score = self.edge_scorer(edge_emb.unsqueeze(0)).squeeze(-1)
                        edge_scores[idx] = score

        # 屏蔽无效操作
        edge_scores = edge_scores * valid_mask

        return edge_scores

class GraphConstructionPolicy(ActorCriticPolicy):
    """
    Actor-critic policy for graph construction.

    Outputs:
    - Actor: Edge selection probabilities
    - Critic: Value function
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        lr_schedule: callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = GraphConstructionExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ):
        # Default architecture
        if net_arch is None:
            net_arch = [dict(pi=[128, 64], vf=[128, 64])]

        # Default feature extractor arguments
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 256,
                'gnn_hidden_dim': 128,
                'gnn_layers': 3,
                'activation': activation_fn
            }

        super(GraphConstructionPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs
        )

        # Custom actor network
        # Extract GNN for direct access to node embeddings
        self.gnn = self.features_extractor.gnn

        # Actor head
        self.actor_head = GraphConstructionActorHead(
            node_embedding_dim=features_extractor_kwargs['gnn_hidden_dim'],
            hidden_dim=128,
            activation=activation_fn
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde: Optional[torch.Tensor] = None) -> torch.distributions.Distribution:
        """
        Get action distribution from latent features.

        Args:
            latent_pi: Latent features from the policy network
            latent_sde: Latent features for the exploration noise (not used)

        Returns:
            Action distribution
        """
        # Ensure we have a batch dimension
        if latent_pi.dim() == 1:
            latent_pi = latent_pi.unsqueeze(0)

        # Get action logits from the action_net
        action_logits = self.action_net(latent_pi)

        # Create categorical distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)

        return action_dist

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy.

        Args:
            obs: Dictionary of observation tensors
            deterministic: Whether to return deterministic actions

        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract node features, edge indices, and edge features
        node_features, edge_indices, edge_features = self.features_extractor.observation_processor(obs)

        # Apply GNN to get node embeddings
        node_embeddings, graph_embedding = self.gnn(node_features, edge_indices, edge_features)

        # Extract valid edges and mask from observation
        valid_mask = obs['valid_mask']
        valid_edges = []

        # Convert valid mask to list of edge indices
        for i in range(len(valid_mask)):
            if i < len(valid_mask) and valid_mask[i]:
                valid_edges.append(i)

        # Get action logits from actor head
        # Note: We'll use a simplified approach here using the action_net directly

        # Extract features
        features = self.features_extractor(obs)

        # Get values
        values = self.value_net(features)

        # Get actions using parent class method
        action_distribution = self._get_action_dist_from_latent(features)

        if deterministic:
            actions = action_distribution.probs.argmax(dim=1)
        else:
            actions = action_distribution.sample()

        log_probs = action_distribution.log_prob(actions)

        return actions, values, log_probs