"""
Node placement policy models for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Type
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from models.common import MLP

class NodePlacementExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for node placement observations.

    Processes the observation vector to extract meaningful features:
    - Fixed nodes (frontline and airport) features
    - Current intermediate nodes features
    - ADI zone parameters features
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(NodePlacementExtractor, self).__init__(observation_space, features_dim)

        # Input dimensions from observation space
        self.obs_dim = observation_space.shape[0]

        # Process the observation structure
        # We need to know how many fixed nodes, max intermediate nodes, and ADI zones we have
        # This information is embedded in the observation size

        # To parse the observation:
        # - Get number of fixed nodes and max intermediate nodes from the observation_space structure
        #   We know that each fixed node has 3 values (x, y, is_frontline)
        #   We know that each intermediate node has 3 values (x, y, is_outlier)
        #   We know that each ADI zone has 4 values (center_x, center_y, inner_radius, outer_radius)

        # We can reverse-engineer these values from the total observation size
        # Obs size = (num_fixed_nodes * 3) + (max_intermediate_nodes * 3) + (num_adi_zones * 4) + 1

        # Assuming 3 ADI zones (from the problem description)
        num_adi_zones = 3

        # Calculate remaining size after ADI zones and counter
        remaining_size = self.obs_dim - (num_adi_zones * 4) - 1

        # Assuming fixed and intermediate nodes both use 3 values per node
        # And using 18 fixed nodes (9 frontline + 9 airport) from the problem description
        num_fixed_nodes = 18
        max_intermediate_nodes = (remaining_size - num_fixed_nodes * 3) // 3

        self.num_fixed_nodes = num_fixed_nodes
        self.max_intermediate_nodes = max_intermediate_nodes
        self.num_adi_zones = num_adi_zones

        # Define feature extraction networks
        # Fixed nodes network
        self.fixed_nodes_net = MLP(
            input_dim=num_fixed_nodes * 3,
            hidden_dims=[128, 128],
            output_dim=128,
            activation=activation,
            layer_norm=True
        )

        # Intermediate nodes network
        self.intermediate_nodes_net = MLP(
            input_dim=max_intermediate_nodes * 3,
            hidden_dims=[128, 128],
            output_dim=128,
            activation=activation,
            layer_norm=True
        )

        # ADI zones network
        self.adi_zones_net = MLP(
            input_dim=num_adi_zones * 4,
            hidden_dims=[64],
            output_dim=64,
            activation=activation,
            layer_norm=True
        )

        # Combined network
        self.combined_net = MLP(
            input_dim=128 + 128 + 64 + 1,  # Fixed + Intermediate + ADI + Counter
            hidden_dims=[features_dim, features_dim],
            output_dim=features_dim,
            activation=activation,
            layer_norm=True
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Observation tensor of shape [batch_size, obs_dim]

        Returns:
            Extracted features of shape [batch_size, features_dim]
        """
        batch_size = observations.size(0)

        # Split the observation into components
        # Fixed nodes: [batch_size, num_fixed_nodes * 3]
        fixed_nodes_start = 0
        fixed_nodes_end = self.num_fixed_nodes * 3
        fixed_nodes_obs = observations[:, fixed_nodes_start:fixed_nodes_end]

        # Intermediate nodes: [batch_size, max_intermediate_nodes * 3]
        interm_start = fixed_nodes_end
        interm_end = interm_start + self.max_intermediate_nodes * 3
        interm_nodes_obs = observations[:, interm_start:interm_end]

        # ADI zones: [batch_size, num_adi_zones * 4]
        adi_start = interm_end
        adi_end = adi_start + self.num_adi_zones * 4
        adi_zones_obs = observations[:, adi_start:adi_end]

        # Counter: [batch_size, 1]
        counter = observations[:, -1].unsqueeze(-1)

        # Process each component
        fixed_nodes_feats = self.fixed_nodes_net(fixed_nodes_obs)
        interm_nodes_feats = self.intermediate_nodes_net(interm_nodes_obs)
        adi_zones_feats = self.adi_zones_net(adi_zones_obs)

        # Combine all features
        combined_feats = torch.cat([
            fixed_nodes_feats,
            interm_nodes_feats,
            adi_zones_feats,
            counter
        ], dim=1)

        # Final processing
        features = self.combined_net(combined_feats)

        return features

class NodePlacementPolicy(ActorCriticPolicy):
    """
    Actor-critic policy for node placement.

    Outputs:
    - Actor: (x, y, node_type) continuous values
    - Critic: Value function
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        lr_schedule: callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs
    ):
        # Use custom feature extractor
        features_extractor_class = NodePlacementExtractor
        features_extractor_kwargs = {'features_dim': 256, 'activation': activation_fn}

        # Default architecture
        if net_arch is None:
            net_arch = [dict(pi=[128, 64], vf=[128, 64])]

        super(NodePlacementPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )

        # Modify the action net if needed
        # For example, to ensure the node type is closer to 0 or 1
        # We can apply a custom activation function to the last dimension

        # Get the original output layer
        original_action_net = self.action_net

        # Create a new module that adds custom post-processing
        class CustomActionNet(nn.Module):
            def __init__(self, base_net):
                super(CustomActionNet, self).__init__()
                self.base_net = base_net

            def forward(self, features):
                # Get raw outputs
                raw_actions = self.base_net(features)

                # Apply custom activation to the node type (third dimension)
                batch_size = raw_actions.size(0)

                # Split into coordinates and node type
                coords = raw_actions[:, :2]
                node_type = raw_actions[:, 2].unsqueeze(-1)

                # Apply sigmoid to node type to get it closer to 0 or 1
                node_type = torch.sigmoid(node_type * 5)  # Steeper sigmoid

                # Recombine
                processed_actions = torch.cat([coords, node_type], dim=1)

                return processed_actions

        # Replace the action net
        self.action_net = CustomActionNet(original_action_net)