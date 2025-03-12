"""
Common model components for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

class MLP(nn.Module):
    """
    Multi-layer perceptron with optional layer normalization and dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU,
        layer_norm: bool = True,
        dropout: float = 0.0,
        output_activation: Optional[nn.Module] = None
    ):
        super(MLP, self).__init__()

        # Create layer dimensions
        all_dims = [input_dim] + hidden_dims + [output_dim]

        # Create layers
        layers = []

        for i in range(len(all_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))

            # Skip activation and normalization for the output layer if output_activation is None
            if i < len(all_dims) - 2 or output_activation is not None:
                # Layer normalization (before activation)
                if layer_norm:
                    layers.append(nn.LayerNorm(all_dims[i + 1]))

                # Activation
                if i < len(all_dims) - 2:
                    layers.append(activation())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                else:
                    if output_activation is not None:
                        layers.append(output_activation())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return self.model(x)

class GNNLayer(nn.Module):
    """
    Graph neural network layer with customized message passing.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        aggr: str = 'mean',
        activation: nn.Module = nn.ReLU
    ):
        super(GNNLayer, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.aggr = aggr

        # Message MLP (node features + edge features -> message)
        self.message_mlp = MLP(
            input_dim=node_dim * 2 + edge_dim,
            hidden_dims=[out_dim * 2],
            output_dim=out_dim,
            activation=activation
        )

        # Update MLP (node features + aggregated messages -> new node features)
        self.update_mlp = MLP(
            input_dim=node_dim + out_dim,
            hidden_dims=[out_dim * 2],
            output_dim=out_dim,
            activation=activation
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: Node feature tensor of shape [num_nodes, node_dim]
            edge_indices: Edge index tensor of shape [2, num_edges]
            edge_features: Edge feature tensor of shape [num_edges, edge_dim]

        Returns:
            Updated node features of shape [num_nodes, out_dim]
        """
        # Get source and target nodes
        src, dst = edge_indices

        # Get node features for source and target
        src_features = node_features[src]
        dst_features = node_features[dst]

        # Prepare message inputs
        if edge_features is not None:
            message_inputs = torch.cat([src_features, dst_features, edge_features], dim=1)
        else:
            message_inputs = torch.cat([src_features, dst_features], dim=1)

        # Compute messages
        messages = self.message_mlp(message_inputs)

        # Aggregate messages
        aggr_messages = self._aggregate_messages(messages, dst, node_features.size(0))

        # Update node features
        update_inputs = torch.cat([node_features, aggr_messages], dim=1)
        updated_node_features = self.update_mlp(update_inputs)

        return updated_node_features

    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        dst_indices: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate messages from neighbors.

        Args:
            messages: Message tensor of shape [num_edges, out_dim]
            dst_indices: Destination node indices of shape [num_edges]
            num_nodes: Total number of nodes

        Returns:
            Aggregated messages of shape [num_nodes, out_dim]
        """
        # Initialize aggregated messages
        aggr_messages = torch.zeros(num_nodes, self.out_dim, device=messages.device)

        # Aggregate messages for each destination node
        if self.aggr == 'mean':
            # Use scatter_add_ and then divide by count
            count = torch.zeros(num_nodes, 1, device=messages.device)
            count.scatter_add_(0, dst_indices.unsqueeze(1).expand(-1, 1), torch.ones_like(dst_indices.unsqueeze(1), dtype=torch.float))

            aggr_messages.scatter_add_(0, dst_indices.unsqueeze(1).expand(-1, messages.size(1)), messages)

            # Avoid division by zero
            count = torch.clamp(count, min=1.0)
            aggr_messages = aggr_messages / count

        elif self.aggr == 'sum':
            # Simply use scatter_add_
            aggr_messages.scatter_add_(0, dst_indices.unsqueeze(1).expand(-1, messages.size(1)), messages)

        elif self.aggr == 'max':
            # Use scatter_reduce_ with 'amax' reduction
            if hasattr(torch.Tensor, 'scatter_reduce_'):
                # PyTorch 1.12+
                aggr_messages.scatter_reduce_(0, dst_indices.unsqueeze(1).expand(-1, messages.size(1)), messages, reduce='amax')
            else:
                # Fallback for older PyTorch versions
                for i in range(num_nodes):
                    mask = dst_indices == i
                    if mask.any():
                        node_messages = messages[mask]
                        aggr_messages[i] = torch.max(node_messages, dim=0)[0]

        return aggr_messages

class GraphNeuralNetwork(nn.Module):
    """
    Multi-layer graph neural network.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        aggr: str = 'mean',
        activation: nn.Module = nn.ReLU,
        layer_norm: bool = True,
        dropout: float = 0.1
    ):
        super(GraphNeuralNetwork, self).__init__()

        self.node_embedding = MLP(
            input_dim=node_dim,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout
        )

        self.edge_embedding = None
        if edge_dim > 0:
            self.edge_embedding = MLP(
                input_dim=edge_dim,
                hidden_dims=[hidden_dim],
                output_dim=hidden_dim,
                activation=activation,
                layer_norm=layer_norm,
                dropout=dropout
            )

        # GNN layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = GNNLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim if edge_dim > 0 else 0,
                out_dim=hidden_dim,
                aggr=aggr,
                activation=activation
            )
            self.layers.append(layer)

        # Output layer
        self.output_layer = MLP(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim],
            output_dim=out_dim,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout
        )

        # Layer normalization for skip connections
        self.layer_norms = None
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            node_features: Node feature tensor of shape [num_nodes, node_dim]
            edge_indices: Edge index tensor of shape [2, num_edges]
            edge_features: Edge feature tensor of shape [num_edges, edge_dim]

        Returns:
            Tuple of (node_embeddings, graph_embedding)
        """
        # Embed nodes
        x = self.node_embedding(node_features)

        # Embed edges if applicable
        edge_attrs = None
        if edge_features is not None and self.edge_embedding is not None:
            edge_attrs = self.edge_embedding(edge_features)

        # Apply GNN layers with skip connections
        for i, layer in enumerate(self.layers):
            x_new = layer(x, edge_indices, edge_attrs)

            # Skip connection
            x = x + x_new

            # Layer normalization
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)

        # Generate node-level outputs
        node_embeddings = self.output_layer(x)

        # Global graph embedding (mean pooling)
        graph_embedding = torch.mean(node_embeddings, dim=0)

        return node_embeddings, graph_embedding

class DictObservationProcessor(nn.Module):
    """
    Process dictionary observations for graph neural networks.
    """

    def __init__(self, observation_space: Dict):
        super(DictObservationProcessor, self).__init__()

    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process dictionary observations.

        Args:
            observations: Dictionary of observation tensors

        Returns:
            Tuple of (node_features, edge_indices, edge_features)
        """
        node_features = observations['node_features']
        edge_indices = observations['edge_indices']
        edge_features = observations['edge_features']

        # Find the actual number of edges (non-padded)
        non_zero_mask = (edge_indices.sum(dim=0) != 0)
        actual_edge_indices = edge_indices[:, non_zero_mask]
        actual_edge_features = edge_features[non_zero_mask]

        return node_features, actual_edge_indices, actual_edge_features