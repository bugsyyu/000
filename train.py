"""
Training script for the airspace network planning system.
"""

import os
import argparse
import numpy as np
import torch
import random
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config.latlon_config import latlon_config
from utils.coordinate_transform import transform_config_to_cartesian
from environment.node_env import NodePlacementEnv
from environment.graph_env import GraphConstructionEnv
from utils.visualization import plot_airspace_network, plot_training_progress, plot_paths_between_points

class TrainingProgressCallback(BaseCallback):
    """
    Callback for tracking and saving training progress.
    """

    def __init__(
        self,
        save_path: str,
        save_freq: int = 10,
        verbose: int = 1
    ):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """
        Called at each step of the training.
        """
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Extract success info if available
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                if 'success' in self.locals['infos'][0]:
                    self.episode_successes.append(self.locals['infos'][0]['success'])
                elif 'num_connected_pairs' in self.locals['infos'][0] and 'total_pairs' in self.locals['infos'][0]:
                    success = self.locals['infos'][0]['num_connected_pairs'] == self.locals['infos'][0]['total_pairs']
                    self.episode_successes.append(success)
                else:
                    self.episode_successes.append(False)
            else:
                self.episode_successes.append(False)

            self.current_episode_reward = 0
            self.current_episode_length = 0

            # Save progress
            if len(self.episode_rewards) % self.save_freq == 0:
                self._save_progress()

        return True

    def _save_progress(self):
        """
        Save training progress to file.
        """
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

        # Plot progress
        fig, _ = plot_training_progress(
            self.episode_rewards,
            self.episode_lengths,
            self.episode_successes
        )

        # Save plot
        plt.savefig(os.path.join(self.save_path, 'training_progress.png'))
        plt.close(fig)

        # Save data
        np.savez(
            os.path.join(self.save_path, 'training_data.npz'),
            rewards=np.array(self.episode_rewards),
            lengths=np.array(self.episode_lengths),
            successes=np.array(self.episode_successes)
        )

def train_node_placement(
    cartesian_config: Dict[str, Any],
    output_dir: str,
    n_envs: int = 4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    device: str = 'auto',
    seed: int = 42,
    max_nodes: int = 30,
    total_timesteps: int = 1000000
) -> Tuple[PPO, Tuple[np.ndarray, List[int]]]:
    """
    Train the node placement model.

    Args:
        cartesian_config: Configuration with Cartesian coordinates
        output_dir: Directory to save outputs
        n_envs: Number of parallel environments
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size for SGD
        n_epochs: Number of epochs for optimization
        learning_rate: Learning rate
        gamma: Discount factor
        device: Device to run the model on
        seed: Random seed
        max_nodes: Maximum number of intermediate nodes
        total_timesteps: Total number of timesteps to train for

    Returns:
        Tuple of (trained model, tuple of (nodes, node_types))
    """
    # Set random seed
    set_random_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'node_placement'), exist_ok=True)

    # Create environment list
    env_fns = []
    for _ in range(n_envs):
        env_fns.append(lambda: NodePlacementEnv(
            cartesian_config=cartesian_config,
            max_nodes=max_nodes,
            min_distance=10.0,
            max_distance=150.0
        ))

    # Create vectorized environments using DummyVecEnv instead of SubprocVecEnv
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps * 10,
        save_path=os.path.join(output_dir, 'node_placement', 'checkpoints'),
        name_prefix='node_placement_model'
    )

    progress_callback = TrainingProgressCallback(
        save_path=os.path.join(output_dir, 'node_placement'),
        save_freq=10
    )

    # Create and train model
    model = PPO(
        'MlpPolicy',  # This is correct for Box observation space
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        device=device,
        seed=seed
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback]
    )

    # Save final model
    model.save(os.path.join(output_dir, 'node_placement', 'final_model'))

    # ===========================================
    # Run a modified evaluation process that avoids getting stuck
    # ===========================================

    print("Running evaluation with improved logic to avoid getting stuck...")

    # Create evaluation environment
    eval_env = NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=max_nodes,
        min_distance=10.0,
        max_distance=150.0
    )

    # Get initial observation
    reset_result = eval_env.reset(seed=seed+100)  # Use a different seed
    obs = reset_result[0]  # Extract just the observation

    # Prepare for evaluation
    max_steps = 50  # Limit to prevent infinite loops
    step_count = 0
    done = False
    repeated_action_count = 0
    last_action = None
    placed_nodes = 0
    max_repeat_tries = 5  # Max number of times to try the same action

    # Improved evaluation loop
    while not done and step_count < max_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=False)  # Use stochastic actions to avoid getting stuck

        # Check if we're repeating the same action
        if last_action is not None and np.allclose(action, last_action):
            repeated_action_count += 1

            # If we've tried the same action too many times, add randomness
            if repeated_action_count >= max_repeat_tries:
                print(f"Action {action} repeated {repeated_action_count} times, adding randomness...")

                # Generate a random action that's more likely to be valid
                # First, get bounds from environment
                x_range = eval_env.x_max - eval_env.x_min
                y_range = eval_env.y_max - eval_env.y_min

                # Get existing nodes to avoid placing too close
                fixed_nodes = eval_env.fixed_nodes

                if len(fixed_nodes) > 0:
                    # Pick a random fixed node as reference
                    random_node = fixed_nodes[random.randint(0, len(fixed_nodes) - 1)]

                    # Add a random offset (30-100km)
                    offset_distance = random.uniform(30.0, 100.0)
                    angle = random.uniform(0, 2 * np.pi)

                    # Calculate new position
                    x = random_node[0] + offset_distance * np.cos(angle)
                    y = random_node[1] + offset_distance * np.sin(angle)

                    # Ensure within bounds
                    x = max(eval_env.x_min, min(eval_env.x_max, x))
                    y = max(eval_env.y_min, min(eval_env.y_max, y))

                    # Random node type (0 for common, 1 for outlier)
                    node_type = random.randint(0, 1)

                    action = np.array([x, y, node_type])
                else:
                    # Fallback to random but more distant
                    action = np.array([
                        eval_env.x_min + x_range * random.random(),
                        eval_env.y_min + y_range * random.random(),
                        random.randint(0, 1)
                    ])

                repeated_action_count = 0  # Reset counter
        else:
            repeated_action_count = 0
            last_action = action.copy()

        # Take step in environment
        step_result = eval_env.step(action)
        obs, reward, done, _, info = step_result

        step_count += 1

        # If this was a valid placement, record it
        if 'reason' in info and info['reason'] == 'Valid placement':
            placed_nodes += 1
            print(f"Successfully placed node {placed_nodes}/{max_nodes}")

            # Reset the repeated action counter after a successful placement
            repeated_action_count = 0

            # If we've placed all nodes, we're done
            if placed_nodes >= max_nodes:
                print("Placed all nodes successfully")
                break

    print(f"Evaluation completed after {step_count} steps with {placed_nodes} nodes placed")

    # Get the final nodes
    final_nodes, final_node_types = eval_env.get_full_nodes()

    # Save the nodes
    np.savez(
        os.path.join(output_dir, 'node_placement', 'final_nodes.npz'),
        nodes=final_nodes,
        node_types=final_node_types
    )

    # Visualize the nodes
    fig, ax = plot_airspace_network(
        nodes=final_nodes,
        edges=[],
        node_types=final_node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        title='Node Placement Result'
    )

    plt.savefig(os.path.join(output_dir, 'node_placement', 'final_nodes.png'))
    plt.close(fig)

    return model, (final_nodes, final_node_types)

def train_graph_construction(
    cartesian_config: Dict[str, Any],
    nodes: np.ndarray,
    node_types: List[int],
    output_dir: str,
    n_envs: int = 1,  # Using 1 environment for graph construction is safer
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    device: str = 'auto',
    seed: int = 42,
    max_edges: int = 100,
    total_timesteps: int = 1000000
) -> Tuple[PPO, List[Tuple[int, int]]]:
    """
    Train the graph construction model.

    Args:
        cartesian_config: Configuration with Cartesian coordinates
        nodes: Node coordinates
        node_types: Node types
        output_dir: Directory to save outputs
        n_envs: Number of parallel environments
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size for SGD
        n_epochs: Number of epochs for optimization
        learning_rate: Learning rate
        gamma: Discount factor
        device: Device to run the model on
        seed: Random seed
        max_edges: Maximum number of edges
        total_timesteps: Total number of timesteps to train for

    Returns:
        Tuple of (trained model, list of edges)
    """
    # Set random seed
    set_random_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'graph_construction'), exist_ok=True)

    # Extract frontline and airport indices
    frontline_indices = [i for i, t in enumerate(node_types) if t == 0]
    airport_indices = [i for i, t in enumerate(node_types) if t == 1]

    # For testing, create a basic environment first instead of vectorized one
    test_env = GraphConstructionEnv(
        nodes=nodes,
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        frontline_indices=frontline_indices,
        airport_indices=airport_indices,
        max_edges=max_edges,
        max_angle_deg=80.0
    )

    # Test reset to make sure it works - extract just the observation
    test_reset_result = test_env.reset()
    test_obs = test_reset_result[0]

    # Create environment factory functions
    env_fns = []
    for _ in range(n_envs):
        env_fns.append(lambda: GraphConstructionEnv(
            nodes=nodes,
            node_types=node_types,
            adi_zones=cartesian_config['adi_zones'],
            danger_zones=cartesian_config['danger_zones'],
            frontline_indices=frontline_indices,
            airport_indices=airport_indices,
            max_edges=max_edges,
            max_angle_deg=80.0
        ))

    # Create vectorized environments using DummyVecEnv instead of SubprocVecEnv
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps * 10,
        save_path=os.path.join(output_dir, 'graph_construction', 'checkpoints'),
        name_prefix='graph_construction_model'
    )

    progress_callback = TrainingProgressCallback(
        save_path=os.path.join(output_dir, 'graph_construction'),
        save_freq=10
    )

    # Create and train model
    # Note: We must use MultiInputPolicy instead of MlpPolicy for Dict observation space
    model = PPO(
        'MultiInputPolicy',  # Changed from 'MlpPolicy' to 'MultiInputPolicy'
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        device=device,
        seed=seed
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback]
    )

    # Save final model
    model.save(os.path.join(output_dir, 'graph_construction', 'final_model'))

    # Run evaluation
    print("Running graph construction evaluation...")

    # Create evaluation environment
    eval_env = GraphConstructionEnv(
        nodes=nodes,
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        frontline_indices=frontline_indices,
        airport_indices=airport_indices,
        max_edges=max_edges,
        max_angle_deg=80.0
    )

    # Get initial observation
    reset_result = eval_env.reset(seed=seed+100)  # Use a different seed
    obs = reset_result[0]  # Extract just the observation

    # Prepare for evaluation
    max_steps = 100  # Limit to prevent infinite loops
    step_count = 0
    done = False
    repeated_action_count = 0
    last_action = None
    added_edges = 0
    max_repeat_tries = 5  # Max number of times to try the same action

    # Evaluation loop with safety checks
    try:
        while not done and step_count < max_steps:
            # Get action from model
            action, _ = model.predict(obs, deterministic=False)  # Use stochastic action to avoid getting stuck

            # Check if we're repeating the same action
            if last_action is not None and action == last_action:
                repeated_action_count += 1

                # If we've tried the same action too many times, try random action
                if repeated_action_count >= max_repeat_tries:
                    print(f"Action {action} repeated {repeated_action_count} times, trying random action...")
                    # Try a random action from the action space
                    action = eval_env.action_space.sample()
                    repeated_action_count = 0
            else:
                repeated_action_count = 0
                last_action = action

            # Take step in environment
            step_result = eval_env.step(action)
            obs, reward, done, _, info = step_result

            step_count += 1

            # If this added an edge, record it
            if 'reason' not in info or info['reason'] != 'Invalid action index':
                added_edges += 1
                print(f"Added edge {added_edges}/{max_edges}")

            # If we've reached connectivity goal, we're done
            if 'num_connected_pairs' in info and 'total_pairs' in info:
                if info['num_connected_pairs'] == info['total_pairs']:
                    print(f"All pairs connected! ({info['num_connected_pairs']}/{info['total_pairs']})")
                    break
    except Exception as e:
        print(f"Error during evaluation: {e}")

    print(f"Graph evaluation completed after {step_count} steps with {added_edges} edges added")

    # Get the final network
    _, _, final_edges = eval_env.get_network()

    # Save the edges
    np.savez(
        os.path.join(output_dir, 'graph_construction', 'final_edges.npz'),
        edges=np.array(final_edges)
    )

    try:
        # Get network evaluation
        network_eval = eval_env.get_network_evaluation()

        # Save evaluation results
        with open(os.path.join(output_dir, 'graph_construction', 'network_evaluation.txt'), 'w') as f:
            for key, value in network_eval.items():
                if key not in ['paths', 'adi_traversal_metrics', 'angle_metrics']:
                    f.write(f"{key}: {value}\n")
    except Exception as e:
        print(f"Warning: Error evaluating network: {e}")

    try:
        # Visualize the network
        fig, ax = plot_airspace_network(
            nodes=nodes,
            edges=final_edges,
            node_types=node_types,
            adi_zones=cartesian_config['adi_zones'],
            danger_zones=cartesian_config['danger_zones'],
            title='Graph Construction Result'
        )

        plt.savefig(os.path.join(output_dir, 'graph_construction', 'final_network.png'))
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Error visualizing network: {e}")

    return model, final_edges

def main():
    """
    Main function to run the training process.
    """
    parser = argparse.ArgumentParser(description='Train the airspace network planning system.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to run on (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--node_timesteps', type=int, default=1000000, help='Total timesteps for node placement')
    parser.add_argument('--graph_timesteps', type=int, default=1000000, help='Total timesteps for graph construction')
    parser.add_argument('--max_nodes', type=int, default=30, help='Maximum number of intermediate nodes')
    parser.add_argument('--max_edges', type=int, default=100, help='Maximum number of edges')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments for node placement')

    args = parser.parse_args()

    # Transform latitude-longitude config to Cartesian coordinates
    cartesian_config = transform_config_to_cartesian(latlon_config)

    # Train node placement model
    print("Training node placement model...")
    node_model, (nodes, node_types) = train_node_placement(
        cartesian_config=cartesian_config,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_nodes=args.max_nodes,
        total_timesteps=args.node_timesteps,
        n_envs=args.n_envs
    )

    print(f"Node placement complete. Generated {len(nodes)} nodes.")

    # Train graph construction model
    print("Training graph construction model...")
    graph_model, edges = train_graph_construction(
        cartesian_config=cartesian_config,
        nodes=nodes,
        node_types=node_types,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_edges=args.max_edges,
        total_timesteps=args.graph_timesteps,
        n_envs=1  # Using 1 environment for graph construction as it can be complex
    )

    print(f"Graph construction complete. Generated {len(edges)} edges.")

    # Save final results (both nodes and edges)
    np.savez(
        os.path.join(args.output_dir, 'final_network.npz'),
        nodes=nodes,
        node_types=node_types,
        edges=np.array(edges)
    )

    print(f"Final results saved to {args.output_dir}/final_network.npz")

if __name__ == '__main__':
    main()