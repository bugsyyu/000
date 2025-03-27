"""
Training script for the airspace network planning system, with user-selectable log verbosity.

We define a CustomLoggingCallback to control:
- log_level=1: minimal logs (episodes overall)
- log_level=2: per-episode logs
- log_level=3: per-step logs
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
from environment.utils import evaluate_network, find_shortest_paths

# ------------------- Custom callback with log_level -------------------
class CustomLoggingCallback(BaseCallback):
    """
    A unified callback that handles:
      - Logging training progress for the user
      - Different verbosity levels (log_level):
         1 = minimal logs (just overall episode count)
         2 = logs per episode
         3 = logs each step
    """

    def __init__(
        self,
        log_level: int = 1,
        save_path: str = None,
        save_freq: int = 10,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_level = log_level
        self.save_path = save_path
        self.save_freq = save_freq
        # We'll track some stats:
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_count = 0

    def _init_callback(self) -> None:
        """
        Called once at the start of training.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        if self.log_level >= 1:
            print("[CustomLoggingCallback] Training start...")

    def _on_step(self) -> bool:
        # Each time we step in the environment
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        actions = self.locals['actions']

        # We'll assume it's vectorized env. Typically we look at index=0
        # if n_envs=1. For multiple envs, you'd loop or handle differently.
        r = rewards[0]
        d = dones[0]
        a = actions[0] if len(actions.shape) > 0 else actions

        self.current_episode_reward += r
        self.current_episode_length += 1

        if self.log_level >= 3:
            # Step-level logs
            print(f"[log_level=3][Step] Episode {self.episode_count} Step {self.current_episode_length} "
                  f"Action={a} Reward={r:.3f} Done={d}")

        if d:
            # Episode just ended
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1

            if self.log_level >= 2:
                # Per-episode logs
                print(f"[log_level=2][Episode End] Episode={self.episode_count} "
                      f"Length={self.current_episode_length} Reward={self.current_episode_reward:.3f}")

            # reset
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # Optionally save progress
            if self.save_path is not None:
                if self.episode_count % self.save_freq == 0:
                    # E.g. we can save a partial plot or something
                    self._save_progress()

        return True

    def _on_rollout_end(self) -> None:
        """
        Called after each rollout collection (i.e. n_steps * n_envs).
        You can also do iteration-level logs here if log_level=1
        """
        if self.log_level == 1:
            # Minimal logs at iteration level
            # e.g. show how many episodes so far
            print(f"[log_level=1] Currently finished {self.episode_count} episodes so far...")

    def _save_progress(self):
        """
        Example saving method: we can do a plot of training progress
        or just store arrays. This is up to you.
        """
        pass  # you could replicate the logic in the old trainingprogresscallback

    def _on_training_end(self) -> None:
        if self.log_level >= 1:
            print(f"[CustomLoggingCallback] Training ended. Total episodes finished: {self.episode_count}")


# ------------------- NodePlacement training function -------------------
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
    total_timesteps: int = 1000000,
    log_level: int = 1  # <--- new param
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

    # Create env
    env_fns = []
    for _ in range(n_envs):
        env_fns.append(lambda: NodePlacementEnv(
            cartesian_config=cartesian_config,
            max_nodes=max_nodes,
            min_distance=10.0,
            max_distance=150.0,
            log_level=log_level
        ))

    # Create vectorized environments using DummyVecEnv instead of SubprocVecEnv
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    # Additional callback for logging
    custom_log_callback = CustomLoggingCallback(
        log_level=log_level,
        save_path=os.path.join(output_dir, 'node_placement'),
        save_freq=50,  # e.g. save every 50 episodes or so
        verbose=1
    )

    # We can also keep a checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps * 10,
        save_path=os.path.join(output_dir, 'node_placement', 'checkpoints'),
        name_prefix='node_placement_model'
    )

    # Create and train model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,  # stable-baselines internal verbosity, separate from our log_level
        device=device,
        seed=seed
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[custom_log_callback, checkpoint_callback]  # pass our custom logger
    )

    # Save final model
    model.save(os.path.join(output_dir, 'node_placement', 'final_model'))

    # =============== Run a quick evaluation to place nodes (unchanged) ===============
    print("Running evaluation with improved logic to avoid getting stuck...")

    # Create evaluation environment
    eval_env = NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=max_nodes,
        min_distance=10.0,
        max_distance=150.0,
        log_level=log_level
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
    max_repeat_tries = 5

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
                import math
                import random as pyrand
                x_range = eval_env.x_max - eval_env.x_min
                y_range = eval_env.y_max - eval_env.y_min

                # Get existing nodes to avoid placing too close
                fixed_nodes = eval_env.fixed_nodes
                if len(fixed_nodes) > 0:
                    base_idx = pyrand.randint(0, len(fixed_nodes)-1)
                    base_x, base_y = fixed_nodes[base_idx]
                    offset_dist = pyrand.uniform(30.0, 100.0)
                    angle = pyrand.uniform(0, 2*math.pi)
                    x = base_x + offset_dist*math.cos(angle)
                    y = base_y + offset_dist*math.sin(angle)
                    x = max(eval_env.x_min, min(eval_env.x_max, x))
                    y = max(eval_env.y_min, min(eval_env.y_max, y))
                    node_type = pyrand.randint(0,1)
                    action = np.array([x,y,node_type])
                else:
                    # Fallback to random but more distant
                    action = np.array([
                        eval_env.x_min + x_range * pyrand.random(),
                        eval_env.y_min + y_range * pyrand.random(),
                        pyrand.randint(0,1)
                    ])
                repeated_action_count = 0
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

# ------------------- Graph construction training function -------------------
def train_graph_construction(
    cartesian_config: Dict[str, Any],
    nodes: np.ndarray,
    node_types: List[int],
    output_dir: str,
    n_envs: int = 1,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    device: str = 'auto',
    seed: int = 42,
    max_edges: int = 100,
    total_timesteps: int = 1000000,
    log_level: int = 1  # <--- new param
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

    # Test environment
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

    # Our custom logging callback
    custom_log_callback = CustomLoggingCallback(
        log_level=log_level,
        save_path=os.path.join(output_dir, 'graph_construction'),
        save_freq=50,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps * 10,
        save_path=os.path.join(output_dir, 'graph_construction', 'checkpoints'),
        name_prefix='graph_construction_model'
    )

    # Use MultiInputPolicy for dict obs
    model = PPO(
        'MultiInputPolicy',
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
        callback=[custom_log_callback, checkpoint_callback]
    )

    # Save final model
    model.save(os.path.join(output_dir, 'graph_construction', 'final_model'))

    # ============= 下面是“训练后自动执行评估”阶段  =============
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

    reset_result = eval_env.reset(seed=seed+100)
    obs = reset_result[0]

    max_steps = 100
    step_count = 0
    done = False
    repeated_action_count = 0
    last_action = None
    added_edges = 0
    max_repeat_tries = 5

    # Evaluation loop with safety checks
    try:
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=False)
            # 处理重复动作
            if last_action is not None and action == last_action:
                repeated_action_count += 1

                # If we've tried the same action too many times, try random action
                if repeated_action_count >= max_repeat_tries:
                    print(f"Action {action} repeated {repeated_action_count} times, trying random action...")
                    a = eval_env.action_space.sample()
                    action = a
                    repeated_action_count = 0
            else:
                repeated_action_count = 0
                last_action = action

            # Take step in environment
            step_result = eval_env.step(action)
            obs, reward, done, _, info = step_result
            step_count += 1

            if 'reason' not in info or info['reason'] != 'action index out of range':
                added_edges += 1
                #print(f"Added edge {added_edges}/{max_edges}")

            # If we've reached connectivity goal, we're done
            if 'num_connected_pairs' in info and 'total_pairs' in info:
                if info['num_connected_pairs'] == info['total_pairs']:
                    print(f"All pairs connected! ({info['num_connected_pairs']}/{info['total_pairs']})")
                    break
    except Exception as e:
        print(f"Error during evaluation: {e}")

    print(f"Graph evaluation completed after {step_count} steps with {added_edges} edges added")

    # 拿到最终网络
    _, _, final_edges = eval_env.get_network()

    # 打印连通信息
    print("[Auto-Eval] Checking final pairwise connectivity:")
    all_paths, success_flags = find_shortest_paths(
        nodes,
        final_edges,
        frontline_indices,
        airport_indices,
        node_types=node_types
    )
    idx_ = 0
    for f_ in frontline_indices:
        for a_ in airport_indices:
            status_ = "CONNECTED" if success_flags[idx_] else "NOT CONNECTED"
            print(f"  Pair(frontline={f_}, airport={a_}): {status_}")
            idx_ += 1
    connected_count = sum(success_flags)
    total_count = len(success_flags)
    print(f"[Auto-Eval] connected pairs: {connected_count}/{total_count}")
    if connected_count == total_count:
        print("[Auto-Eval] => All pairs are connected! ✅")
    else:
        print("[Auto-Eval] => Not all pairs connected. ❌")

    # 保存 edges
    np.savez(
        os.path.join(output_dir, 'graph_construction', 'final_edges.npz'),
        edges=np.array(final_edges)
    )

    # try evaluate network
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

    # New argument for log_level
    parser.add_argument('--log_level', type=int, default=1, help='Logging verbosity level (1=minimal,2=episode,3=step)')

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
        n_envs=args.n_envs,
        log_level=args.log_level  # pass
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
        n_envs=1,
        log_level=args.log_level  # pass
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
