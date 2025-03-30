"""
Training script for the airspace network planning system, with user-selectable log verbosity,
and allowing custom n_steps for PPO in node_env & graph_env.

We define a CustomLoggingCallback to control:
- log_level=1: minimal logs (episodes overall)
- log_level=2: per-episode logs
- log_level=3: per-step logs
And we use Python logging instead of print, also configured to output to a dedicated log file.

IMPORTANT FIX:
- Removed all unicode emojis (✅, ❌) in logs to avoid gbk encoding error on Windows.
- We add --node_n_steps and --graph_n_steps to override the default n_steps=2048, so that
  if your environment ends in 1 step, you won't be forced to run 2048 episodes per update.
"""

import logging
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
from utils.visualization import plot_airspace_network, plot_training_progress
from environment.utils import evaluate_network, find_shortest_paths

logger = logging.getLogger(__name__)

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
            logger.info("[CustomLoggingCallback] Training start...")

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
            logger.debug("[log_level=3][Step] Episode %d Step %d Action=%s Reward=%.3f Done=%s",
                         self.episode_count, self.current_episode_length, str(a), r, d)
        if d:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            if self.log_level >= 2:
                logger.info("[log_level=2][Episode End] Episode=%d Length=%d Reward=%.3f",
                            self.episode_count, self.current_episode_length, self.current_episode_reward)
            self.current_episode_reward = 0
            self.current_episode_length = 0
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
            logger.info("[log_level=1] Currently finished %d episodes so far...", self.episode_count)

    def _save_progress(self):
        """
        Example saving method: we can do a plot of training progress
        or just store arrays. This is up to you.
        """
        pass

    def _on_training_end(self) -> None:
        if self.log_level >= 1:
            logger.info("[CustomLoggingCallback] Training ended. Total episodes finished: %d", self.episode_count)


# ------------------- NodePlacement training function -------------------
def train_node_placement(
    cartesian_config: Dict[str, Any],
    output_dir: str,
    n_envs: int = 4,
    n_steps: int = 2048,  # <--- We can now pass a smaller number to avoid 2048 episodes
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    device: str = 'auto',
    seed: int = 42,
    max_nodes: int = 30,
    total_timesteps: int = 1000000,
    log_level: int = 1
) -> Tuple[PPO, Tuple[np.ndarray, List[int]]]:
    """
    Train the node placement model.

    Args:
        cartesian_config: Configuration with Cartesian coordinates
        output_dir: Directory to save outputs
        n_envs: Number of parallel environments
        n_steps: Number of steps per environment per update (PPO param)
        batch_size: Minibatch size for SGD
        n_epochs: Number of epochs for optimization
        learning_rate: Learning rate
        gamma: Discount factor
        device: Device to run the model on
        seed: Random seed
        max_nodes: Maximum number of intermediate nodes
        total_timesteps: Total number of timesteps to train for
        log_level: Logging verbosity level (1= minimal, 2= info, 3= debug)

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
            debug_invalid_placement=True,
            log_level=log_level
        ))

    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    # Additional callback for logging
    custom_log_callback = CustomLoggingCallback(
        log_level=log_level,
        save_path=os.path.join(output_dir, 'node_placement'),
        save_freq=50,
        verbose=1
    )

    # We can also keep a checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps * 10,
        save_path=os.path.join(output_dir, 'node_placement', 'checkpoints'),
        name_prefix='node_placement_model'
    )

    # Create and train model, overriding n_steps
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,          # <=== KEY: override from argument
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,  # stable-baselines internal verbosity
        device=device,
        seed=seed
    )

    logger.info("Starting NodePlacementEnv training with total_timesteps=%d and n_steps=%d ...",
                total_timesteps, n_steps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[custom_log_callback, checkpoint_callback]
    )

    # Save final model
    model.save(os.path.join(output_dir, 'node_placement', 'final_model'))
    logger.info("NodePlacementEnv model saved at: %s", os.path.join(output_dir, 'node_placement', 'final_model'))

    # Quick evaluation
    logger.info("Running node placement evaluation (quick test)...")
    eval_env = NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=max_nodes,
        min_distance=10.0,
        max_distance=150.0,
        debug_invalid_placement=True,
        log_level=log_level
    )

    # Reset
    reset_result = eval_env.reset(seed=seed+100)
    obs = reset_result[0]

    max_steps = 50
    step_count = 0
    done = False

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, info = eval_env.step(action)
        step_count += 1
        if 'reason' in info and info['reason'] == 'Nodes placed in reset()':
            # We break immediately
            break

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
    n_steps: int = 2048,  # <=== again, we allow override
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    device: str = 'auto',
    seed: int = 42,
    max_edges: int = 100,
    total_timesteps: int = 1000000,
    log_level: int = 1
) -> Tuple[PPO, List[Tuple[int, int]]]:
    """
    Train the graph construction model.
    """
    set_random_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'graph_construction'), exist_ok=True)

    frontline_indices = [i for i, t in enumerate(node_types) if t == 0]
    airport_indices = [i for i, t in enumerate(node_types) if t == 1]

    test_env = GraphConstructionEnv(
        nodes=nodes,
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        frontline_indices=frontline_indices,
        airport_indices=airport_indices,
        max_edges=max_edges,
        max_angle_deg=80.0,
        log_level=log_level
    )
    test_reset_result = test_env.reset()

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
            max_angle_deg=80.0,
            log_level=log_level
        ))

    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

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

    model = PPO(
        'MultiInputPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,  # <=== override from argument
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        device=device,
        seed=seed
    )

    logger.info("Starting GraphConstructionEnv training with total_timesteps=%d and n_steps=%d ...",
                total_timesteps, n_steps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[custom_log_callback, checkpoint_callback]
    )

    model.save(os.path.join(output_dir, 'graph_construction', 'final_model'))
    logger.info("GraphConstructionEnv model saved at: %s", os.path.join(output_dir, 'graph_construction', 'final_model'))

    # quick test
    logger.info("Running graph construction evaluation (quick test) ...")
    eval_env = GraphConstructionEnv(
        nodes=nodes,
        node_types=node_types,
        adi_zones=cartesian_config['adi_zones'],
        danger_zones=cartesian_config['danger_zones'],
        frontline_indices=frontline_indices,
        airport_indices=airport_indices,
        max_edges=max_edges,
        max_angle_deg=80.0,
        log_level=log_level
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

    try:
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=False)
            if last_action == action:
                repeated_action_count += 1
                if repeated_action_count >= max_repeat_tries:
                    action = eval_env.action_space.sample()
                    repeated_action_count = 0
            else:
                repeated_action_count = 0
                last_action = action
            obs, reward, done, _, info = eval_env.step(action)
            step_count += 1
            if 'reason' not in info or info['reason'] != 'action index out of range':
                added_edges += 1

            if 'num_connected_pairs' in info and info['total_pairs'] == info['num_connected_pairs']:
                logger.info("All pairs connected! Done at step %d", step_count)
                break
    except Exception as e:
        logger.error("Error during evaluation: %s", str(e))

    logger.info("Graph evaluation completed after %d steps with %d edges added", step_count, added_edges)
    _, _, final_edges = eval_env.get_network()
    logger.info("[Auto-Eval] Checking final pairwise connectivity:")
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
            logger.info("  Pair(frontline=%d, airport=%d): %s", f_, a_, status_)
            idx_ += 1
    connected_count = sum(success_flags)
    total_count = len(success_flags)
    logger.info("[Auto-Eval] connected pairs: %d/%d", connected_count, total_count)
    if connected_count == total_count:
        logger.info("[Auto-Eval] => All pairs are connected! [OK]")
    else:
        logger.info("[Auto-Eval] => Not all pairs connected. [X]")

    np.savez(
        os.path.join(output_dir, 'graph_construction', 'final_edges.npz'),
        edges=np.array(final_edges)
    )

    try:
        network_eval = eval_env.get_network_evaluation()
        with open(os.path.join(output_dir, 'graph_construction', 'network_evaluation.txt'), 'w') as f:
            for key, value in network_eval.items():
                if key not in ['paths', 'adi_traversal_metrics', 'angle_metrics']:
                    f.write(f"{key}: {value}\n")
    except Exception as e:
        logger.error("Warning: Error evaluating network: %s", str(e))

    try:
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
        logger.error("Warning: Error visualizing network: %s", str(e))

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
    parser.add_argument('--log_level', type=int, default=1, help='Logging verbosity level (1=min,2=info,3=debug)')

    # NEW: let user override PPO n_steps for node & graph
    parser.add_argument('--node_n_steps', type=int, default=8, help='Number of steps per rollout in node_env PPO')
    parser.add_argument('--graph_n_steps', type=int, default=2048, help='Number of steps per rollout in graph_env PPO')

    args = parser.parse_args()

    level = logging.INFO
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'train.log')
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    cartesian_config = transform_config_to_cartesian(latlon_config)

    # 1) Node placement
    logger.info("Training node placement model...")
    node_model, (nodes, node_types) = train_node_placement(
        cartesian_config=cartesian_config,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_nodes=args.max_nodes,
        total_timesteps=args.node_timesteps,
        n_envs=args.n_envs,
        log_level=args.log_level,
        # override n_steps for node env
        n_steps=args.node_n_steps
    )
    logger.info("Node placement complete. Generated %d nodes.", len(nodes))

    # 2) Graph construction
    logger.info("Training graph construction model...")
    graph_model, edges = train_graph_construction(
        cartesian_config=cartesian_config,
        nodes=nodes,
        node_types=node_types,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_edges=args.max_edges,
        total_timesteps=args.graph_timesteps,
        n_envs=1,  # typically we do 1 env for graph, but could do more
        log_level=args.log_level,
        # override n_steps for graph env
        n_steps=args.graph_n_steps
    )
    logger.info("Graph construction complete. Generated %d edges.", len(edges))

    # Save final results
    np.savez(
        os.path.join(args.output_dir, 'final_network.npz'),
        nodes=nodes,
        node_types=node_types,
        edges=np.array(edges)
    )
    logger.info("Final results saved to %s", os.path.join(args.output_dir, 'final_network.npz'))

if __name__ == '__main__':
    main()
