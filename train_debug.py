"""
Debug version of the training script with extensive logging.
"""

import os
import argparse
import numpy as np
import torch
import time
import sys
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
from utils.visualization import plot_airspace_network


def debug_print(msg, flush=True):
    """Print debug message with timestamp."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {msg}", flush=flush)


class DebugCallback(BaseCallback):
    """
    Callback for debugging training issues.
    """

    def __init__(self, verbose=0):
        super(DebugCallback, self).__init__(verbose)
        self.update_count = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % 100 == 0:
            debug_print(f"Training step {self.step_count}")
        return True

    def _on_rollout_end(self) -> None:
        self.update_count += 1
        debug_print(f"Finished rollout {self.update_count}")


def train_node_placement_debug(
        cartesian_config: Dict[str, Any],
        output_dir: str,
        total_timesteps: int = 100,
        seed: int = 42,
        max_nodes: int = 30
) -> Tuple[PPO, Tuple[np.ndarray, List[int]]]:
    """
    Debug version of node placement training with minimal options.

    Args:
        cartesian_config: Configuration with Cartesian coordinates
        output_dir: Directory to save outputs
        total_timesteps: Total timesteps for training
        seed: Random seed
        max_nodes: Maximum number of intermediate nodes

    Returns:
        Tuple of (trained model, tuple of (nodes, node_types))
    """
    debug_print("Setting up output directory")
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    debug_print(f"Setting random seed: {seed}")
    set_random_seed(seed)

    # Create a single environment for debugging
    debug_print("Creating node placement environment")
    env = NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=max_nodes,
        min_distance=10.0,
        max_distance=150.0
    )

    # Test the environment manually
    debug_print("Testing environment reset")
    reset_result = env.reset(seed=seed)
    debug_print(f"Reset result type: {type(reset_result)}")

    # Check if reset returns the expected tuple
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
        debug_print(f"Observation shape: {obs.shape}, Info: {info}")
    else:
        debug_print(f"WARNING: Unexpected reset result format: {reset_result}")
        debug_print("Attempting to continue anyway...")
        obs = reset_result  # Try to use whatever was returned

    # Test a single step
    debug_print("Testing environment step")
    action = env.action_space.sample()
    debug_print(f"Sampled action: {action}")

    try:
        step_result = env.step(action)
        debug_print(f"Step result type: {type(step_result)}")

        # Check step result format
        if isinstance(step_result, tuple):
            if len(step_result) == 5:  # New Gym API
                obs, reward, done, truncated, info = step_result
                debug_print(f"Step successful - Reward: {reward}, Done: {done}")
            elif len(step_result) == 4:  # Old Gym API
                obs, reward, done, info = step_result
                debug_print(f"Step successful (old API) - Reward: {reward}, Done: {done}")
            else:
                debug_print(f"WARNING: Unexpected step result length: {len(step_result)}")
        else:
            debug_print(f"WARNING: Step result is not a tuple: {step_result}")
    except Exception as e:
        debug_print(f"ERROR in environment step: {e}")
        import traceback
        traceback.print_exc()
        return None, (None, None)

    # If environment tests passed, wrap in VecEnv for SB3
    debug_print("Wrapping environment for SB3")
    vec_env = DummyVecEnv([lambda: NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=max_nodes,
        min_distance=10.0,
        max_distance=150.0
    )])

    vec_env = VecMonitor(vec_env)

    # Create PPO model with minimal settings
    debug_print("Creating PPO model")
    try:
        model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            seed=seed,
            n_steps=64,  # Smaller batch for faster updates
            batch_size=32
        )
    except Exception as e:
        debug_print(f"ERROR creating PPO model: {e}")
        import traceback
        traceback.print_exc()
        return None, (None, None)

    # Add debug callback
    debug_callback = DebugCallback()

    # Train model with minimal steps
    debug_print(f"Starting training for {total_timesteps} steps")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[debug_callback]
        )
        debug_print("Training completed successfully")
    except Exception as e:
        debug_print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return model, (None, None)

    # Save the model
    debug_print("Saving model")
    model_path = os.path.join(output_dir, 'node_model.zip')
    model.save(model_path)
    debug_print(f"Model saved to {model_path}")

    # Run evaluation
    debug_print("Starting evaluation")
    eval_env = NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=max_nodes,
        min_distance=10.0,
        max_distance=150.0
    )

    # Generate nodes using the trained model
    debug_print("Getting initial observation")
    try:
        reset_result = eval_env.reset()

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            debug_print(f"WARNING: Unexpected eval reset format: {reset_result}")
            obs = reset_result

        debug_print(f"Initial observation shape: {obs.shape}")
        done = False
        steps = 0
        max_steps = 100  # Safety limit

        debug_print("Running prediction loop")
        while not done and steps < max_steps:
            debug_print(f"Prediction step {steps}")
            action, _ = model.predict(obs, deterministic=True)
            debug_print(f"Predicted action: {action}")

            step_result = eval_env.step(action)

            if isinstance(step_result, tuple):
                if len(step_result) == 5:  # New Gym API
                    obs, reward, done, _, info = step_result
                elif len(step_result) == 4:  # Old Gym API
                    obs, reward, done, info = step_result
                else:
                    debug_print(f"WARNING: Unexpected eval step length: {len(step_result)}")
                    break
            else:
                debug_print(f"WARNING: Eval step result not a tuple: {step_result}")
                break

            debug_print(f"Step result - Reward: {reward}, Done: {done}, Info: {info}")
            steps += 1

        debug_print(f"Evaluation completed after {steps} steps, Done: {done}")

        # Get the final nodes
        debug_print("Getting final nodes")
        final_nodes, final_node_types = eval_env.get_full_nodes()
        debug_print(f"Generated {len(final_nodes)} nodes")

        # Save the nodes
        debug_print("Saving nodes")
        np.savez(
            os.path.join(output_dir, 'final_nodes.npz'),
            nodes=final_nodes,
            node_types=final_node_types
        )

        # Try to visualize
        debug_print("Visualizing nodes")
        try:
            fig, ax = plot_airspace_network(
                nodes=final_nodes,
                edges=[],
                node_types=final_node_types,
                adi_zones=cartesian_config['adi_zones'],
                danger_zones=cartesian_config['danger_zones'],
                title='Node Placement Result'
            )

            plt.savefig(os.path.join(output_dir, 'nodes.png'))
            plt.close(fig)
            debug_print("Visualization saved")
        except Exception as e:
            debug_print(f"ERROR in visualization: {e}")

        return model, (final_nodes, final_node_types)

    except Exception as e:
        debug_print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return model, (None, None)


def main():
    """
    Main function for the debug script.
    """
    parser = argparse.ArgumentParser(description='Debug the node placement training.')
    parser.add_argument('--output_dir', type=str, default='./debug_output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--timesteps', type=int, default=100, help='Number of timesteps')

    args = parser.parse_args()

    debug_print("Starting debug script")
    debug_print(f"Python version: {sys.version}")
    debug_print(f"NumPy version: {np.__version__}")
    debug_print(f"PyTorch version: {torch.__version__}")

    # Transform latitude-longitude config to Cartesian coordinates
    debug_print("Loading configuration")
    cartesian_config = transform_config_to_cartesian(latlon_config)

    # Train node placement model with debugging
    debug_print("Starting node placement training debug")
    node_model, (nodes, node_types) = train_node_placement_debug(
        cartesian_config=cartesian_config,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        seed=args.seed
    )

    debug_print("Debug script completed")


if __name__ == '__main__':
    main()