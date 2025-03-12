"""
Test script to debug the evaluation part of the node placement.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random

from config.latlon_config import latlon_config
from utils.coordinate_transform import transform_config_to_cartesian
from environment.node_env import NodePlacementEnv
from utils.visualization import plot_airspace_network


def debug_print(msg, flush=True):
    """Print debug message with timestamp."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {msg}", flush=flush)


def test_environment():
    """
    Test the environment directly without involving training.
    """
    debug_print("Starting environment test")

    # Load configuration
    debug_print("Loading configuration")
    cartesian_config = transform_config_to_cartesian(latlon_config)

    # Create environment
    debug_print("Creating environment")
    env = NodePlacementEnv(
        cartesian_config=cartesian_config,
        max_nodes=10,  # Smaller for quicker testing
        min_distance=10.0,
        max_distance=150.0
    )

    # Test reset
    debug_print("Testing reset")
    reset_result = env.reset(seed=42)

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
        debug_print(f"Reset successful - Observation shape: {obs.shape}, Info: {info}")
    else:
        debug_print(f"WARNING: Unexpected reset format: {reset_result}")
        obs = reset_result

    # Run a few manual steps
    debug_print("Running manual steps")
    num_steps = 10
    done = False
    step_count = 0

    while not done and step_count < num_steps:
        # Create a valid action (important to avoid invalid placements)
        # We'll use a smarter approach than pure random sampling

        # Get bounds from the observation
        fixed_nodes = env.fixed_nodes

        # Sample a position near an existing node
        if len(fixed_nodes) > 0:
            # Pick a random fixed node
            random_node = fixed_nodes[random.randint(0, len(fixed_nodes) - 1)]

            # Add a small random offset (20-40km)
            offset_distance = random.uniform(20.0, 40.0)
            angle = random.uniform(0, 2 * np.pi)

            x = random_node[0] + offset_distance * np.cos(angle)
            y = random_node[1] + offset_distance * np.sin(angle)

            # Ensure within bounds
            x = max(env.x_min, min(env.x_max, x))
            y = max(env.y_min, min(env.y_max, y))

            # Random node type (0 for common, 1 for outlier)
            node_type = random.randint(0, 1)

            action = np.array([x, y, node_type])
        else:
            # Fallback to random sampling
            action = env.action_space.sample()

        debug_print(f"Step {step_count} - Action: {action}")

        # Take step
        step_result = env.step(action)

        if isinstance(step_result, tuple):
            if len(step_result) == 5:  # New Gym API
                next_obs, reward, done, truncated, info = step_result
                debug_print(f"Step result - Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
            elif len(step_result) == 4:  # Old Gym API
                next_obs, reward, done, info = step_result
                debug_print(f"Step result (old API) - Reward: {reward}, Done: {done}, Info: {info}")
            else:
                debug_print(f"WARNING: Unexpected step result length: {len(step_result)}")
                break
        else:
            debug_print(f"WARNING: Step result is not a tuple: {step_result}")
            break

        obs = next_obs
        step_count += 1

    debug_print(f"Completed {step_count} steps, Final done state: {done}")

    # Get final nodes
    debug_print("Getting final nodes")
    final_nodes, final_node_types = env.get_full_nodes()
    debug_print(f"Got {len(final_nodes)} nodes, {len(final_node_types)} node types")

    # Try to visualize
    debug_print("Visualizing nodes")
    try:
        os.makedirs("./test_output", exist_ok=True)

        fig, ax = plot_airspace_network(
            nodes=final_nodes,
            edges=[],
            node_types=final_node_types,
            adi_zones=cartesian_config['adi_zones'],
            danger_zones=cartesian_config['danger_zones'],
            title='Environment Test Result'
        )

        plt.savefig("./test_output/env_test.png")
        plt.close(fig)
        debug_print("Visualization saved to ./test_output/env_test.png")
    except Exception as e:
        debug_print(f"ERROR in visualization: {e}")
        import traceback
        traceback.print_exc()

    debug_print("Environment test completed")


if __name__ == "__main__":
    test_environment()