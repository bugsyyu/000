# airspace_netword_planning
# 空间路网规划项目使用指南

本项目使用强化学习来规划空中通道网络，满足内径避开、外径穿越、转角控制等多种约束。
## Project Structure

```
airspace_network_planning/
├── config/
│   └── latlon_config.py   # Configuration for coordinates and zones
├── environment/
│   ├── __init__.py
│   ├── node_env.py        # Environment for node generation
│   ├── graph_env.py       # Environment for graph construction
│   └── utils.py           # Common utilities for environments
├── models/
│   ├── __init__.py
│   ├── node_placement.py  # Node placement policy network
│   ├── graph_construction.py  # Graph construction policy network
│   └── common.py          # Common model components
├── utils/
│   ├── __init__.py
│   ├── geometry.py        # Geometric calculations
│   ├── visualization.py   # Visualization utilities
│   ├── clustering.py      # Clustering for outlier detection
│   └── coordinate_transform.py  # Coordinate transformation utilities
├── train.py               # Main training script
└── evaluate.py            # Evaluation and visualization script


```