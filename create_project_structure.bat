@echo off
echo Creating airspace network planning project directory structure...

REM Create main project directory
REM mkdir airspace_network_planning
REM cd airspace_network_planning

REM Create config directory
mkdir config
echo Creating Python module files...
echo # -*- coding: utf-8 -*-> config\__init__.py

REM Create environment directory
mkdir environment
echo # -*- coding: utf-8 -*-> environment\__init__.py

REM Create models directory
mkdir models
echo # -*- coding: utf-8 -*-> models\__init__.py

REM Create utils directory
mkdir utils
echo # -*- coding: utf-8 -*-> utils\__init__.py

REM Create output and evaluation directories
mkdir output
mkdir evaluation

REM Create the initial Python files
echo # -*- coding: utf-8 -*-> train.py
echo # -*- coding: utf-8 -*-> evaluate.py
echo # -*- coding: utf-8 -*-> README.md
echo # -*- coding: utf-8 -*-> requirements.txt

REM Create module structure in environment
echo """Initialization file for the environment module."""> environment\__init__.py
echo """Node placement environment for reinforcement learning."""> environment\node_env.py
echo """Graph construction environment for reinforcement learning."""> environment\graph_env.py
echo """Utilities for the reinforcement learning environments."""> environment\utils.py

REM Create module structure in models
echo """Initialization file for the models module."""> models\__init__.py
echo """Common model components for reinforcement learning."""> models\common.py
echo """Node placement policy models for reinforcement learning."""> models\node_placement.py
echo """Graph construction policy models for reinforcement learning."""> models\graph_construction.py

REM Create module structure in utils
echo """Initialization file for the utils module."""> utils\__init__.py
echo """Geometry utility functions for airspace network planning."""> utils\geometry.py
echo """Clustering utilities for identifying outlier nodes and groups."""> utils\clustering.py
echo """Visualization utilities for the airspace network planning system."""> utils\visualization.py
echo """Utilities for transforming geographical coordinates to Cartesian coordinates."""> utils\coordinate_transform.py

REM Create config file
echo """Configuration file for latitude-longitude data of SAM zones, airports, danger zones, and frontline positions."""> config\latlon_config.py

REM Output success message
echo.
echo Project directory structure successfully created!
echo.
cd ..
echo Directory structure created in %CD%\airspace_network_planning
echo.
pause