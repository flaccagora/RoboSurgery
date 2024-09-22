"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)


register(
     id='lungs-v0',
     entry_point='simulator.envs.lung.lungs:LungEnv',
     max_episode_steps=250,
)



# Hook to load plugins from entry points
load_plugin_envs()
