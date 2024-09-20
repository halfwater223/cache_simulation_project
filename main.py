"""
Cache Simulation Project
Author: Yinxuan Wu, Ning Wang
Institution: Network Science Lab, ECE Department, UMass Amherst
License: MIT License

MIT License

Copyright (c) 2024 Yinxuan Wu, Ning Wang
"""

from cache_replacement_policy import LFUCache, QLearningCache
from simulation_environment import SimulationEnvironment

if __name__ == '__main__':
    # Initialize cache policies
    num_types = 20
    cache_size = 5
    lfu_cache = LFUCache(num_types, cache_size)
    ql_cache = QLearningCache(num_types, cache_size)
    # Initialize the simulation
    env = SimulationEnvironment(num_types=num_types, cache_size=cache_size, cache_policies=[lfu_cache, ql_cache])
    env.simulate()
    env.plot_results()
