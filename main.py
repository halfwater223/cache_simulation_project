"""
Cache Simulation Project
Author: Yinxuan Wu, Ning Wang
Institution: Network Science Lab, ECE Department, UMass Amherst
License: MIT License

MIT License

Copyright (c) 2024 Yinxuan Wu, Ning Wang
"""
from simulation import SimulationEnvironment

if __name__ == '__main__':
    # Initialize the simulation
    env = SimulationEnvironment()
    env.simulate()
    env.plot_results()
