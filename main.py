from simulation import SimulationEnvironment

if __name__ == '__main__':
    # Initialize the simulation
    env = SimulationEnvironment()
    env.simulate()
    env.plot_results()
