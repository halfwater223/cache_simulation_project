# Cache Simulation Project

### Authors: Yinxuan Wu, Ning Wang  
**Institution**: Network Science Lab, ECE Department, University of Massachusetts Amherst 

## Introduction

This project provides a reference simulation environment for students enrolled in the **ECE 627** (Spring 2024) course at UMass Amherst. The simulation is designed to model a caching system where multiple types of tasks are requested by users, and the system must manage a cache of limited size to optimize hit rates.

The current code implements a linear model-based softmax policy gradient optimization caching problem. The system simulates task requests and dynamically adjusts the cache to maximize the **cache hit rate**. The code can be expanded upon by students to further optimize additional metrics such as **computational wait queues** and **latency**, as this current version only optimizes the cache hit rate.

### Key Features:
- **Cache management**: Dynamically adjusts the cache to store 5 contents out of 20 types based on policy optimization.
- **Queue management**: Simulates a server-side queue and tracks queue length over time.
- **Policy optimization**: Uses a softmax policy gradient optimization approach with a tunable learning rate.
- **Data visualization**: Generates several plots to visualize results over iterations, including cache hit rates, queue lengths, content selection frequencies, and task requests vs. hits.

## Figures Generated

Here are the figures generated from running the simulation:

1. **Hit Rate Over Iterations**: Shows the trend of cache hit rate over time.
   ![Hit Rate](file-ed6nspZvrXdpMP5JcLcLcCwM)

2. **Queue Length Over Iterations**: Displays how the length of the server queue fluctuates over time.
   ![Queue Length](file-oQ9PgvsvqcRgynUrKayv2vt3)

3. **Content Selection Frequencies**: Indicates how often each type of content was selected for caching during the simulation.
   ![Content Selection Frequencies](file-kA81jFL95EphbBvrXY5Mt28e)

4. **Total Requests vs. Total Hits**: Compares the total number of requests made for each content type against the number of successful cache hits.
   ![Requests vs Hits](file-G02VowUctGLyFsjPgruiBOuV)

5. **Average Queue Time Per Task Type**: Shows the average time spent by each task type in the server queue.
   ![Average Queue Time](file-TlJJLoQmxr8Hdnixuhf27xFP)

## How to Run the Project

### Requirements

This project is built using Python and requires the following libraries:
- `numpy`
- `matplotlib`
- `pytest` (for running tests)

You can install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Running the Simulation
To run the simulation, execute the following command:

```bash
python main.py
```
This will generate the figures listed above and save them to the working directory.

## Running the Tests
Unit tests are included to verify the behavior of the cache system. To run the tests, use the
```bash
pytest
```

## Docker Support
You can also run the project inside a Docker container. Make sure Docker is installed on your machine. Build and run the container using the following commands:

Build the Docker Image:
```bash
docker build -t cache_simulation .
```
Run the Container:
```bash
docker run -it cache_simulation
```

# Expanding the Project
Students in the ECE 627 course are encouraged to build upon this project by:

- Adding optimizations for queue management, such as minimizing the wait time for queued tasks.
- Improving latency metrics.
- Introducing more realistic task request patterns and varying computational complexities for tasks.

This code serves as a foundation for building more sophisticated and complex simulation environments tailored for specific cache and queue management challenges.


# Contributions
This project was developed by Yinxuan Wu and Ning Wang as a reference for students in the Network Science Lab at UMass Amherst.



