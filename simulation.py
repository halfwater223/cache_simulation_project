import numpy as np
import matplotlib.pyplot as plt
from cache_system import CacheSystem

class SimulationEnvironment:
    def __init__(self, num_types=20, cache_size=5, iterations=5000, alpha=0.08, max_queue_length=10, user_request_probability=0.9):
        self.num_types = num_types
        self.cache_size = cache_size
        self.iterations = iterations
        self.alpha = alpha
        self.max_queue_length = max_queue_length
        self.user_request_probability = user_request_probability
        self.computation_times = np.random.randint(2, 20, size=num_types)
        self.queue = []
        self.theta = np.random.randn(num_types, num_types * 2) * 0.01
        self.total_waiting_time = np.zeros(num_types)
        self.queue_count = np.zeros(num_types)
        self.hit_rates = []
        self.queue_lengths = []
        self.content_frequencies = np.zeros(num_types)
        self.total_requests = np.zeros(num_types)
        self.total_hits = np.zeros(num_types)
        self.cache_system = CacheSystem(num_types, cache_size)

    def simulate(self):
        for i in range(self.iterations):
            state = np.zeros(self.num_types * 2)
            state[self.cache_system.cache] = 1  # Update state with cache info
            if len(self.queue) > 0:
                queue_state = [task for task, remaining_time in self.queue]
                state[self.num_types + np.array(queue_state)] = 1   # Update state with queue info

                # Process the queue
                self.process_queue()
                # Append the current queue length after processing
                self.queue_lengths.append(len(self.queue))

            probabilities = self.softmax(self.theta, state)
            selected_content = np.random.choice(self.num_types, self.cache_size, p=probabilities, replace=False)
            self.cache_system.update_cache(selected_content)

            if np.random.rand() < self.user_request_probability:
                requested_task = np.random.choice(self.num_types, p=self.generate_task_probabilities())
                self.total_requests[requested_task] += 1
                hit = self.cache_system.is_hit(requested_task)
                if hit:
                    self.total_hits[requested_task] += 1
                else:
                    if len(self.queue) < self.max_queue_length:
                        self.queue.append((requested_task, self.computation_times[requested_task]))
                self.hit_rates.append(hit)

            hit_mean = np.mean(self.hit_rates) if len(self.hit_rates) > 0 else 0
            gradient = self.compute_policy_gradient(hit, hit_mean, selected_content, probabilities, state)
            self.theta += self.alpha * gradient

            for item in selected_content:
                self.content_frequencies[item] += 1

    def generate_task_probabilities(self):
        task_probabilities = np.zeros(self.num_types)
        task_probabilities[[0, 1, 2, 3, 4]] = 0.15  # Higher probability tasks
        task_probabilities[5:] = 0.01  # Lower probability tasks
        task_probabilities /= task_probabilities.sum()  # Normalize
        return task_probabilities

    def softmax(self, theta, state):
        scores = np.dot(theta, state)
        exp_scores = np.exp(scores - np.max(scores))  # Stability improvement
        return exp_scores / exp_scores.sum()

    def compute_policy_gradient(self, hit, hit_mean, selected_content, probabilities, state):
        selected_mask = np.zeros(self.num_types)
        selected_mask[selected_content] = 1  # One-hot encoding of the selected content
        reward = self.compute_reward(hit, hit_mean)
        gradient = reward * np.outer((selected_mask - probabilities), state)
        return gradient

    def compute_reward(self, hit, hit_mean):
        return (hit - hit_mean)

    def process_queue(self):
        new_queue = []
        for task, remaining_time in self.queue:
            if task in self.cache_system.cache:
                continue  # Cached tasks are processed immediately
            delay = np.random.randint(0, 2)
            remaining_time += delay
            self.total_waiting_time[task] += 1
            if remaining_time > 1:
                new_queue.append((task, remaining_time - 1))
        self.queue = new_queue

    def plot_results(self):
        self.plot_hit_rate()
        self.plot_queue_lengths()
        self.plot_content_frequencies()
        self.plot_total_requests_vs_hits()
        self.plot_avg_queue_time()

    def plot_hit_rate(self):
        plt.figure(figsize=(12, 6))
        plt.plot(np.convolve(self.hit_rates, np.ones(100) / 100, mode='valid'))  # Smoothed hit rate
        plt.title('Hit Rate Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Hit Rate')
        plt.grid(True)
        plt.show()

    def plot_queue_lengths(self):
        if len(self.queue_lengths) == 0:
            print("No data to plot for queue lengths.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.queue_lengths, label='Queue Length')
        plt.title('Queue Length Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Queue Length')
        plt.grid(True)
        plt.show()

    def plot_content_frequencies(self):
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.num_types), self.content_frequencies)
        plt.title('Content Selection Frequencies')
        plt.xlabel('Content Type')
        plt.ylabel('Selection Frequency')
        plt.grid(True)
        plt.show()

    def plot_total_requests_vs_hits(self):
        plt.figure(figsize=(12, 6))
        x = np.arange(self.num_types)
        plt.bar(x - 0.2, self.total_requests, width=0.4, label='Total Requests')
        plt.bar(x + 0.2, self.total_hits, width=0.4, label='Total Hits')
        plt.title('Total Requests vs Total Hits')
        plt.xlabel('Content Type')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_avg_queue_time(self):
        average_waiting_time = np.divide(self.total_waiting_time, self.total_requests, out=np.zeros_like(self.total_waiting_time),
                                         where=self.total_requests != 0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.num_types), average_waiting_time)
        plt.title('Average Queue Time Per Task Type')
        plt.xlabel('Task Type')
        plt.ylabel('Average Time in Queue')
        plt.grid(True)
        plt.show()
