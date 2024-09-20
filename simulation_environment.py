# simulation_environment.py

import numpy as np
import matplotlib.pyplot as plt
import os

class SimulationEnvironment:
    def __init__(self, num_types=20, cache_size=5, iterations=5000, alpha=0.08, max_queue_length=10, user_request_probability=0.9, cache_policies=None):
        self.num_types = num_types
        self.cache_size = cache_size
        self.iterations = iterations
        self.alpha = alpha
        self.max_queue_length = max_queue_length
        self.user_request_probability = user_request_probability
        self.computation_times = np.random.randint(2, 20, size=num_types)
        self.queue = []
        self.task_probabilities = self.generate_task_probabilities()
        if cache_policies is None:
            cache_policies = []
        self.cache_policies = cache_policies  # List of cache policies to compare
        # Performance metrics for each policy
        self.performance_metrics = {
            policy.__class__.__name__: {'hit_rates': [], 'queue_lengths': [], 'total_waiting_time': np.zeros(num_types),
                                        'queue_count': np.zeros(num_types), 'total_requests': np.zeros(num_types),
                                        'total_hits': np.zeros(num_types),
                                        'content_frequencies': np.zeros(num_types)} for policy in self.cache_policies}
        self.queues = {policy.__class__.__name__: [] for policy in self.cache_policies}

    def generate_task_probabilities(self):
        task_probabilities = np.zeros(self.num_types)
        num_high_prob_task = 10
        high_probabilities = np.random.uniform(0.02, 0.10, size=num_high_prob_task)
        task_probabilities[:num_high_prob_task] = high_probabilities
        task_probabilities[num_high_prob_task:] = 0.01
        task_probabilities /= task_probabilities.sum()  # Normalize
        return task_probabilities

    def simulate(self):
        for i in range(self.iterations):
            # Process queues for each policy
            self.process_queues()
            # Generate user request
            if np.random.rand() < self.user_request_probability:
                requested_task = np.random.choice(self.num_types, p=self.task_probabilities)
                for policy in self.cache_policies:
                    policy_name = policy.__class__.__name__
                    metrics = self.performance_metrics[policy_name]
                    metrics['total_requests'][requested_task] += 1
                    hit = policy.is_hit(requested_task)
                    if hit:
                        metrics['total_hits'][requested_task] += 1
                    else:
                        queue = self.queues[policy_name]
                        if len(queue) < self.max_queue_length:
                            queue.append((requested_task, self.computation_times[requested_task]))
                    metrics['hit_rates'].append(hit)
                    # Update cache
                    policy.update_cache(requested_task)
                    # Update content frequencies
                    for item in policy.cache:
                        metrics['content_frequencies'][item] += 1

    def process_queues(self):
        for policy in self.cache_policies:
            policy_name = policy.__class__.__name__
            queue = self.queues[policy_name]
            new_queue = []
            for task, remaining_time in queue:
                remaining_time -= 1
                self.performance_metrics[policy_name]['total_waiting_time'][task] += 1
                if remaining_time > 0:
                    new_queue.append((task, remaining_time))
            self.queues[policy_name] = new_queue
            # Update queue lengths
            self.performance_metrics[policy_name]['queue_lengths'].append(len(new_queue))

    def plot_results(self):
        # Ensure the 'results' directory exists
        os.makedirs('results', exist_ok=True)
        self.plot_hit_rate()
        self.plot_queue_lengths()
        self.plot_content_frequencies()
        self.plot_total_requests_vs_hits()
        self.plot_hit_rates_per_policy()
        self.plot_avg_queue_time()

    def plot_hit_rate(self):
        plt.figure(figsize=(12, 6))
        for policy in self.cache_policies:
            policy_name = policy.__class__.__name__
            hit_rates = self.performance_metrics[policy_name]['hit_rates']
            smoothed_hit_rates = np.convolve(hit_rates, np.ones(100) / 100, mode='valid')
            plt.plot(smoothed_hit_rates, label=policy_name)
        plt.title('Hit Rate Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Hit Rate')
        plt.legend()
        plt.grid(True)
        # Save the figure
        plt.savefig('results/hit_rate_over_iterations.png')
        plt.show()

    def plot_queue_lengths(self, num_samples=50):
        plt.figure(figsize=(12, 6))
        # Define different line styles and markers for distinction
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'D', 'x', '^']
        for i, policy in enumerate(self.cache_policies):
            policy_name = policy.__class__.__name__
            queue_lengths = self.performance_metrics[policy_name]['queue_lengths']
            if len(queue_lengths) == 0:
                print(f"No data to plot for queue lengths for {policy_name}.")
                continue
            # Determine the indices for sampling
            total_points = len(queue_lengths)
            if total_points > num_samples:
                indices = np.linspace(0, total_points - 1, num_samples, dtype=int)
                sampled_queue_lengths = [queue_lengths[i] for i in indices]
            else:
                sampled_queue_lengths = queue_lengths  # Use all points if fewer than num_samples
            # Use different line styles and markers for each curve
            line_style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            plt.plot(sampled_queue_lengths, label=policy_name, linestyle=line_style, marker=marker)
        plt.title('Queue Length Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Queue Length')
        plt.legend()
        plt.grid(True)
        # Save the figure
        plt.savefig('results/queue_length_over_iterations.png')
        plt.show()

    def plot_content_frequencies(self):
        plt.figure(figsize=(12, 6))
        x = np.arange(self.num_types)
        total_width = 0.8
        width = total_width / len(self.cache_policies)
        for i, policy in enumerate(self.cache_policies):
            policy_name = policy.__class__.__name__
            content_frequencies = self.performance_metrics[policy_name]['content_frequencies']
            plt.bar(x + i * width, content_frequencies, width=width, label=policy_name)
        plt.title('Content Selection Frequencies')
        plt.xlabel('Content Type')
        plt.ylabel('Selection Frequency')
        plt.legend()
        plt.grid(True)
        # Save the figure
        plt.savefig('results/content_selection_frequencies.png')
        plt.show()

    def plot_total_requests_vs_hits(self):
        plt.figure(figsize=(12, 6))
        x = np.arange(self.num_types)
        total_width = 0.8
        width = total_width / (len(self.cache_policies) + 1)  # +1 for total requests
        # Compute total requests once
        total_requests = np.zeros(self.num_types)
        for requested_task in range(self.num_types):
            total_requests[requested_task] = np.sum([
                self.performance_metrics[policy.__class__.__name__]['total_requests'][requested_task]
                for policy in self.cache_policies
            ]) / len(self.cache_policies)  # Average across policies
        # Plot total requests once
        plt.bar(x - total_width / 2 + width / 2, total_requests, width=width, label='Total Requests', color='grey')
        # Plot total hits for each policy
        for i, policy in enumerate(self.cache_policies):
            policy_name = policy.__class__.__name__
            total_hits = self.performance_metrics[policy_name]['total_hits']
            plt.bar(x - total_width / 2 + (i + 1.5) * width, total_hits, width=width, label=f'{policy_name} Hits')
        plt.title('Total Requests and Total Hits per Policy')
        plt.xlabel('Content Type')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        # Save the figure
        plt.savefig('results/total_requests_vs_hits.png')
        plt.show()

    def plot_hit_rates_per_policy(self):
        plt.figure(figsize=(12, 6))
        x = np.arange(self.num_types)
        # Use total_requests from the first policy (since they are the same)
        first_policy_name = self.cache_policies[0].__class__.__name__
        total_requests = self.performance_metrics[first_policy_name]['total_requests']
        # Plot hit rates for each policy
        for policy in self.cache_policies:
            policy_name = policy.__class__.__name__
            total_hits = self.performance_metrics[policy_name]['total_hits']
            hit_rates = np.divide(
                total_hits, total_requests, out=np.zeros_like(total_hits), where=total_requests != 0)
            plt.plot(x, hit_rates, marker='o', label=f'{policy_name} Hit Rate')
        plt.title('Hit Rates per Content Type')
        plt.xlabel('Content Type')
        plt.ylabel('Hit Rate')
        plt.legend()
        # Save the figure
        plt.savefig('results/hit_rates_per_content_type.png')
        plt.show()

    def plot_avg_queue_time(self):
        plt.figure(figsize=(12, 6))
        x = np.arange(self.num_types)
        total_width = 0.8
        width = total_width / len(self.cache_policies)
        for i, policy in enumerate(self.cache_policies):
            policy_name = policy.__class__.__name__
            total_waiting_time = self.performance_metrics[policy_name]['total_waiting_time']
            total_requests = self.performance_metrics[policy_name]['total_requests']
            average_waiting_time = np.divide(
                total_waiting_time, total_requests, out=np.zeros_like(total_waiting_time), where=total_requests != 0)
            plt.bar(x + i * width, average_waiting_time, width=width, label=policy_name)
        plt.title('Average Queue Time Per Task Type')
        plt.xlabel('Task Type')
        plt.ylabel('Average Time in Queue')
        plt.legend()
        plt.grid(True)
        # Save the figure
        plt.savefig('results/average_queue_time_per_task_type.png')
        plt.show()