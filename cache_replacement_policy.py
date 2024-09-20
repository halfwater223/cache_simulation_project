# cache_replacement_policy.py

import numpy as np

# Base class for cache replacement policies
class CacheReplacementPolicy:
    def __init__(self, num_types, cache_size):
        self.num_types = num_types
        self.cache_size = cache_size
        self.cache = []

    def is_hit(self, task):
        raise NotImplementedError

    def update_cache(self, task):
        raise NotImplementedError

# LFU Cache Replacement Policy
class LFUCache(CacheReplacementPolicy):
    def __init__(self, num_types, cache_size):
        super().__init__(num_types, cache_size)
        self.request_counts = np.zeros(num_types)
        self.cache = np.random.choice(num_types, cache_size, replace=False)

    def is_hit(self, task):
        return 1 if task in self.cache else 0

    def update_cache(self, task):
        self.request_counts[task] += 1
        # Update cache using LFU strategy
        top_items = np.argsort(-self.request_counts)[:self.cache_size]
        self.cache = top_items

# Q-Learning Cache Replacement Policy
class QLearningCache(CacheReplacementPolicy):
    def __init__(self, num_types, cache_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(num_types, cache_size)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.cache = np.random.choice(num_types, cache_size, replace=False)
        self.num_states = num_types * 2  # task * 2 + hit_or_miss
        self.num_actions = cache_size + 1  # Actions 0 to cache_size -1, plus 'do nothing' action
        self.Q_table = np.zeros((self.num_states, self.num_actions))  # States x Actions

    def is_hit(self, task):
        return 1 if task in self.cache else 0

    def update_cache(self, task):
        hit = self.is_hit(task)
        state = task * 2 + hit
        if hit:
            # Available action is only 'do nothing' (action index cache_size)
            action = self.cache_size
            reward = 1  # Cache hit
            # No cache update needed
        else:
            # Available actions: replace cache positions 0 to cache_size -1
            # Select action using epsilon-greedy
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.cache_size)
            else:
                # Since only actions 0 to cache_size -1 are available, select best action
                action = np.argmax(self.Q_table[state, :self.cache_size])
            # Replace the item at position 'action' with the requested task
            self.cache[action] = task
            reward = 0  # Cache miss
        # Next state
        next_hit = self.is_hit(task)
        next_state = task * 2 + next_hit
        if next_hit:
            max_Q_next = self.Q_table[next_state, self.cache_size]  # Only 'do nothing' action
        else:
            max_Q_next = np.max(self.Q_table[next_state, :self.cache_size])  # Actions 0 to cache_size -1
        # Update Q-table
        td_target = reward + self.gamma * max_Q_next
        self.Q_table[state, action] += self.alpha * (td_target - self.Q_table[state, action])
