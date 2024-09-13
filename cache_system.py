import numpy as np

class CacheSystem:
    def __init__(self, num_types, cache_size):
        self.cache_size = cache_size
        self.num_types = num_types
        self.cache = np.random.choice(num_types, cache_size, replace=False)

    def update_cache(self, selected_content):
        for item in selected_content:
            if item not in self.cache:
                self.cache[np.random.randint(self.cache_size)] = item

    def is_hit(self, task):
        return 1 if task in self.cache else 0
