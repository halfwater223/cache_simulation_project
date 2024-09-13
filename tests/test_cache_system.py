import pytest
from cache_system import CacheSystem

def test_cache_hit():
    cache_system = CacheSystem(num_types=20, cache_size=5)
    cache_system.cache = [0, 1, 2, 3, 4]
    assert cache_system.is_hit(3) == 1
    assert cache_system.is_hit(10) == 0

def test_update_cache():
    cache_system = CacheSystem(num_types=20, cache_size=5)
    old_cache = cache_system.cache.copy()
    cache_system.update_cache([15])
    assert any(item not in old_cache for item in cache_system.cache)
