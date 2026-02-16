import threading
import time
from concurrent.futures import ThreadPoolExecutor

def print_numbers():
    for i in range(5):
        print(f"Number: {i}")
        time.sleep(1)
    return ['a', 'b', 'c', 'd', 'e']

def print_terrains():
    terrains = ['Mountain', 'River', 'Forest', 'Desert', 'Plains']
    for terrain in terrains:
        print(f"Terrain: {terrain}")
        time.sleep(1.5)
    return terrains

def print_sample():
    for i in range(3):
        print(f"Sample: {i}")
    return [1, 3, 3]
        # time.sleep(2)
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as executor:
        # print(executor.submit(print_numbers))
        # print(executor.submit(print_terrains))
        # print(executor.submit(print_numbers))
        future1 = executor.submit(print_numbers)
        future2 = executor.submit(print_terrains)
        future3 = executor.submit(print_sample)
        future1.result()
        future2.result()
        future3.result()
        # result1 = future1.result()
        # result2 = future2.result()
        # result3 = future3.result()
        # print(result1+ result2 + result3)

import time
import threading

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        """
        :param capacity: The maximum number of tokens the bucket can hold (C).
        :param refill_rate: The rate at which tokens are added, per second (R).
        """
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)  # Start with a full bucket
        self.last_check = time.time()
        self.lock = threading.Lock() # For thread-safe operations

    def get_tokens_to_add(self):
        """Calculate how many tokens should be added based on elapsed time."""
        now = time.time()
        # Calculate time elapsed since last check in seconds
        time_elapsed = now - self.last_check
        # Tokens to add = elapsed time * refill rate
        tokens_to_add = time_elapsed * self.refill_rate
        self.last_check = now
        return tokens_to_add

    def consume(self, tokens_requested):
        """
        Consume tokens if available.
        :param tokens_requested: The number of tokens needed for the request.
        :return: True if the request is allowed, False otherwise.
        """
        with self.lock:
            # Refill tokens first
            tokens_to_add = self.get_tokens_to_add()
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)

            # Check if enough tokens are available
            if tokens_requested <= self.tokens:
                self.tokens -= tokens_requested
                return True
            else:
                return False

# Example Usage:
# A bucket with a capacity of 10 tokens that refills at 1 token/second
bucket = TokenBucket(capacity=10, refill_rate=1)

for i in range(15):
    time.sleep(0.2) # Simulate requests arriving every 0.2 seconds
    if bucket.consume(1):
        print(f"Request {i+1}: ALLOWED")
    else:
        print(f"Request {i+1}: DENIED (not enough tokens)")

import time
from functools import wraps


class RateLimitExceeded(Exception):
    """Custom exception raised when a rate limit is exceeded."""
    pass


def throttle(interval):
    """
    A decorator that throttles a function to be called at most once every 'interval' seconds.
    """
    last_run = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal last_run
        now = time.time()
        if now - last_run < interval:
            # Optionally, sleep here instead of raising an exception to wait
            # time.sleep(interval - (now - last_run))
            raise RateLimitExceeded(f"Call again after {last_run + interval - now:.2f} seconds")

        last_run = now
        return func(*args, **kwargs)

    return wrapper


# Example Usage:
@throttle(interval=2)  # Limit to one call every 2 seconds
def fetch_data(item):
    print(f"Fetching data for {item} at {time.strftime('%H:%M:%S')}")
    # Simulate some work
    time.sleep(0.1)
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        max_value = float('-inf')
        left = 0
        total = 0
        for i in range(len(nums)):
            total += nums[i]
            if i >= k:
                max_value = max(max_value, total)
                total -= nums[left]
                left += 1
        return max_value/k


# Calling the function in a loop to test throttling
for i in range(5):
    try:
        fetch_data(i)
    except RateLimitExceeded as e:
        print(e)
    time.sleep(0.5)
