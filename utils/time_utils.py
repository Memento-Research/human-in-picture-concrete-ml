import time
from functools import wraps


def log_time(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__} execution time:{total_time:.4f} seconds')
        return result

    return timeit_wrapper
