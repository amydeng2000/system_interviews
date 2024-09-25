from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import time

NUM_CPU_CORES = 8
numbers = [i * 100000 for i in range(100)]


def cpu_bound_func(n):
    """
    A slow function due to complex computations
    """
    return sum(i * i for i in range(n))


def no_parallelize():
    outputs = []
    for n in tqdm(numbers):
        outputs.append(cpu_bound_func(n))
    return outputs


def process_pool_parallelized():
    with ProcessPoolExecutor(max_workers=NUM_CPU_CORES) as executor:
        outputs = list(tqdm(executor.map(cpu_bound_func, numbers), total=len(numbers)))
    return outputs


def thread_pool_parallelized():
    with ThreadPoolExecutor(max_workers=NUM_CPU_CORES) as executor:
        outputs = list(tqdm(executor.map(cpu_bound_func, numbers), total=len(numbers)))
    return outputs
