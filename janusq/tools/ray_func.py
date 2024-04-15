from functools import reduce
import inspect
# from collections import Iterable
from collections import defaultdict
from concurrent.futures._base import Future

import ray
from sklearn.utils import shuffle
from janusq.tools import pb

import multiprocessing
 



DEFAUKT_N_PROCESS = multiprocessing.cpu_count() - 3

def map(func, data, multi_process = False, n_process = None, show_progress = False, **kwargs): 

    if multi_process:
        if n_process is None:
            n_process = DEFAUKT_N_PROCESS
        
        @ray.remote
        def _map(data, **kwargs):
            return map(func, data, **kwargs)
        
        batch_size = len(data) // n_process
        if batch_size < 10:
            batch_size = 10
        
        futures = []
        for sub_dataset in batch(data, batch_size=batch_size):
            futures.append(_map.remote(sub_dataset, **kwargs))
        futures = wait(futures, show_progress=show_progress)
        return reduce(lambda a, b: a + b, futures)
    else:
        results = []
        for elm in pb(data, show_progress):
            results.append(func(elm, **kwargs))
        return results

def batch(X, *args, batch_size = 100, should_shuffle = False):
    if len(args) >= 1:
        # X = np.array(X)
        if should_shuffle:
            args = shuffle(X, *args)
        for start in range(0, len(X), batch_size):
            yield [arg[start: start+batch_size] for arg in args]
    else:
        # X = np.array(X)
        if should_shuffle:
            X = shuffle(X)
        for start in range(0, len(X), batch_size):
            yield X[start: start+batch_size]

# 是不是需要远程执行的函数
def is_ray_func(func):
    for name, f in inspect.getmembers(func, lambda f: hasattr(f, '_remote')):
        return True
    return False

def is_ray_future(obj):
    return isinstance(obj, ray._raylet.ObjectRef)

def wait(future, show_progress = False):
    if isinstance(future, (list, set, tuple)):
        futures = future
        results = []
        for future in pb(futures, show_progress):
            results.append(wait(future))
        return results
    elif is_ray_future(future):
        return ray.get(future)
    elif isinstance(future, Future):
        return future.result()
    elif isinstance(future, (dict, defaultdict)):
        return {
            key: wait(item)
            for key, item in future.items()
        }
    else:
        return future

    