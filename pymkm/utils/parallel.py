"""
Parallelization utility for determining optimal worker count.

This module provides a helper function:

- :func:`optimal_worker_count`:
  Computes the optimal number of worker processes to use for parallel tasks,
  based on workload size and available CPU cores.

It ensures at least one worker is used, avoids oversubscription, and supports
both iterable and integer inputs for task sizing.

Intended for internal use in MKTable computation and other parallel workflows.
"""

import os

def optimal_worker_count(workload) -> int:
    """
    Determine the optimal number of worker processes based on workload size and CPU availability.

    If the workload size is smaller than the number of CPU cores, the number of workers
    is set equal to the workload size. Otherwise, one core is reserved to avoid oversubscription.

    :param workload: The total number of tasks to process. Can be an iterable or an integer.
    :type workload: iterable or int

    :returns: Optimal number of worker processes (always at least 1).
    :rtype: int
    """
    # Get the number of CPU cores available, fallback to 1 if not detected
    num_cores = os.cpu_count() or 1

    # Determine size of workload
    size = len(workload) if hasattr(workload, '__len__') else int(workload)

    # Ensure at least one worker, even if workload is empty
    if size == 0:
        return 1

    # Use all elements if fewer than available cores; else, reserve one core
    return min(size, max(1, num_cores - 1))