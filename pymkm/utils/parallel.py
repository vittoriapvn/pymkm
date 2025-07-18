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
import warnings

def optimal_worker_count(workload, user_requested: int = None) -> int:
    """
    Determine the optimal number of worker processes based on workload size, CPU availability,
    and optionally user input.

    If the workload size is smaller than the number of CPU cores, the number of workers
    is set equal to the workload size. Otherwise, one core is reserved to avoid oversubscription.

    If a user-requested value is provided, it will be respected unless it exceeds CPU-1.

    :param workload: The total number of tasks to process. Can be an iterable or an integer.
    :type workload: iterable or int
    :param user_requested: Optional user-defined number of workers.
    :type user_requested: int or None

    :returns: Optimal number of worker processes (always at least 1).
    :rtype: int
    """
    # Get the number of CPU cores available, fallback to 1 if not detected
    num_cores = os.cpu_count() or 1
    max_workers = max(1, num_cores - 1)

    # Determine size of workload
    size = len(workload) if hasattr(workload, '__len__') else int(workload)

    # Ensure at least one worker, even if workload is empty
    if size == 0:
        return 1

    if user_requested is not None:
        capped = max(1, min(user_requested, max_workers))
        if user_requested > max_workers:
            warnings.warn(
                f"Requested {user_requested} workers, but only {max_workers} allowed based on CPU count. "
                f"Using {capped} workers instead."
            )
        return capped

    return min(size, max_workers)
