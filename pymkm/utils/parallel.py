import os

def optimal_worker_count(workload) -> int:
    """
    Determine an optimal number of workers based on workload size and CPU availability.

    Parameters:
    - workload: iterable or int
        The size of the task set (e.g., number of elements to compute).

    Returns:
    - int: Number of worker processes to use (minimum 1).
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