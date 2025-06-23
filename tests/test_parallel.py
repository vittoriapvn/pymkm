
import pytest
import os
from pymkm.utils.parallel import optimal_worker_count

@pytest.mark.parametrize("workload, expected_exact", [
    ([], 1),
    ([1], 1),
    (list(range(2)), 2),
])
def test_optimal_worker_count_small_cases(workload, expected_exact):
    result = optimal_worker_count(workload)
    assert isinstance(result, int)
    assert result == expected_exact

def test_optimal_worker_count_large_workload():
    workload = list(range(1000))
    result = optimal_worker_count(workload)
    num_cores = os.cpu_count() or 1
    assert isinstance(result, int)
    assert 1 <= result <= num_cores
    assert result == num_cores - 1

def test_optimal_worker_count_integer_input():
    result = optimal_worker_count(10)
    assert isinstance(result, int)
    assert 1 <= result <= (os.cpu_count() or 1)

def test_optimal_worker_count_zero_input():
    result = optimal_worker_count(0)
    assert result == 1
