
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

def test_optimal_worker_count_user_requested_valid():
    result = optimal_worker_count(10, user_requested=2)
    assert result == 2

def test_optimal_worker_count_user_requested_exceeds_cpu(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 4)  # fissa a 4 CPU
    with pytest.warns(UserWarning, match=r"only 3 allowed"):
        result = optimal_worker_count(list(range(10)), user_requested=10)
        assert result == 3  # CPU - 1

def test_optimal_worker_count_empty_with_user_requested():
    result = optimal_worker_count([], user_requested=2)
    assert result == 1  # workload vuoto → set to 1
