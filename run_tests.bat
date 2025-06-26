@echo off
:: Run pyMKM tests with coverage in editable mode using Anaconda Python

:: Set path to Anaconda Python executable
set PYTHON_EXEC=C:\ProgramData\Anaconda3\python.exe

echo [1/3] Installing pyMKM in editable mode with dev dependencies...
%PYTHON_EXEC% -m pip install -e .[dev]

echo [2/3] Setting PYTHONPATH and launching tests with coverage...
set PYTHONPATH=.
%PYTHON_EXEC% -m pytest --cov=pymkm --cov-report=term-missing tests

echo [3/3] Done. Press any key to close.
pause
