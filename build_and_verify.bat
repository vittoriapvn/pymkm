@echo off
:: Run sync_dependencies_grep.py and build pyMKM distributions

:: Set path to Anaconda Python executable
set PYTHON_EXEC=C:\ProgramData\Anaconda3\python.exe

cd /d %~dp0

echo [1/3] Syncing dependencies from source code...
%PYTHON_EXEC% sync_dependencies_grep.py
if errorlevel 1 (
    echo Failed to sync dependencies. Aborting.
    pause
    exit /b 1
)

echo [2/3] Building source and wheel distributions...
%PYTHON_EXEC% -m build
if errorlevel 1 (
    echo Build failed. Aborting.
    pause
    exit /b 1
)

echo [3/3] Listing contents of generated packages...
for %%F in (dist\*.tar.gz) do (
    echo --- Contents of %%F (source) ---
    tar -tzf %%F
)

for %%F in (dist\*.whl) do (
    echo --- Contents of %%F (wheel) ---
    unzip -l %%F
)

echo Done. Press any key to close.
pause