@echo off
echo.
echo HESA LLM Visualization - Integration Tests
echo ==========================================
echo.

REM Set the PYTHONPATH to include the project root
set PYTHONPATH=%~dp0..\..\;%PYTHONPATH%

REM Check if pytest is installed
python -c "import pytest" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pytest is not installed.
    echo Please install pytest with: pip install pytest
    goto :end
)

echo Running integration tests...
echo.

REM Run pytest with appropriate options
python -m pytest -v %~dp0..\integration\

echo.
echo Integration tests completed!
echo.

:end
pause 