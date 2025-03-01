@echo off
echo.
echo HESA Natural Language Query Processor
echo ==================================
echo.
echo This script will process a natural language query about HESA data.
echo.

REM Set the PYTHONPATH to include the current directory
set PYTHONPATH=%~dp0;%PYTHONPATH%

REM Sample query about enrollment trends
set QUERY="Show me enrollment trends over the last 5 years"

REM Run the query processor with the sample query
python -m core.query_processing.query_processor --query %QUERY% --data-dir data/cleaned_files

echo.
echo Query processing completed!
echo.

pause 