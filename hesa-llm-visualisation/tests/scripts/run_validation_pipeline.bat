@echo off
echo.
echo HESA Data Validation Pipeline
echo ============================
echo.
echo This script will run the validation pipeline on all files in the data/raw_files directory.
echo It can also generate sample data if none exists.
echo.

REM Set the PYTHONPATH to include the project root
set PYTHONPATH=%~dp0..\..\;%PYTHONPATH%

REM Run the validation pipeline with sample data generation
python -m core.data_processing.validation.validation_pipeline --generate-samples

echo.
echo Validation pipeline completed!
echo.

pause 