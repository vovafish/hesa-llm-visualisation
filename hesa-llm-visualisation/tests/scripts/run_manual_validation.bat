@echo off
echo.
echo HESA Manual Data Validation
echo ==========================
echo.
echo This script will run the manual data validator on a selected file.
echo.

REM Set the PYTHONPATH to include the project root
set PYTHONPATH=%~dp0..\..\;%PYTHONPATH%

REM Run the manual validator on the enrollment.csv file
python -m core.data_processing.validation.manual_validator --file ../../data/raw_files/enrollment.csv --verbose

echo.
echo Manual validation completed!
echo.

pause 