@echo off
echo.
echo HESA Data Visualization Web Application
echo =====================================
echo.
echo This script will start the Django development server.
echo The web interface will be available at http://localhost:8000
echo.

REM Set the PYTHONPATH to include the current directory
set PYTHONPATH=%~dp0;%PYTHONPATH%

REM Create necessary directories if they don't exist
if not exist "data\figures" mkdir data\figures
if not exist "data\reports" mkdir data\reports
if not exist "data\raw_files" mkdir data\raw_files
if not exist "data\cleaned_files" mkdir data\cleaned_files
if not exist "logs" mkdir logs

REM Run the Django development server
python manage.py runserver

echo.
echo Web application server has stopped.
echo.

pause 