@echo off
setlocal
set "PROJECT_DIR=%~dp0"
if defined PYTHONPATH (
    set "PYTHONPATH=%PROJECT_DIR%src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%PROJECT_DIR%src"
)
if exist "%PROJECT_DIR%.venv\Scripts\python.exe" goto venv
where py >nul 2>nul
if errorlevel 1 goto python
py -3 -m urdf_dh.gui %*
exit /b %errorlevel%

:venv
"%PROJECT_DIR%.venv\Scripts\python.exe" -m urdf_dh.gui %*
exit /b %errorlevel%

:python
python -m urdf_dh.gui %*
exit /b %errorlevel%
