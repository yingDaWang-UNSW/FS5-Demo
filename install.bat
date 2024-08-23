@echo off
REM Define Python executable path based on local installation
set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
set PIP_EXE=%LOCALAPPDATA%\Programs\Python\Python312\Scripts\pip.exe

REM Specify the directory where your packages are stored
set PACKAGE_DIR=%~dp0\pkgs

REM Check for Python and print the version
%PYTHON_EXE% --version 2>NUL
if %ERRORLEVEL% neq 0 goto installPython
echo Python is already installed.
goto installPackages

:installPython
echo Python is not installed.
echo Using local Python installer...
start /wait .\pkgs\python-3.12.5-amd64.exe /quiet InstallAllUsers=0 PrependPath=1
if %ERRORLEVEL% neq 0 (
    echo Failed to install Python.
    goto end
)
echo Python installed successfully.

:installPackages
echo Installing required Python packages from local directory...
%PIP_EXE% install --no-index --find-links="%PACKAGE_DIR%" warp-lang pyglet trimesh meshio Rtree gooey matplotlib
if %ERRORLEVEL% neq 0 (
    echo Failed to install packages.
    goto end
)
echo Packages installed successfully.

:end
pause
