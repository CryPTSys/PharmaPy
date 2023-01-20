@echo OFF
echo -------------------------------------------------- 
echo Welcome to the PharmaPy installation wizard.
echo -------------------------------------------------- 
set /p env_name=Enter environment name: 
echo ------------------------------
echo Creating pandas environment...
echo ------------------------------
call conda create -n %env_name% python=3.9 --file requirements.txt -c conda-forge
call conda activate %env_name%
echo ----------------------
echo Installing PharmaPy...
echo ----------------------
call python setup.py develop
echo Done!
