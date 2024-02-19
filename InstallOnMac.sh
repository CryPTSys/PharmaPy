#!/bin/bash
echo -------------------------------------------------- 
echo Welcome to the PharmaPy installation wizard.
echo -------------------------------------------------- 
echo Enter environment name:
read env_name 
echo ------------------------------
echo Creating pandas environment...
echo ------------------------------
conda create -n $env_name python=3.9 --file requirements.txt -c conda-forge
conda activate $env_name
echo ----------------------
echo Installing PharmaPy...
echo ----------------------
python setup.py develop
