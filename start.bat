@echo off
setlocal EnableDelayedExpansion
cd %~dp0

.\runtime\python.exe -m  streamlit run app.py

pause