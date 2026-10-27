@echo off
setlocal
cd /d "%~dp0"
python -X utf8 -m streamlit run app.py
endlocal