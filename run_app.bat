@echo off
echo Starting Streamlit application...

:: Activate the virtual environment
call .\venv\Scripts\activate.bat

:: Run the streamlit app
echo Running: streamlit run app.py
streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to run the Streamlit application.
    echo Please check for any errors in the output above.
)

pause
