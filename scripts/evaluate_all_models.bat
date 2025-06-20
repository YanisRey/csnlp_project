@echo off
REM ===== Run evaluations in the correct virtual environment =====

:: Activate the virtual environment
call .\venv\Scripts\activate

:: Define the scripts to run (relative paths)
set script1=..\evaluate\avg_cos_per_misspelling.py
set script2=..\evaluate\human_scored_pairs.py

:: Counter for script tracking
set count=0

:: Run first script
set /a count+=1
echo.
echo 📊 Running evaluation script %count%/2: %script1%
python %script1%
if %errorlevel% neq 0 (
    echo ❌ Error running %script1%
    goto :eof
)
echo ✅ Finished: %script1%

:: Run second script
set /a count+=1
echo.
echo 📊 Running evaluation script %count%/2: %script2%
python %script2%
if %errorlevel% neq 0 (
    echo ❌ Error running %script2%
    goto :eof
)
echo ✅ Finished: %script2%

echo.
echo 🎉 All evaluations completed successfully!
pause
