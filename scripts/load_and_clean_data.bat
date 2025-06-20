@echo off
REM ===== Batch script to run preprocessing steps =====
REM Ensure the virtual environment is activated
call .\venv\Scripts\activate

setlocal enabledelayedexpansion

:: Define scripts to run (same order as Python version)
set scripts[1]=..\preprocess\text_preprocessing\load_data.py
set scripts[2]=..\preprocess\text_preprocessing\simplify_phonetics.py
set scripts[3]=..\preprocess\misspellings_preprocessing\load_mispelling.py
set scripts[4]=..\preprocess\misspellings_preprocessing\clean_misspellings.py

set total=4
set count=0

:loop
set /a count+=1
if !count! gtr %total% goto end

echo.
echo 🔄 Running script !count!/%total%: !scripts[%count%]!
python "!scripts[%count%]!"

if %errorlevel% neq 0 (
    echo ❌ Error while running !scripts[%count%]!
    pause
    exit /b %errorlevel%
)
echo ✅ Finished: !scripts[%count%]!
goto loop

:end
echo.
echo 🎉 All preprocessing scripts completed successfully!
pause