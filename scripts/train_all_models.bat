@echo off
REM ===== Batch script to run training processes =====
REM Activate Python virtual environment
call .\venv\Scripts\activate

setlocal enabledelayedexpansion

:: Define training scripts to run
set scripts[1]=..\training\train_embeddings.py
set scripts[2]=..\training\train_simplified_phonetics.py
set scripts[3]=..\training\train_finetuned_fasttext.py
set scripts[4]=..\training\patch_finetuned_fasttext.py

set total=4
set count=0

:training_loop
set /a count+=1
if !count! gtr %total% goto training_complete

echo.
echo 🚀 Running script !count!/%total%: !scripts[%count%]!
python "!scripts[%count%]!"

if %errorlevel% neq 0 (
    echo ❌ Error in script !scripts[%count%]!
    echo Last error code: %errorlevel%
    pause
    exit /b %errorlevel%
)
echo ✅ Completed: !scripts[%count%]!
goto training_loop

:training_complete
echo.
echo 🎉 All training scripts executed successfully!
echo.
pause