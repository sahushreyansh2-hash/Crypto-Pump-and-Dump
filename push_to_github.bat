@echo off
cd /d "%~dp0"
echo =======================================================
echo Pushing Crypto Pump Detection System to GitHub
echo Remote: https://github.com/sahushreyansh2-hash/crypto-pump-detection.git
echo =======================================================
echo.
git push -u origin main
if %ERRORLEVEL% equ 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Repository successfully pushed to GitHub!
    echo Visit: https://github.com/sahushreyansh2-hash/crypto-pump-detection
    echo =======================================================
) else (
    echo.
    echo [NOTE] If repository does not exist on GitHub yet:
    echo 1. Go to https://github.com/new
    echo 2. Repository name: crypto-pump-detection
    echo 3. Click 'Create repository' (leave empty)
    echo 4. Run this script again!
)
pause
