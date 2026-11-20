@echo off
cd /d "%~dp0"
echo =======================================================
echo Pushing Crypto Pump Detection System to GitHub
echo Remote: https://github.com/sahushreyansh2-hash/Crypto-Pump-and-Dump.git
echo =======================================================
echo.
git push -u origin main
if %ERRORLEVEL% equ 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Repository successfully pushed to GitHub!
    echo Visit: https://github.com/sahushreyansh2-hash/Crypto-Pump-and-Dump
    echo =======================================================
) else (
    echo.
    echo [NOTE] Git push failed. Please verify credentials or run 'gh auth login'.
)
pause
