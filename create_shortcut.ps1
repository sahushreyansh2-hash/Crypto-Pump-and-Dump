$wsh = New-Object -ComObject WScript.Shell
$sc = $wsh.CreateShortcut("$env:USERPROFILE\Desktop\Crypto Pump Early Warning System.lnk")
$sc.TargetPath = "cmd.exe"
$sc.Arguments = "/c `"$env:USERPROFILE\Desktop\Launch Crypto Pump Warning System.bat`""
$sc.WorkingDirectory = "C:\Users\shrey\.gemini\antigravity\scratch\crypto-pump-detection"
$sc.IconLocation = "$env:SystemRoot\System32\shell32.dll,220"
$sc.Description = "Launch AI Crypto Pump-and-Dump Early Warning Terminal"
$sc.Save()
Write-Host "Shortcut created successfully!"
