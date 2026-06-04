' BIST Sinyal Robotu — Sessiz WSL Başlatıcı
' Bu dosya Windows açılışında arka planda WSL'yi başlatır
' Görev Zamanlayıcısı'na ekleyin

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "wsl -d Ubuntu -u hyayan -e bash -c 'sudo systemctl start bist-robot 2>/dev/null; sleep 2'", 0, False
