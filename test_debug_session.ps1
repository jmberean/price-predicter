# Debug script to see actual JSON output

$secret = [guid]::NewGuid().ToString()
Write-Host "[TEST] Secret: $secret`n" -ForegroundColor Yellow

$prompt1 = "Remember this token: $secret. Reply with: ACK"

Write-Host "=== First call (RAW OUTPUT) ===" -ForegroundColor Cyan
$raw1 = .\autoclaude.ps1 $prompt1
Write-Host $raw1 -ForegroundColor Gray
Write-Host "`n=== Parsing JSON ===" -ForegroundColor Cyan
$json1 = $raw1 | ConvertFrom-Json
Write-Host "Session ID: $($json1.session_id)" -ForegroundColor Green
Write-Host "Content: $($json1.content)" -ForegroundColor Green
Write-Host "Keys: $($json1 | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name)" -ForegroundColor Gray
