# test_autoclaude_sessions_robust.ps1
# More robust version that waits for claude process to exit

$sessionId = [guid]::NewGuid().ToString()
$secret = [guid]::NewGuid().ToString()
Write-Host "[SECRET] Token (only you see this): $secret" -ForegroundColor Yellow
Write-Host "[SESSION] Using session: $sessionId`n" -ForegroundColor Gray

$prompt1 = "We are running a statelessness test.`n`n" +
           "REMEMBER this exact secret token and nothing else:`n" +
           "$secret`n`n" +
           "Reply with exactly: ACK"

Write-Host "`n=== First call to autoclaude.ps1 ===`n" -ForegroundColor Cyan
.\autoclaude.ps1 $prompt1 -SessionId $sessionId

# Wait for all claude/node processes to exit
Write-Host "`n[WAIT] Waiting for claude process to fully exit..." -ForegroundColor Gray
$maxWait = 10  # seconds
$waited = 0
while ((Get-Process -Name "node" -ErrorAction SilentlyContinue) -and ($waited -lt $maxWait)) {
    Start-Sleep -Milliseconds 500
    $waited += 0.5
    Write-Host "." -NoNewline -ForegroundColor Gray
}
Write-Host " Done ($waited seconds)" -ForegroundColor Gray

$prompt2 = "This is a completely new request.`n`n" +
           "Earlier I asked you to remember a secret token.`n`n" +
           "If you remember it EXACTLY, reply with ONLY that token (no explanation, no quotes).`n`n" +
           "If you do NOT remember, are UNSURE, or are GUESSING, reply with EXACTLY: NO_MEMORY"

Write-Host "`n=== Second call to autoclaude.ps1 ===`n" -ForegroundColor Cyan
.\autoclaude.ps1 $prompt2 -SessionId $sessionId

Write-Host "`n[DONE] Test complete. Claude should have recalled: $secret" -ForegroundColor Green
