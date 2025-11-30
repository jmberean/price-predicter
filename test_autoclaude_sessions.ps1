# test_autoclaude_sessions.ps1
# Tests session persistence by storing a secret in call 1, recalling in call 2

$secret = [guid]::NewGuid().ToString()
Write-Host "[SECRET] Token (only you see this): $secret" -ForegroundColor Yellow

$prompt1 = "We are running a statelessness test.`n`n" +
           "REMEMBER this exact secret token and nothing else:`n" +
           "$secret`n`n" +
           "Reply with exactly: ACK"

Write-Host "`n=== First call (creating new session) ===`n" -ForegroundColor Cyan
$response1 = .\autoclaude.ps1 $prompt1 | ConvertFrom-Json

# Extract session ID from JSON response
$sessionId = $response1.session_id
Write-Host "`n[SESSION] Captured session ID: $sessionId" -ForegroundColor Gray

if (-not $sessionId) {
    Write-Error "Failed to capture session ID from first call"
    exit 1
}

$prompt2 = "This is a completely new request.`n`n" +
           "Earlier I asked you to remember a secret token.`n`n" +
           "If you remember it EXACTLY, reply with ONLY that token (no explanation, no quotes).`n`n" +
           "If you do NOT remember, are UNSURE, or are GUESSING, reply with EXACTLY: NO_MEMORY"

Write-Host "`n=== Second call (resuming session) ===`n" -ForegroundColor Cyan
$response2 = .\autoclaude.ps1 $prompt2 -SessionId $sessionId | ConvertFrom-Json

Write-Host "`n[RESULT] Claude's response:" -ForegroundColor Cyan
Write-Host $response2.result -ForegroundColor White

if ($response2.result.Trim() -eq $secret) {
    Write-Host "`n[SUCCESS] Session persistence WORKS! Claude recalled the secret." -ForegroundColor Green
} else {
    Write-Host "`n[FAIL] Session persistence failed. Expected: $secret" -ForegroundColor Red
    Write-Host "Got: $($response2.result)" -ForegroundColor Red
}
