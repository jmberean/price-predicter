# AutoClaude - Stateful Claude Code Automation

Wrapper for running Claude Code CLI with session persistence for automated workflows.

## How It Works

Based on official Claude Code documentation, session persistence requires:
1. **First call**: Create session with `--output-format json`, capture `session_id`
2. **Subsequent calls**: Use `--resume <session-id> --output-format json`

**Do NOT use `--session-id` for sequential calls** - this causes a lock error ([GitHub Issue #5524](https://github.com/anthropics/claude-code/issues/5524))

## Usage

### Basic (New Session Each Time)
```powershell
.\autoclaude.ps1 "Create a file test.txt"
```

### With Session Persistence
```powershell
# First call - creates session and returns JSON
$response1 = .\autoclaude.ps1 "Task 1" | ConvertFrom-Json
$sessionId = $response1.session_id

# Subsequent calls - resume the same session
$response2 = .\autoclaude.ps1 "Task 2" -SessionId $sessionId | ConvertFrom-Json
$response3 = .\autoclaude.ps1 "Task 3" -SessionId $sessionId | ConvertFrom-Json

# Access results
Write-Host $response2.result
```

### JSON Response Structure
```json
{
  "result": "ACK",
  "session_id": "440e82ac-0e11-4109-9f0c-fc09c3551c3a",
  "duration_ms": 2432,
  "total_cost_usd": 0.030217,
  "num_turns": 1,
  "is_error": false
}
```

## Manager-Engineer Workflow Example

```powershell
# Create a persistent session for the workflow
$session = $null

# Manager assigns task
$managerResponse = .\autoclaude.ps1 "Analyze auth.py and identify bugs" | ConvertFrom-Json
$session = $managerResponse.session_id
Write-Host "Manager: $($managerResponse.result)"

# Engineer works on it (remembers context from manager)
$engineerResponse = .\autoclaude.ps1 "Fix the bugs you identified" -SessionId $session | ConvertFrom-Json
Write-Host "Engineer: $($engineerResponse.result)"

# Manager evaluates (remembers all previous context)
$evalResponse = .\autoclaude.ps1 "Are we done or is there more work?" -SessionId $session | ConvertFrom-Json
Write-Host "Manager: $($evalResponse.result)"
```

## Testing

Run the test suite to verify session persistence:
```powershell
.\test_autoclaude_sessions.ps1
```

Expected output:
```
[SUCCESS] Session persistence WORKS! Claude recalled the secret.
```

## References

- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Session Management Guide](https://www.vibesparking.com/en/blog/ai/claude-code/docs/cli/2025-08-28-mastering-claude-code-sessions-continue-resume-automate/)
- [GitHub Issue #5524: Session ID lock bug](https://github.com/anthropics/claude-code/issues/5524)
- [GitHub Issue #1967: Resume in print mode](https://github.com/anthropics/claude-code/issues/1967)
