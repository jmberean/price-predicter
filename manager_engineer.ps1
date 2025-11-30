# manager_engineer.ps1
# Two-agent workflow: Manager assigns tasks, Engineer executes, loop until complete

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Goal,

    [Parameter()]
    [int]$MaxIterations = 20,

    [Parameter()]
    [string]$LogFile = "agent_workflow_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
)

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    $logEntry = "[$timestamp] $Message"
    Write-Host $logEntry -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $logEntry
}

function Parse-Response {
    param([string]$Text)

    # Check for DONE:: marker
    if ($Text -match '^DONE::(.*)') {
        return @{ Status = "DONE"; Message = $Matches[1].Trim() }
    }

    # Check for TASK:: marker
    if ($Text -match 'TASK::(.*)') {
        return @{ Status = "TASK"; Task = $Matches[1].Trim() }
    }

    # No special marker - continue working
    return @{ Status = "CONTINUE"; Message = $Text }
}

# Initialize
Write-Log "=== MANAGER-ENGINEER WORKFLOW ===" -Color Cyan
Write-Log "Goal: $Goal" -Color Yellow
Write-Log "Max Iterations: $MaxIterations" -Color Gray
Write-Log "Log File: $LogFile" -Color Gray
Write-Log ""

$session = $null
$iteration = 0
$managerMode = $true  # Start with Manager assigning initial task

while ($iteration -lt $MaxIterations) {
    $iteration++
    Write-Log "`n=== ITERATION $iteration ===" -Color Cyan

    try {
        if ($managerMode) {
            # MANAGER: Analyze and assign next task
            Write-Log "" -Color White
            Write-Log "╔══════════════════════════════════════════════════════════════╗" -Color Magenta
            Write-Log "║                    MANAGER AGENT                              ║" -Color Magenta
            Write-Log "╚══════════════════════════════════════════════════════════════╝" -Color Magenta

            if ($session -eq $null) {
                # First iteration: Set up the goal
                Write-Log "[MANAGER] Performing initial codebase analysis..." -Color Magenta
                Write-Log "[MANAGER] Goal: $Goal" -Color Magenta
                Write-Log "" -Color White

                $prompt = "You are the MANAGER in a two-agent workflow. Your role:`n" +
                          "1. Analyze the current state of the project`n" +
                          "2. Break down complex goals into specific, actionable tasks`n" +
                          "3. Assign ONE concrete task to the Engineer`n" +
                          "4. Evaluate Engineer's work and decide next steps`n`n" +
                          "GOAL: $Goal`n`n" +
                          "INSTRUCTIONS:`n" +
                          "- First, examine the codebase to understand the current state`n" +
                          "- Identify the root cause of poor SOTA training results`n" +
                          "- Assign ONE specific diagnostic or fix task`n" +
                          "- Format your response as: TASK:: <specific action for engineer>`n`n" +
                          "Begin your analysis now."

                $response = .\autoclaude.ps1 $prompt | ConvertFrom-Json
                $session = $response.session_id
                Write-Log "" -Color White
                Write-Log "[SESSION] Created: $session" -Color Gray
            } else {
                # Subsequent iterations: Evaluate and assign next task
                Write-Log "[MANAGER] Reviewing Engineer's work and deciding next step..." -Color Magenta
                Write-Log "" -Color White

                $prompt = "Review the Engineer's work above. Now:`n" +
                          "1. Did the Engineer complete the assigned task successfully?`n" +
                          "2. What did we learn?`n" +
                          "3. What should we do next?`n`n" +
                          "Respond with EXACTLY ONE of:`n" +
                          "- TASK:: <next specific action> (if more work needed)`n" +
                          "- DONE:: <summary of what was accomplished> (if goal achieved)"

                $response = .\autoclaude.ps1 $prompt -SessionId $session | ConvertFrom-Json
            }

            $parsed = Parse-Response $response.result
            Write-Log "" -Color White
            Write-Log "┌─ MANAGER DECISION ─────────────────────────────────────────┐" -Color Magenta
            Write-Log "$($response.result)" -Color White
            Write-Log "└────────────────────────────────────────────────────────────┘" -Color Magenta

            if ($parsed.Status -eq "DONE") {
                Write-Log "" -Color White
                Write-Log "╔══════════════════════════════════════════════════════════════╗" -Color Green
                Write-Log "║               WORKFLOW COMPLETE - SUCCESS!                    ║" -Color Green
                Write-Log "╚══════════════════════════════════════════════════════════════╝" -Color Green
                Write-Log "Summary: $($parsed.Message)" -Color Green
                break
            }

            if ($parsed.Status -eq "TASK") {
                Write-Log "" -Color White
                Write-Log ">>> TASK ASSIGNED TO ENGINEER >>>" -Color Yellow
                Write-Log "$($parsed.Task)" -Color Yellow
                Write-Log ">>> Switching to ENGINEER mode <<<" -Color Yellow
                $managerMode = $false  # Switch to Engineer
            } else {
                Write-Log "" -Color White
                Write-Log "[WARNING] Manager didn't provide clear TASK:: or DONE:: marker" -Color Yellow
                Write-Log "[WARNING] Will prompt Manager again..." -Color Yellow
            }

        } else {
            # ENGINEER: Execute the assigned task
            Write-Log "" -Color White
            Write-Log "╔══════════════════════════════════════════════════════════════╗" -Color Cyan
            Write-Log "║                   ENGINEER AGENT                              ║" -Color Cyan
            Write-Log "╚══════════════════════════════════════════════════════════════╝" -Color Cyan
            Write-Log "[ENGINEER] Executing assigned task..." -Color Cyan
            Write-Log "" -Color White

            $prompt = "You are the ENGINEER in a two-agent workflow.`n`n" +
                      "Execute the task assigned by the Manager above. Be thorough:`n" +
                      "- Use Read, Grep, Glob tools to investigate`n" +
                      "- Use Edit/Write to make changes if needed`n" +
                      "- Use Bash to run tests/scripts`n" +
                      "- Report your findings clearly`n`n" +
                      "When done, summarize what you did and what you found."

            $response = .\autoclaude.ps1 $prompt -SessionId $session | ConvertFrom-Json

            Write-Log "" -Color White
            Write-Log "┌─ ENGINEER REPORT ──────────────────────────────────────────┐" -Color Cyan
            Write-Log "$($response.result)" -Color White
            Write-Log "└────────────────────────────────────────────────────────────┘" -Color Cyan

            # Cost tracking
            Write-Log "" -Color White
            Write-Log "[METRICS] Turns: $($response.num_turns) | Cost: `$$($response.total_cost_usd) | Duration: $($response.duration_ms)ms" -Color Gray
            Write-Log ">>> Switching to MANAGER for evaluation <<<" -Color Yellow

            $managerMode = $true  # Switch back to Manager for evaluation
        }

    } catch {
        Write-Log "[ERROR] $($_.Exception.Message)" -Color Red
        Write-Log "[ERROR] Stack: $($_.ScriptStackTrace)" -Color Red

        # Try to recover by switching back to Manager
        $managerMode = $true
    }

    # Safety check - avoid infinite loops on same task
    if ($iteration -eq $MaxIterations) {
        Write-Log "`n[TIMEOUT] Reached max iterations ($MaxIterations)" -Color Red
        Write-Log "Last session ID: $session" -Color Gray
        Write-Log "You can resume manually: .\autoclaude.ps1 'Continue from where we left off' -SessionId $session" -Color Yellow
    }
}

Write-Log "`n=== WORKFLOW COMPLETE ===" -Color Cyan
Write-Log "Total iterations: $iteration" -Color Gray
Write-Log "Session ID: $session" -Color Gray
Write-Log "Full log: $LogFile" -Color Gray
