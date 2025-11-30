<#
.SYNOPSIS
    Wrapper for headless Claude Code interactions on Windows.
.EXAMPLE
    .\autoclaude.ps1 "Say hi"
.EXAMPLE
    Get-Content error.log | .\autoclaude.ps1 "Fix this"
#>
[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [string]$Prompt,

    [Parameter(ValueFromPipeline=$true)]
    [string]$PipelineInput,

    [Parameter()]
    [string]$SessionId = "new"  # "new" creates fresh session, or provide session ID to resume
)

begin {
    # Check if claude is installed
    if (-not (Get-Command "claude" -ErrorAction SilentlyContinue)) {
        Write-Error "[ERROR] 'claude' command not found. Please install it globally (npm install -g @anthropic-ai/claude-code)."
        exit 1
    }
    
    # Define tools to auto-approve
    $AllowedTools = "Bash,Edit,Read,Write"
    $FullContext = ""
}

process {
    # Accumulate pipeline input if it exists
    if ($PipelineInput) {
        $FullContext += $PipelineInput + "`n"
    }
}

end {
    # Combine pipeline data and argument prompt
    $FinalPrompt = $Prompt
    if ($FullContext) {
        $FinalPrompt = "Context:`n$FullContext`n`nInstruction: $Prompt"
    }

    if (-not $FinalPrompt) {
        Write-Warning "[WARN] Usage: .\autoclaude.ps1 'Your instruction'"
        return
    }

    # Check if we're resuming an existing session (SessionId provided and not default)
    if ($SessionId -and $SessionId -ne "new") {
        Write-Host "[AUTO] Resuming session: $SessionId" -ForegroundColor Cyan
        Write-Host "[WORKING] Claude is thinking and using tools..." -ForegroundColor Gray
        $output = claude -p "$FinalPrompt" --allowedTools "$AllowedTools" --resume "$SessionId" --output-format stream-json 2>&1
    } else {
        Write-Host "[AUTO] Starting new session..." -ForegroundColor Cyan
        Write-Host "[WORKING] Claude is thinking and using tools..." -ForegroundColor Gray
        # Output JSON to capture session ID for future resume
        $output = claude -p "$FinalPrompt" --allowedTools "$AllowedTools" --output-format stream-json 2>&1
    }

    # Process streaming JSON output
    $finalResult = $null
    $output | ForEach-Object {
        $line = $_.ToString()

        # Try to parse as JSON event
        try {
            $event = $line | ConvertFrom-Json -ErrorAction Stop

            # Display thinking/progress in real-time
            if ($event.type -eq "tool_use") {
                Write-Host "  [TOOL] $($event.name): $($event.input | ConvertTo-Json -Compress)" -ForegroundColor Yellow
            } elseif ($event.type -eq "text") {
                Write-Host "  $($event.text)" -ForegroundColor White
            } elseif ($event.type -eq "result") {
                $finalResult = $event
            }
        } catch {
            # Non-JSON line, might be stderr
            if ($line -and $line -notmatch '^\s*$') {
                Write-Host "  $line" -ForegroundColor Gray
            }
        }
    }

    # Return final JSON result for parsing
    if ($finalResult) {
        $finalResult | ConvertTo-Json -Depth 10
    } else {
        # Fallback: return last line if no result event found
        $output | Select-Object -Last 1
    }
}