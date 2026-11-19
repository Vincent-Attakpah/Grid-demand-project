# Backup trigger for the hourly logger (GitHub's own schedule is best effort and often skips).
# Run by a Windows scheduled task at :17 each hour. Starts the logger workflow only if no
# run was created in the last 50 minutes, so it never doubles up with GitHub's schedule.
# Register it with: powershell -File scripts\install_logger_task.ps1

$ErrorActionPreference = "Stop"
$repo = "Vincent-Attakpah/Grid-demand-project"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
$log = Join-Path $env:USERPROFILE "nowcaster-data\logs\logger_trigger.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Write-Log($msg) {
    Add-Content -Path $log -Value ("{0:u} {1}" -f (Get-Date).ToUniversalTime(), $msg)
}

try {
    $runs = & $gh run list -R $repo -w logger.yml -L 5 --json createdAt | ConvertFrom-Json
    $cutoff = (Get-Date).ToUniversalTime().AddMinutes(-50)
    $recent = @($runs | Where-Object { ([datetime]$_.createdAt).ToUniversalTime() -gt $cutoff })
    if ($recent.Count -gt 0) {
        Write-Log "skip: GitHub already ran the logger at $($recent[0].createdAt)"
        exit 0
    }
    & $gh workflow run logger.yml -R $repo
    if ($LASTEXITCODE -ne 0) { throw "gh workflow run exited $LASTEXITCODE" }
    Write-Log "triggered"
} catch {
    Write-Log "error: $_"
    exit 1
}
