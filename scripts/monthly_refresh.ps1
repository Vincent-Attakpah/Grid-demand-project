# Monthly refit of the live forecaster (run on the laptop early each month, ideally on the 1st-3rd).
# Pulls the newest EIA-930 bulk file and weather, rebuilds the model inputs, refits M3/M4 on
# everything published so far, and replaces the model bundle on the GitHub release
# `live-models`. The hourly GitHub workflow picks up the new bundle on its next run.
# Takes about 10-30 minutes (mostly the weather download). Log: ~/nowcaster-data/logs/monthly_refresh.log

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $env:USERPROFILE ".venvs\nowcaster\Scripts\python.exe"
$log = Join-Path $env:USERPROFILE "nowcaster-data\logs\monthly_refresh.log"
$env:PYTHONPATH = Join-Path $repo "src"
$env:PYTHONWARNINGS = "ignore"
Set-Location $repo

function Invoke-Step($name, [string[]]$pyArgs) {
    "$(Get-Date -Format s) $name" | Tee-Object -FilePath $log -Append
    & $py @pyArgs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "step failed: $name (see $log)" }
}

Invoke-Step "download newest EIA-930 bulk file" @("-m", "nowcaster.ingest.eia_bulk", "--end", (Get-Date).Year, "--refresh-current")
Invoke-Step "clean EIA data" @("-m", "nowcaster.ingest.clean")
Invoke-Step "download new weather months" @("-m", "nowcaster.weather.backfill")
Invoke-Step "rebuild model inputs" @("-m", "nowcaster.features.build")
Invoke-Step "refit and upload live models" @("-m", "nowcaster.live.train", "--upload")
"$(Get-Date -Format s) done" | Tee-Object -FilePath $log -Append
