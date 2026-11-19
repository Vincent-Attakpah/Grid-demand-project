# Registers the "Nowcaster logger backup" scheduled task: runs trigger_logger.ps1 at :17
# every hour while you are logged in; a run missed while the laptop slept fires on wake.
# Remove with: Unregister-ScheduledTask -TaskName "Nowcaster logger backup" -Confirm:$false

$script = Join-Path $PSScriptRoot "trigger_logger.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$start = (Get-Date).Date.AddHours((Get-Date).Hour).AddMinutes(17)
if ($start -lt (Get-Date)) { $start = $start.AddHours(1) }
$trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "Nowcaster logger backup" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Backup hourly trigger for the GitHub logger workflow" -Force |
    Select-Object TaskName, State
