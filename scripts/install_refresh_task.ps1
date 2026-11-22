# Registers the "Nowcaster monthly refit" scheduled task: runs monthly_refresh.ps1 on the 2nd of
# every month at 09:00 (the 2nd, so the previous month's last day is fully published).
# If the laptop is off or asleep then, it runs as soon as it is available (and online).
# Remove with: Unregister-ScheduledTask -TaskName "Nowcaster monthly refit" -Confirm:$false

$name = "Nowcaster monthly refit"
$script = Join-Path $PSScriptRoot "monthly_refresh.ps1"
$args_ = [Security.SecurityElement]::Escape(
    "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`"")
$start = (Get-Date -Day 2 -Hour 9 -Minute 0 -Second 0)
if ($start -lt (Get-Date)) { $start = $start.AddMonths(1) }
$user = [Security.SecurityElement]::Escape([Security.Principal.WindowsIdentity]::GetCurrent().Name)

# The PowerShell task cmdlets have no monthly trigger, so the task is defined in XML.
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Monthly refit of the grid demand nowcaster models</Description></RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>$($start.ToString("s"))</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByMonth>
        <DaysOfMonth><Day>2</Day></DaysOfMonth>
        <Months><January/><February/><March/><April/><May/><June/><July/><August/><September/><October/><November/><December/></Months>
      </ScheduleByMonth>
    </CalendarTrigger>
  </Triggers>
  <Principals><Principal id="Author"><UserId>$user</UserId><LogonType>InteractiveToken</LogonType></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <ExecutionTimeLimit>PT3H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>powershell.exe</Command><Arguments>$args_</Arguments></Exec>
  </Actions>
</Task>
"@
Register-ScheduledTask -TaskName $name -Xml $xml -Force | Out-Null
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State,
    @{n = "NextRun"; e = { (Get-ScheduledTaskInfo -TaskName $name).NextRunTime } }
