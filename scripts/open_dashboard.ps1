# Opens the latest live dashboard in your browser (works while the repo is private).
# Downloads site/index.html from the `live` branch with the GitHub CLI.
$repo = "Vincent-Attakpah/Grid-demand-project"
$out = Join-Path $env:TEMP "nowcaster_dashboard.html"
$b64 = gh api "repos/$repo/contents/site/index.html?ref=live" --jq .content
if (-not $b64) { throw "could not fetch the dashboard (is gh logged in?)" }
[IO.File]::WriteAllBytes($out, [Convert]::FromBase64String(($b64 -join "")))
Start-Process $out
