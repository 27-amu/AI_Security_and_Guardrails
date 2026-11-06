<#
run_redteam.ps1 - the one command that starts the multi-turn red team (Module 4, Video 6)

Before pressing it the script:
  1. goes to this folder and puts the project's Python tools (..\.venv) first on the PATH, so `promptfoo` is found
  2. reads the OpenAI key from ..\.env (it never prints the key)
  3. tells promptfoo to stay on this PC: no account, no online attack generator, no usage reporting
  4. resets the agent's world: fresh bookings and fresh memory (only our own copies under .\runs are touched)
  5. runs promptfooconfig_multiturn.yaml one attack at a time (-j 1: the agent's memory is shared by all attacks and
     lives in a local database that allows one process at a time) and writes the result to results.json

Run it from PowerShell in this folder:  .\run_redteam.ps1        See the report card:  promptfoo view
#>

Set-Location $PSScriptRoot
$venv = Resolve-Path "..\.venv\Scripts"

Get-Content "..\.env" | ForEach-Object {
    if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2].Trim() }
}

$env:PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION = "true"
$env:PROMPTFOO_DISABLE_REMOTE_GENERATION = "true"
$env:PROMPTFOO_DISABLE_TELEMETRY = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$venv;$env:PATH"

promptfoo redteam run -c promptfooconfig_multiturn.yaml --no-cache -j 1 -o results.json
