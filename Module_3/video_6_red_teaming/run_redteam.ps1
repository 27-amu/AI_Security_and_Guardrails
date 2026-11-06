<#
run_redteam.ps1 - the one command that starts the whole red team

What is this file?
In simple words
  It is the "start" button. Before pressing it the script:
    1. goes to this folder
    2. puts our project's Python tools first on the PATH, so the `promptfoo` command is found
    3. reads the OpenAI key from Module_3\.env (it never prints the key)
    4. tells promptfoo to stay on this PC: no account, no online attack generator, no usage reporting
    5. runs the red team and writes the result to results.json

Why do we need it?
  Starting promptfoo by hand needs about six separate setup steps (find the Python tools, load the OpenAI key,
  switch off online features). 
  
  Forget one and the run fails, or quietly sends attack generation to an online service. 
  
  The script does all of it the same way every time, so every run is a single command.



Run it from PowerShell in this folder:  .\run_redteam.ps1        

See the report card:  promptfoo view
#>

# 1. work from this folder, and find the project's Python tools (one level up, in .venv)
Set-Location $PSScriptRoot
$venv = Resolve-Path "..\.venv\Scripts"

# 2. load Module_3\.env (the line may be "KEY =value", so trim around "=")
Get-Content "..\.env" | ForEach-Object {
    if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2].Trim() }
}

# 3. everything stays on this PC: no promptfoo account, no hosted generation
$env:PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION = "true"
$env:PROMPTFOO_DISABLE_REMOTE_GENERATION = "true"
$env:PROMPTFOO_DISABLE_TELEMETRY = "1"
$env:PATH = "$venv;$env:PATH"

# 4. fire every attack once and save the result (--no-cache = always ask the model again, no stale answers)
promptfoo redteam run --no-cache -o results.json
