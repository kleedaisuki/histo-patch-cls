<#
.SYNOPSIS
Example runner for the histoclass CLI.

.DESCRIPTION
This script demonstrates an end-to-end PowerShell workflow for this project:
1) Locate repository root from the script location.
2) Optionally activate the project virtual environment (.venv).
3) Build CLI args in the canonical form:
   histoclass <config_name> ... [--options] ...
4) Execute the command and forward the process exit code.

It prefers the installed console script `histoclass`.
If that command is not available, it falls back to:
`python -m histoclass_cli.main`.

.PARAMETER ConfigName
Configuration name used as the first positional argument.
Typical value: default (maps to configs/default.json in the CLI layer).

.PARAMETER Mode
Pipeline mode. Allowed values: train, eval, train_eval.
Default: train_eval.

.PARAMETER Checkpoint
Checkpoint path. Usually required for eval mode.
When provided, this script appends: --checkpoint <path>.

.PARAMETER NoVenv
Skip .venv activation.
Use this only if your shell already uses the intended Python environment.

.PARAMETER ExtraArgs
Extra arguments passed through to histoclass.
Use this for forward-compatibility when CLI adds new options.

.NOTES
Recommended one-time setup:
    pip install -e .
This installs the `histoclass` console entrypoint from pyproject.toml.

.EXAMPLE
.\scripts\run_histoclass_example.ps1 -ConfigName default

.EXAMPLE
.\scripts\run_histoclass_example.ps1 -ConfigName default -Mode train

.EXAMPLE
.\scripts\run_histoclass_example.ps1 -ConfigName default -Mode eval -Checkpoint "outputs/checkpoints/best.pt"

.EXAMPLE
.\scripts\run_histoclass_example.ps1 -ConfigName default -ExtraArgs @('--mode', 'train')

.EXAMPLE
.\scripts\run_histoclass_example.ps1 -ConfigName default -NoVenv
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ConfigName,

    [ValidateSet('train', 'eval', 'train_eval')]
    [string]$Mode = 'train_eval',

    [string]$Checkpoint,

    [switch]$NoVenv,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# scripts directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# repository root is one level above scripts
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')

Write-Host ('[info] Repo root: {0}' -f $repoRoot)

Push-Location $repoRoot
try {
    # Step 1: activate venv unless explicitly disabled
    if (-not $NoVenv) {
        $activateScript = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'
        if (Test-Path $activateScript) {
            Write-Host ('[info] Activating venv: {0}' -f $activateScript)
            . $activateScript
        }
        else {
            Write-Warning ('Venv activation script not found: {0}' -f $activateScript)
            Write-Warning 'Continue with current shell Python environment.'
        }
    }

    # Step 2: assemble canonical CLI arguments
    $cliArgs = @($ConfigName, '--mode', $Mode)

    if ($Checkpoint) {
        $cliArgs += @('--checkpoint', $Checkpoint)
    }

    if ($ExtraArgs) {
        $cliArgs += $ExtraArgs
    }

    Write-Host ('[info] Effective args: {0}' -f ($cliArgs -join ' '))

    # Step 3: use installed console entrypoint when available
    $histoclassCmd = Get-Command histoclass -ErrorAction SilentlyContinue
    if ($null -ne $histoclassCmd) {
        Write-Host ('[info] Running: histoclass {0}' -f ($cliArgs -join ' '))
        & histoclass @cliArgs
        exit $LASTEXITCODE
    }

    # Step 4: fallback path for editable source execution
    Write-Warning 'histoclass command not found, fallback to python -m histoclass_cli.main'
    Write-Host ('[info] Running: python -m histoclass_cli.main {0}' -f ($cliArgs -join ' '))
    & python -m histoclass_cli.main @cliArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}