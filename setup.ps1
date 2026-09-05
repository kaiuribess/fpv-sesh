[CmdletBinding()]
param([string]$PythonPath = '', [switch]$CheckOnly, [switch]$Development)
$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$rootPrefix = $appRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
. (Join-Path $PSScriptRoot 'setup-common.ps1')
$setupGuard = $null

function Get-AppPath([string]$relativePath) {
    if ([IO.Path]::IsPathRooted($relativePath)) { throw 'Expected an application-relative path.' }
    $absolutePath = [IO.Path]::GetFullPath((Join-Path $appRoot $relativePath))
    if (-not $absolutePath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escapes the application folder.' }
    return $absolutePath
}
function Test-Hash([string]$path, [string]$expected) {
    if ($expected -notmatch '^[0-9a-fA-F]{64}$' -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    $hasher = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($path)
    try { return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '') -eq $expected }
    finally { $stream.Dispose(); $hasher.Dispose() }
}
function Test-Python([string]$path) {
    if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    $ErrorActionPreference = 'Continue'
    & $path -c "import struct,sys,tkinter; assert sys.platform=='win32' and sys.version_info[:2] in {(3,12),(3,13)} and struct.calcsize('P')==8" 2>$null
    return $LASTEXITCODE -eq 0
}
function Find-Python {
    if ($PythonPath) {
        $candidate = (Resolve-Path -LiteralPath $PythonPath).Path
        if (Test-Python $candidate) { return $candidate }
        throw 'The selected Python needs to be 64-bit version 3.12 or 3.13, including Tcl/Tk.'
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($minor in @('-3.13', '-3.12')) {
            $ErrorActionPreference = 'Continue'
            $candidate = & $launcher.Source $minor -c 'import sys; print(sys.executable)' 2>$null
            $probeResult = $LASTEXITCODE
            $ErrorActionPreference = 'Stop'
            if ($probeResult -eq 0 -and (Test-Python $candidate)) { return $candidate }
        }
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike '*\WindowsApps\*' -and (Test-Python $command.Source)) { return $command.Source }
    throw 'Install 64-bit Python 3.13 with Tcl/Tk from https://www.python.org/downloads/windows/ and run install.cmd again. Python 3.12 is also supported.'
}
function Install-Lock([string]$name) {
    & $venvPython -m pip --isolated install --disable-pip-version-check --index-url https://pypi.org/simple --only-binary=:all: --require-hashes --no-deps --force-reinstall --retries 3 --timeout 60 -r (Get-AppPath $name)
    if ($LASTEXITCODE -ne 0) { throw "Package installation failed for $name. Check the internet connection and rerun install.cmd." }
}

Push-Location -LiteralPath $appRoot
try {
    if (-not $CheckOnly) { $setupGuard = Enter-FpvSetupLock -AppRoot $appRoot }
    Write-Host 'FPV Sesh setup - editing and exports stay on this computer.'
    Write-Host 'Setup downloads verified Python packages and video tools. Optional models are installed separately.'
    $venvPython = Get-AppPath '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if ($CheckOnly) { throw 'Run install.cmd to create the application environment.' }
        if (Test-Path -LiteralPath (Get-AppPath '.venv')) { throw 'The .venv folder is incomplete. Extract a fresh release copy; your old projects will remain safe.' }
        $basePython = Find-Python
        & $basePython -m venv (Get-AppPath '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the application environment. Use a writable folder outside Program Files.' }
    }
    if (-not (Test-Python $venvPython)) { throw 'This environment needs 64-bit Python 3.12 or 3.13 with Tkinter. Extract a fresh release copy and run install.cmd.' }
    & $venvPython -c "from pathlib import Path; import sys; assert sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() == Path('.venv').resolve()"
    if ($LASTEXITCODE -ne 0) { throw 'The Python environment is outside this application.' }
    if (-not $CheckOnly) {
        foreach ($directory in @('input','music','output','cache','models','logs','tools/downloads')) {
            $directoryPath = Get-AppPath $directory
            if (-not (Test-Path -LiteralPath $directoryPath)) { New-Item -ItemType Directory -Path $directoryPath | Out-Null }
        }
        Install-Lock 'requirements-bootstrap-lock.txt'
        # PySceneDetect now publishes a separate headless distribution. Both
        # variants own the same module; remove the old name before replacing it.
        & $venvPython -m pip --isolated uninstall --yes scenedetect
        if ($LASTEXITCODE -ne 0) { throw 'Could not replace the older scene detector.' }
        Install-Lock 'requirements-lock.txt'
        if ($Development) { Install-Lock 'requirements-dev-lock.txt' }
    }
    & $venvPython -m pip --isolated check
    if ($LASTEXITCODE -ne 0) { throw 'Installed packages conflict. Extract a clean application copy and run install.cmd.' }
    $manifest = Get-Content -LiteralPath (Get-AppPath 'tools/dependencies.json') -Raw | ConvertFrom-Json
    foreach ($tool in $manifest.tools | Where-Object { $_.required }) {
        $archivePath = Get-AppPath $tool.archive
        $destinationPath = Get-AppPath $tool.extract_to
        $toolValid = $true
        foreach ($entry in @(@('ffmpeg','executable'),@('ffprobe','probe_executable'),@('ffplay','player_executable'))) {
            if (-not (Test-Hash (Get-AppPath $tool.($entry[1])) $tool.executable_sha256.($entry[0]))) { $toolValid = $false }
        }
        if (-not $toolValid) {
            if ($CheckOnly) { throw 'The verified video tools are missing or changed. Run install.cmd to repair them.' }
            Write-Host "Downloading, extracting and checking $($tool.name) $($tool.version)..."
            & $venvPython (Get-AppPath 'scripts/fetch_tool.py')
            if ($LASTEXITCODE -ne 0) { throw 'Video-tool setup did not finish. Check internet access/free space, then run install.cmd again.' }
            foreach ($entry in @(@('ffmpeg','executable'),@('ffprobe','probe_executable'),@('ffplay','player_executable'))) {
                if (-not (Test-Hash (Get-AppPath $tool.($entry[1])) $tool.executable_sha256.($entry[0]))) { throw 'Extracted video-tool integrity check failed. Run install.cmd again.' }
            }
        }
    }
    & $venvPython -m fpvsesh.installation
    if ($LASTEXITCODE -ne 0) { throw 'The installation check failed. Run doctor.cmd for guidance.' }
    Write-Host 'Setup complete. Double-click launch.cmd to start editing.'
} finally {
    if ($setupGuard) { $setupGuard.Dispose() }
    Pop-Location
}
