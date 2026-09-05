[CmdletBinding()]
param(
    [string]$PythonPath = '',
    [switch]$IncludeOptionalModels,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$rootPrefix = $appRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

function Get-AppPath([string]$relativePath) {
    if ([IO.Path]::IsPathRooted($relativePath)) { throw "Application paths must be relative: $relativePath" }
    $absolutePath = [IO.Path]::GetFullPath((Join-Path $appRoot $relativePath))
    if (-not $absolutePath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the application folder: $relativePath"
    }
    return $absolutePath
}

function Confirm-ArchiveHash([string]$archivePath, [string]$expectedHash) {
    if ($expectedHash -notmatch '^[0-9a-fA-F]{64}$') { throw "Missing valid recorded SHA256 for $archivePath" }
    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    if ($actualHash -ne $expectedHash) {
        throw "Checksum mismatch: $archivePath. The archive was not used."
    }
}

Push-Location -LiteralPath $appRoot
try {
    Write-Host 'FPV Sesh local setup'
    Write-Host 'This uses only the project environment and application folders. No GPU drivers or system settings are changed.'
    foreach ($directory in @('input', 'music', 'output', 'cache', 'models', 'logs', 'tools', 'tools/downloads')) {
        $directoryPath = Get-AppPath $directory
        if (-not (Test-Path -LiteralPath $directoryPath)) {
            if ($CheckOnly) { throw "Missing application folder: $directoryPath" }
            New-Item -ItemType Directory -Path $directoryPath | Out-Null
        }
    }

    $venvPython = Get-AppPath '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if ($CheckOnly) { throw 'The project Python environment has not been installed.' }
        if (Test-Path -LiteralPath (Get-AppPath '.venv')) {
            throw 'The existing .venv is incomplete. Setup will not erase it; repair it deliberately or use a clean application copy.'
        }
        if ($PythonPath) {
            $basePython = (Resolve-Path -LiteralPath $PythonPath).Path
        } else {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if (-not $pythonCommand) { throw 'Install Python 3.12 with Tkinter from https://www.python.org/downloads/windows/, or supply -PythonPath to an existing Python 3.12 executable.' }
            $basePython = $pythonCommand.Source
        }
        & $basePython -c "import sys, tkinter; assert sys.version_info[:2] == (3, 12), 'Use the tested Python 3.12 runtime.'"
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 with Tkinter is required for this tested setup.' }
        & $basePython -m venv (Get-AppPath '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the isolated project environment failed.' }
    }

    & $venvPython -c "import pathlib,sys,tkinter; assert sys.version_info[:2] == (3,12), 'Expected Python 3.12'; assert sys.prefix != sys.base_prefix, 'Expected an isolated environment'; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path('.venv').resolve(), 'Environment is outside this application'"
    if ($LASTEXITCODE -ne 0) { throw 'The project environment check failed.' }
    if (-not $CheckOnly) {
        Write-Host 'Installing the exact tested Python packages from PyPI into .venv...'
        & $venvPython -m pip --isolated install --disable-pip-version-check --index-url 'https://pypi.org/simple' --only-binary=:all: -r (Get-AppPath 'requirements-lock.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Installing the pinned packages failed.' }
    }
    & $venvPython -c "from importlib.metadata import version; from pathlib import Path; pins=[line.strip().split('==') for line in Path('requirements-lock.txt').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]; bad=[name+' expected '+want+' found '+version(name) for name,want in pins if version(name)!=want]; assert not bad, '; '.join(bad); print('Pinned Python package versions verified.')"
    if ($LASTEXITCODE -ne 0) { throw 'Installed packages do not match requirements-lock.txt.' }

    $manifest = Get-Content -LiteralPath (Get-AppPath 'tools/dependencies.json') -Raw | ConvertFrom-Json
    foreach ($tool in $manifest.tools) {
        if (-not $tool.required -and -not $IncludeOptionalModels) { continue }
        $executablePath = Get-AppPath $tool.executable
        $archivePath = Get-AppPath $tool.archive
        $destinationPath = Get-AppPath $tool.extract_to
        if (Test-Path -LiteralPath $archivePath -PathType Leaf) {
            Confirm-ArchiveHash $archivePath $tool.sha256
        }
        if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
            if ($CheckOnly) { throw "Missing tool: $($tool.name). Run setup without -CheckOnly." }
            if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
                $sourceUri = [uri]$tool.url
                if ($sourceUri.Scheme -ne 'https' -or $sourceUri.Host -ne 'github.com') {
                    throw "Unexpected download origin for $($tool.name); inspect tools/dependencies.json."
                }
                Write-Host "Downloading $($tool.name) $($tool.version) from its recorded upstream release..."
                $partialPath = $archivePath + '.download'
                Invoke-WebRequest -Uri $sourceUri.AbsoluteUri -OutFile $partialPath -UseBasicParsing
                Confirm-ArchiveHash $partialPath $tool.sha256
                Move-Item -LiteralPath $partialPath -Destination $archivePath
            }
            if (-not (Test-Path -LiteralPath $destinationPath)) {
                New-Item -ItemType Directory -Path $destinationPath | Out-Null
            }
            Write-Host "Extracting $($tool.name) into the application tools folder..."
            if ($tool.format -eq 'zip') {
                Expand-Archive -LiteralPath $archivePath -DestinationPath $destinationPath -Force
            } elseif ($tool.format -eq '7z') {
                $tarCommand = Get-Command tar.exe -ErrorAction SilentlyContinue
                if (-not $tarCommand) { throw 'Windows tar.exe with 7z support is required to extract this verified FFmpeg archive.' }
                & $tarCommand.Source -xf $archivePath -C $destinationPath
                if ($LASTEXITCODE -ne 0) { throw "Extracting $($tool.name) failed." }
            } else {
                throw "Unsupported archive format: $($tool.format)"
            }
        }
        if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
            throw "Expected executable missing after extraction: $executablePath"
        }
        Write-Host "$($tool.name) is available. License: $($tool.license)."
        if ($tool.required -and $tool.name -eq 'FFmpeg') {
            & $executablePath -hide_banner -version | Select-Object -First 1
            if ($LASTEXITCODE -ne 0) { throw 'The required FFmpeg executable could not run.' }
            $probePath = Get-AppPath $tool.probe_executable
            & $probePath -hide_banner -version | Select-Object -First 1
            if ($LASTEXITCODE -ne 0) { throw 'The required ffprobe executable could not run.' }
        }
    }

    if ($IncludeOptionalModels) {
        foreach ($model in $manifest.models) {
            foreach ($modelFile in $model.files) {
                Confirm-ArchiveHash (Get-AppPath $modelFile.path) $modelFile.sha256
            }
        }
        Write-Host 'Optional Video2X and bundled model files verified. This does not enable the unstable AI backend.'
    }
    & $venvPython -m fpvsesh.cli --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The FPV Sesh command line failed its import/startup check.' }
    Write-Host 'Setup verified. Double-click launch.cmd to open FPV Sesh.'
    Write-Host 'Archive hashes are recorded local hashes; publisher-checksum availability and licenses are documented in tools/dependencies.json.'
} finally {
    Pop-Location
}
