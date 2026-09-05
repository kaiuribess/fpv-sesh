[CmdletBinding()]
param(
    [string]$PythonPath = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$rootPrefix = $appRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

function Get-AiPath([string]$relativePath) {
    if ([IO.Path]::IsPathRooted($relativePath)) { throw "Expected an application-relative path: $relativePath" }
    $absolutePath = [IO.Path]::GetFullPath((Join-Path $appRoot $relativePath))
    if (-not $absolutePath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the application folder: $relativePath"
    }
    return $absolutePath
}

function Confirm-AiFile([string]$path, [string]$sha256, [long]$size = -1) {
    if ($sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw "Missing SHA256 for $path" }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
    if ($size -ge 0 -and (Get-Item -LiteralPath $path).Length -ne $size) {
        throw "Recorded size does not match: $path"
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $sha256) {
        throw "SHA256 mismatch: $path. This file was not accepted."
    }
}

function Install-AiArtifact($entry, [string]$folder) {
    $destination = Get-AiPath ($folder + '/' + $entry.file)
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        Confirm-AiFile $destination $entry.sha256 $entry.size_bytes
        return
    }
    if ($CheckOnly) { throw "Missing $($entry.file). Run setup-ai.ps1 without -CheckOnly to install it." }
    $uri = [uri]$entry.url
    $allowed = $uri.Scheme -eq 'https' -and (
        ($uri.Host -eq 'github.com' -and $uri.AbsolutePath.StartsWith('/xinntao/Real-ESRGAN/releases/download/')) -or
        ($uri.Host -eq 'raw.githubusercontent.com' -and (
            $uri.AbsolutePath.StartsWith('/xinntao/Real-ESRGAN/') -or
            $uri.AbsolutePath.StartsWith('/XPixelGroup/BasicSR/'))))
    if (-not $allowed) { throw "Unexpected model/license origin: $($entry.url)" }
    $parentFolder = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $parentFolder)) { New-Item -ItemType Directory -Path $parentFolder | Out-Null }
    $partialPath = $destination + '.' + [guid]::NewGuid().ToString('N') + '.download'
    Write-Host "Downloading verified upstream asset $($entry.file)..."
    Invoke-WebRequest -Uri $uri.AbsoluteUri -OutFile $partialPath -UseBasicParsing
    Confirm-AiFile $partialPath $entry.sha256 $entry.size_bytes
    Move-Item -LiteralPath $partialPath -Destination $destination
}

$previousTemp = $env:TEMP
$previousTmp = $env:TMP
Push-Location -LiteralPath $appRoot
try {
    Write-Host 'FPV Sesh optional realistic AI setup'
    Write-Host 'Checking the separate .venv-ai runtime, pinned packages and model files.'
    $venvPython = Get-AiPath '.venv-ai/Scripts/python.exe'
    $lockPath = Get-AiPath 'requirements-ai-lock.txt'
    $dependencyPath = Get-AiPath 'tools/ai-python-dependencies.json'
    $modelPath = Get-AiPath 'models/real-esrgan-cuda/manifest.json'
    foreach ($requiredFile in @($lockPath, $dependencyPath, $modelPath)) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "Missing setup manifest: $requiredFile" }
    }
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if ($CheckOnly) { throw 'The optional .venv-ai runtime is missing. Run setup-ai.ps1 to install it.' }
        if (Test-Path -LiteralPath (Get-AiPath '.venv-ai')) {
            throw 'The existing .venv-ai is incomplete. This setup will not erase an existing environment.'
        }
        if ($PythonPath) {
            $basePython = (Resolve-Path -LiteralPath $PythonPath).Path
        } elseif (Test-Path -LiteralPath (Get-AiPath '.venv/Scripts/python.exe')) {
            $basePython = Get-AiPath '.venv/Scripts/python.exe'
        } else {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if (-not $pythonCommand) { throw 'Python 3.12 (64-bit) is required. Supply -PythonPath to its existing executable.' }
            $basePython = $pythonCommand.Source
        }
        & $basePython -c "import struct,sys; assert sys.platform=='win32' and sys.version_info[:2]==(3,12) and struct.calcsize('P')==8, 'Use Windows 64-bit Python 3.12'"
        if ($LASTEXITCODE -ne 0) { throw 'The optional AI lock requires Windows 64-bit Python 3.12.' }
        & $basePython -m venv (Get-AiPath '.venv-ai')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the project-local AI environment failed.' }
    }

    & $venvPython -c "import pathlib,struct,sys; assert sys.platform=='win32' and sys.version_info[:2]==(3,12) and struct.calcsize('P')==8; assert sys.prefix!=sys.base_prefix; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path('.venv-ai').resolve(), 'Unexpected environment location'"
    if ($LASTEXITCODE -ne 0) { throw 'The AI environment must be a project-local Windows 64-bit Python 3.12 venv.' }

    if (-not $CheckOnly) {
        $cachePath = Get-AiPath 'cache/ai-setup'
        $tempPath = Get-AiPath 'cache/ai-setup/temp'
        foreach ($directory in @($cachePath, $tempPath)) {
            if (-not (Test-Path -LiteralPath $directory)) { New-Item -ItemType Directory -Path $directory | Out-Null }
        }
        $env:TEMP = $tempPath
        $env:TMP = $tempPath
        $torchLock = Get-AiPath 'cache/ai-setup/torch-lock.txt'
        $otherLock = Get-AiPath 'cache/ai-setup/pypi-lock.txt'
        $pins = @(Get-Content -LiteralPath $lockPath | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') })
        foreach ($line in $pins) {
            if ($line -notmatch '^[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+ --hash=sha256:[0-9a-f]{64}$') {
                throw "Unexpected lock syntax: $line"
            }
        }
        $torchPins = @($pins | Where-Object { $_ -match '^torch==' })
        if ($torchPins.Count -ne 1 -or $torchPins[0] -notmatch '^torch==2\.9\.1\+cu128 ') {
            throw 'Expected exactly the recorded torch2.9.1+cu128 package.'
        }
        $torchPins | Set-Content -LiteralPath $torchLock -Encoding ascii
        $pins | Where-Object { $_ -notmatch '^torch==' } | Set-Content -LiteralPath $otherLock -Encoding ascii
        Write-Host 'Installing the exact tested supporting packages from PyPI...'
        & $venvPython -m pip --isolated install --disable-pip-version-check --cache-dir $cachePath --index-url 'https://pypi.org/simple' --only-binary=:all: --no-deps --require-hashes -r $otherLock
        if ($LASTEXITCODE -ne 0) { throw 'Installing the hash-locked supporting packages failed.' }
        Write-Host 'Installing the recorded CUDA PyTorch wheel from its official index...'
        & $venvPython -m pip --isolated install --disable-pip-version-check --cache-dir $cachePath --index-url 'https://download.pytorch.org/whl/cu128' --only-binary=:all: --no-deps --require-hashes -r $torchLock
        if ($LASTEXITCODE -ne 0) { throw 'Installing the hash-locked CUDA PyTorch wheel failed.' }
    }

    $verifyVersions = @'
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
import re
pins = {}
for line in Path('requirements-ai-lock.txt').read_text().splitlines():
    if not line.strip() or line.lstrip().startswith('#'):
        continue
    name, wanted = line.split()[0].split('==')
    pins[re.sub(r'[-_.]+','-',name.lower())] = wanted
bad = []
for name,wanted in pins.items():
    try:
        found = version(name)
    except PackageNotFoundError:
        found = 'missing'
    if found != wanted:
        bad.append(f'{name}: expected {wanted}, found {found}')
extra = sorted(re.sub(r'[-_.]+','-',d.metadata['Name'].lower()) for d in distributions()
               if re.sub(r'[-_.]+','-',d.metadata['Name'].lower()) not in pins)
if extra:
    bad.append('Unexpected packages: '+', '.join(extra))
if bad:
    raise SystemExit('; '.join(bad))
print(f'All {len(pins)} installed package versions match the complete AI lock.')
'@
    & $venvPython -c $verifyVersions
    if ($LASTEXITCODE -ne 0) { throw 'Installed AI packages differ from requirements-ai-lock.txt.' }
    & $venvPython -m pip --isolated check
    if ($LASTEXITCODE -ne 0) { throw 'AI package dependency consistency check failed.' }

    $dependencyManifest = Get-Content -LiteralPath $dependencyPath -Raw | ConvertFrom-Json
    foreach ($package in $dependencyManifest.packages) {
        foreach ($notice in $package.license_files) {
            Confirm-AiFile (Get-AiPath $notice.path) $notice.sha256
        }
    }
    $modelManifest = Get-Content -LiteralPath $modelPath -Raw | ConvertFrom-Json
    foreach ($entry in @($modelManifest.models) + @($modelManifest.license_artifacts)) {
        Install-AiArtifact $entry 'models/real-esrgan-cuda'
    }
    Write-Host 'All three model hashes and bundled license notices verified.'

    $verifyCuda = @'
import cv2
import numpy
import PIL
import torch
from fpvsesh.ai_models import Restorer
assert torch.__version__ == '2.9.1+cu128', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available(), 'This optional runtime requires an available CUDA GPU.'
print('CUDA12.8 runtime is available; AI adapter imports successfully. No inference was run.')
'@
    & $venvPython -c $verifyCuda
    if ($LASTEXITCODE -ne 0) { throw 'The installed AI runtime or CUDA availability check failed.' }
    Write-Host 'Optional AI setup verified. The main application environment remains separate.'
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    Pop-Location
}
