[CmdletBinding()]
param(
    [string]$PythonPath = '',
    [switch]$Upgrade,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$rootPrefix = $appRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
. (Join-Path $PSScriptRoot 'setup-common.ps1')
$setupGuard = $null

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
    $hasher = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($path)
    try { $actualHash = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hasher.Dispose() }
    if ($actualHash -ne $sha256) {
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
    # FileStream treats brackets and other wildcard characters literally.
    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(10)
    $sourceStream = $null
    $targetStream = $null
    try {
        $sourceStream = $client.GetStreamAsync($uri.AbsoluteUri).GetAwaiter().GetResult()
        $targetStream = [IO.File]::Open($partialPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        $sourceStream.CopyTo($targetStream)
        $targetStream.Dispose()
        $targetStream = $null
        Confirm-AiFile $partialPath $entry.sha256 $entry.size_bytes
        Move-Item -LiteralPath $partialPath -Destination $destination
    } finally {
        if ($targetStream) { $targetStream.Dispose() }
        if ($sourceStream) { $sourceStream.Dispose() }
        $client.Dispose()
        if (Test-Path -LiteralPath $partialPath -PathType Leaf) { Remove-Item -LiteralPath $partialPath }
    }
}

$previousTemp = $env:TEMP
$previousTmp = $env:TMP
Push-Location -LiteralPath $appRoot
try {
    if (-not $CheckOnly) { $setupGuard = Enter-FpvSetupLock -AppRoot $appRoot }
    if ($Upgrade -and $CheckOnly) { throw 'Use -Upgrade to install updates or -CheckOnly to inspect, not both.' }
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
            if (-not $pythonCommand) { throw 'Python 3.12 or 3.13 (64-bit) is required; 3.13 is recommended. Supply -PythonPath to its executable.' }
            $basePython = $pythonCommand.Source
        }
        & $basePython -c "import struct,sys; assert sys.platform=='win32' and sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P')==8, 'Use Windows 64-bit Python 3.12 or 3.13'"
        if ($LASTEXITCODE -ne 0) { throw 'The optional AI lock requires Windows 64-bit Python 3.12 or 3.13.' }
        & $basePython -m venv (Get-AiPath '.venv-ai')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the project-local AI environment failed.' }
    }

    & $venvPython -c "import pathlib,struct,sys; assert sys.platform=='win32' and sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P')==8; assert sys.prefix!=sys.base_prefix; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path('.venv-ai').resolve(), 'Unexpected environment location'"
    if ($LASTEXITCODE -ne 0) { throw 'The AI environment must be a project-local Windows 64-bit Python 3.12 or 3.13 venv.' }

    if (-not $CheckOnly) {
        $checkUpgrade = @'
from importlib.metadata import distributions
from pathlib import Path
import re, sys
normalize = lambda name: re.sub(r'[-_.]+','-',name.lower())
expected = {}
for filename in ('requirements-ai-lock.txt','requirements-video-lock.txt'):
    for line in Path(filename).read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name,wanted = line.split()[0].split('==')
            expected[normalize(name)] = wanted
installed = {normalize(d.metadata['Name']):d.version for d in distributions()}
extra = sorted(set(installed)-set(expected))
if extra:
    raise SystemExit('Unexpected optional packages: '+', '.join(extra)+'. Use a fresh application copy.')
changed = [name for name,version in installed.items() if name not in ('pip','setuptools') and expected[name]!=version]
if changed and sys.argv[1]!='upgrade':
    raise SystemExit('The optional runtime uses older or different pins. Run setup-ai.ps1 -Upgrade to update its declared packages, or use a fresh application copy. No packages changed.')
'@
        & $venvPython -B -c $checkUpgrade $(if ($Upgrade) { 'upgrade' } else { 'install' })
        if ($LASTEXITCODE -ne 0) { throw 'Optional runtime preflight stopped before package installation.' }
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
            if ($line -notmatch '^[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+( --hash=sha256:[0-9a-f]{64})+$') {
                throw "Unexpected lock syntax: $line"
            }
        }
        $torchPins = @($pins | Where-Object { $_ -match '^torch==' })
        if ($torchPins.Count -ne 1 -or $torchPins[0] -notmatch '^torch==2\.14\.0\+cu126 ') {
            throw 'Expected exactly the recorded torch2.14.0+cu126 package.'
        }
        $torchPins | Set-Content -LiteralPath $torchLock -Encoding ascii
        $pins | Where-Object { $_ -notmatch '^torch==' } | Set-Content -LiteralPath $otherLock -Encoding ascii
        $repairArgs = @()
        if ($Upgrade) { $repairArgs = @('--force-reinstall') }
        Write-Host 'Installing the exact tested supporting packages from PyPI...'
        & $venvPython -m pip --isolated install --disable-pip-version-check --cache-dir $cachePath --index-url 'https://pypi.org/simple' --only-binary=:all: --no-deps --require-hashes @repairArgs -r $otherLock
        if ($LASTEXITCODE -ne 0) { throw 'Installing the hash-locked supporting packages failed.' }
        Write-Host 'Installing the recorded CUDA PyTorch wheel from its official index...'
        & $venvPython -m pip --isolated install --disable-pip-version-check --cache-dir $cachePath --index-url 'https://download.pytorch.org/whl/cu126' --only-binary=:all: --no-deps --require-hashes @repairArgs -r $torchLock
        if ($LASTEXITCODE -ne 0) { throw 'Installing the hash-locked CUDA PyTorch wheel failed.' }
        if ($Upgrade -and (Test-Path -LiteralPath (Get-AiPath '.venv-ai/Lib/site-packages/transformers/models/qwen3_vl') -PathType Container)) {
            Write-Host 'Updating the already-installed video extension to its matching release pins...'
            & (Get-AiPath 'setup-video.ps1') -Upgrade -ParentSetupGuard $setupGuard
            if (-not $?) { throw 'The optional video extension upgrade did not complete.' }
        }
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
installed = {re.sub(r'[-_.]+','-',d.metadata['Name'].lower()) for d in distributions()}
video_lock = Path('requirements-video-lock.txt')
if video_lock.is_file():
    video_pins = {}
    for line in video_lock.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, wanted = line.split()[0].split('==')
        video_pins[re.sub(r'[-_.]+','-',name.lower())] = wanted
    if (set(video_pins) - set(pins)) & installed:
        if any(name in pins and pins[name] != wanted for name, wanted in video_pins.items()):
            raise SystemExit('Optional video lock conflicts with the existing AI runtime')
        pins.update(video_pins)
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
from fpvsesh.runtime_dlls import prepare_torch_dlls
prepare_torch_dlls()
import torch
from fpvsesh.ai_models import Restorer
assert torch.__version__ == '2.14.0+cu126', torch.__version__
assert torch.version.cuda == '12.6', torch.version.cuda
print('Pinned optional runtime and AI adapter import successfully. No inference was run.')
print('Compatible CUDA GPU available: '+str(torch.cuda.is_available()))
if not torch.cuda.is_available():
    print('Scene context can use CPU. AI detail and Qwen video inference require a compatible NVIDIA GPU.')
'@
    & $venvPython -c $verifyCuda
    if ($LASTEXITCODE -ne 0) { throw 'The installed AI runtime import check failed.' }
    Write-Host 'Optional AI setup verified. The main application environment remains separate.'
} finally {
    if ($setupGuard) { $setupGuard.Dispose() }
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    Pop-Location
}
