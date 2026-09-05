[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$rootPrefix = $appRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$pythonPath = Join-Path $appRoot '.venv-ai/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Run setup-ai.ps1 first to install the separate CUDA runtime, then run setup-video.ps1.'
}

function Get-VideoPath([string]$relative) {
    if ([IO.Path]::IsPathRooted($relative)) { throw 'Expected an application-relative path.' }
    $absolute = [IO.Path]::GetFullPath((Join-Path $appRoot $relative))
    if (-not $absolute.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Video asset path escapes this application.' }
    return $absolute
}

function Confirm-VideoFile([string]$path, [string]$sha256, [long]$size = -1) {
    if ($sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw 'Video asset checksum is missing.' }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing video asset: $path" }
    if ($size -ge 0 -and (Get-Item -LiteralPath $path).Length -ne $size) { throw "Video asset size mismatch: $path" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $sha256) { throw "Video asset checksum mismatch: $path" }
}

$checkVersions = @'
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
import re, struct, sys
assert sys.platform == 'win32' and sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8
assert Path(sys.prefix).resolve() == Path('.venv-ai').resolve() and sys.prefix != sys.base_prefix
normalize = lambda name: re.sub(r'[-_.]+', '-', name.lower())
expected = {}
for filename in ('requirements-ai-lock.txt', 'requirements-video-lock.txt'):
    for line in Path(filename).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+ --hash=sha256:[0-9a-f]{64}', line):
            raise SystemExit('Unexpected dependency lock syntax')
        name, wanted = line.split()[0].split('==')
        name = normalize(name)
        if name in expected and expected[name] != wanted:
            raise SystemExit('Video lock conflicts with an existing AI pin: '+name)
        expected[name] = wanted
        try:
            actual = version(name)
        except PackageNotFoundError:
            if filename == 'requirements-video-lock.txt' and sys.argv[1] == 'allow-missing':
                continue
            raise SystemExit('Missing package: '+name)
        if actual != wanted:
            raise SystemExit(f'{name}: expected {wanted}, found {actual}; existing packages are not replaced')
extra = sorted(normalize(d.metadata['Name']) for d in distributions() if normalize(d.metadata['Name']) not in expected)
if extra:
    raise SystemExit('Unexpected packages in the AI runtime: '+', '.join(extra))
print('Base AI versions preserved; video dependency pins verified.')
'@

$downloadAsset = @'
import hashlib
from pathlib import Path
import sys, urllib.request, uuid
url, target, checksum, size = sys.argv[1:]
destination = Path(target)
temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.download')
digest = hashlib.sha256()
total = 0
try:
    request = urllib.request.Request(url, headers={'User-Agent':'FPV-Sesh-local-setup/0.3'})
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open('xb') as stream:
        while data := source.read(4*1024*1024):
            total += len(data)
            if total > int(size):
                raise ValueError('Download exceeds the recorded size')
            digest.update(data)
            stream.write(data)
    if total != int(size) or digest.hexdigest() != checksum:
        raise ValueError('Downloaded asset failed its size/SHA256 check')
    temporary.replace(destination)
finally:
    temporary.unlink(missing_ok=True)
'@

Push-Location -LiteralPath $appRoot
try {
    Write-Host 'FPV Sesh optional internet-pretrained video understanding'
    Write-Host 'Official Qwen3-VL-2B: approximately 4.27 GB once. Inference stays on this computer.'
    & $pythonPath -B -c $checkVersions $(if ($CheckOnly) { 'complete' } else { 'allow-missing' })
    if ($LASTEXITCODE -ne 0) { throw 'Existing runtime differs from the recorded base/video pins.' }
    $dependencyManifest = Get-Content -LiteralPath (Get-VideoPath 'tools/video-python-dependencies.json') -Raw | ConvertFrom-Json
    foreach ($package in $dependencyManifest.packages) {
        foreach ($notice in $package.license_files) {
            Confirm-VideoFile (Get-VideoPath $notice.path) $notice.sha256
        }
    }
    if (-not $CheckOnly) {
        $cacheFolder = Get-VideoPath 'cache/video-setup'
        if (-not (Test-Path -LiteralPath $cacheFolder)) { New-Item -ItemType Directory -Path $cacheFolder | Out-Null }
        $pins = @(Get-Content -LiteralPath (Get-VideoPath 'requirements-video-lock.txt') | Where-Object { $_.Trim() -and -not $_.StartsWith('#') })
        $visionPins = @($pins | Where-Object { $_ -match '^torchvision==' })
        if ($visionPins.Count -ne 1 -or $visionPins[0] -notmatch '^torchvision==0\.24\.1\+cu128 ') { throw 'Expected the CUDA torchvision package matching PyTorch 2.9.1.' }
        $pypiLock = Join-Path $cacheFolder 'pypi-lock.txt'
        $visionLock = Join-Path $cacheFolder 'torchvision-lock.txt'
        $pins | Where-Object { $_ -notmatch '^torchvision==' } | Set-Content -LiteralPath $pypiLock -Encoding ascii
        $visionPins | Set-Content -LiteralPath $visionLock -Encoding ascii
        & $pythonPath -m pip --isolated install --disable-pip-version-check --cache-dir $cacheFolder --index-url 'https://pypi.org/simple' --only-binary=:all: --no-deps --require-hashes -r $pypiLock
        if ($LASTEXITCODE -ne 0) { throw 'Video supporting package installation failed.' }
        & $pythonPath -m pip --isolated install --disable-pip-version-check --cache-dir $cacheFolder --index-url 'https://download.pytorch.org/whl/cu128' --only-binary=:all: --no-deps --require-hashes -r $visionLock
        if ($LASTEXITCODE -ne 0) { throw 'CUDA torchvision installation failed.' }
    }
    & $pythonPath -B -c $checkVersions 'complete'
    if ($LASTEXITCODE -ne 0) { throw 'Installed video packages failed verification.' }
    & $pythonPath -m pip --isolated check
    if ($LASTEXITCODE -ne 0) { throw 'Video dependency compatibility check failed.' }

    $modelFolder = Get-VideoPath 'models/qwen3-vl-2b'
    $manifest = Get-Content -LiteralPath (Join-Path $modelFolder 'manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.repository -ne 'Qwen/Qwen3-VL-2B-Instruct' -or $manifest.revision -notmatch '^[0-9a-f]{40}$') { throw 'Unexpected pretrained model manifest.' }
    foreach ($entry in $manifest.assets) {
        if ([IO.Path]::GetFileName($entry.file) -ne $entry.file -or $entry.file -match '\.(py|bin|pkl|pickle|pt|pth)$') { throw 'Model manifest contains an unsupported asset.' }
        $destination = Join-Path $modelFolder $entry.file
        if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
            if ($CheckOnly) { throw "Missing $($entry.file). Run setup-video.ps1 to download it." }
            $uri = [uri]$entry.url
            $expectedPrefix = '/Qwen/Qwen3-VL-2B-Instruct/resolve/' + $manifest.revision + '/'
            $modelOrigin = $uri.Scheme -eq 'https' -and $uri.Host -eq 'huggingface.co' -and $uri.AbsolutePath -eq ($expectedPrefix + $entry.file)
            $licenseOrigin = $uri.Scheme -eq 'https' -and $uri.Host -eq 'raw.githubusercontent.com' -and $uri.AbsolutePath -match '^/QwenLM/Qwen3-VL/[0-9a-f]{40}/LICENSE$' -and $entry.file -eq 'LICENSE.txt'
            if (-not ($modelOrigin -or $licenseOrigin)) { throw 'Unexpected video model download origin.' }
            Write-Host "Downloading verified model asset $($entry.file)..."
            & $pythonPath -B -c $downloadAsset $uri.AbsoluteUri $destination $entry.sha256 ([string]$entry.size_bytes)
            if ($LASTEXITCODE -ne 0) { throw 'Video model download failed its verification.' }
        }
        Confirm-VideoFile $destination $entry.sha256 $entry.size_bytes
    }
    $checkImports = @'
import torch, torchvision, transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
assert torch.__version__ == '2.9.1+cu128'
assert torchvision.__version__ == '0.24.1+cu128'
assert transformers.__version__ == '4.57.6'
assert torch.cuda.is_available(), 'An available CUDA GPU is required for this optional video model'
processor = AutoProcessor.from_pretrained('models/qwen3-vl-2b', local_files_only=True, trust_remote_code=False)
assert processor.__class__.__name__ == 'Qwen3VLProcessor'
print('Video model assets, local processor, and CUDA imports verified. No video inference or footage upload was performed.')
'@
    & $pythonPath -B -c $checkImports
    if ($LASTEXITCODE -ne 0) { throw 'Video processor or CUDA import verification failed.' }
} finally {
    Pop-Location
}
