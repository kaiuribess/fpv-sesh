[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$modelFolder = Join-Path $appRoot 'models/places365'
$pythonPath = Join-Path $appRoot '.venv-ai/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'The optional Torch runtime is missing. Run setup-ai.ps1 first, then setup-vision.ps1. Scene mapping also works without this optional model using motion estimates.'
}
$manifestPath = Join-Path $modelFolder 'manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

function Confirm-SceneFile([string]$path, $entry) {
    if ($entry.sha256 -notmatch '^[0-9a-f]{64}$') { throw 'Missing expected model checksum.' }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing scene asset: $($entry.file)" }
    if ((Get-Item -LiteralPath $path).Length -ne $entry.size_bytes) { throw "Scene asset size mismatch: $($entry.file)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Scene asset SHA256 mismatch: $($entry.file)" }
}

Write-Host 'Optional Places365 scene context: approximately 46 MB. No source footage is uploaded.'
Write-Host 'Pretrained weights: CC BY attribution (upstream does not specify version). See models/places365/MODEL-LICENSE.md.'
foreach ($entry in $manifest.assets) {
    if ([IO.Path]::GetFileName($entry.file) -ne $entry.file) { throw 'Scene manifest must contain simple filenames.' }
    $destination = Join-Path $modelFolder $entry.file
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        Confirm-SceneFile $destination $entry
        continue
    }
    if ($CheckOnly) { throw "Missing $($entry.file). Run setup-vision.ps1 without -CheckOnly." }
    $uri = [uri]$entry.url
    $officialModel = $uri.AbsoluteUri -eq 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
    $officialText = $uri.Scheme -eq 'https' -and $uri.Host -eq 'raw.githubusercontent.com' -and (
        $uri.AbsolutePath.StartsWith('/CSAILVision/places365/') -or $uri.AbsolutePath.StartsWith('/pytorch/vision/'))
    if (-not ($officialModel -or $officialText)) { throw "Unexpected scene asset origin: $($entry.url)" }
    $partialPath = $destination + '.' + [guid]::NewGuid().ToString('N') + '.download'
    try {
        Write-Host "Downloading $($entry.file)..."
        Invoke-WebRequest -Uri $uri.AbsoluteUri -OutFile $partialPath -UseBasicParsing
        Confirm-SceneFile $partialPath $entry
        Move-Item -LiteralPath $partialPath -Destination $destination
    } finally {
        if (Test-Path -LiteralPath $partialPath -PathType Leaf) { Remove-Item -LiteralPath $partialPath }
    }
}
Push-Location -LiteralPath $appRoot
try {
    & $pythonPath -B -c "from fpvsesh.vision_models import SceneModel; import numpy as np; m=SceneModel(); r=m.predict([np.zeros((224,224,3),dtype=np.uint8)]); assert len(r)==1; print('Scene inference ready on '+str(m.device)+'. Scene estimates do not identify named FPV tricks.')"
    if ($LASTEXITCODE -ne 0) { throw 'Scene model safe-load or inference check failed.' }
} finally {
    Pop-Location
}
