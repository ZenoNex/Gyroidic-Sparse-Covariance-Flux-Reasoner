# setup_cuda_env.ps1 - Hardware and Environment Configuration for Acer Nitro 5 (GTX 1050 Ti)
Set-Location "D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner"

# 1. Ensure scratch directories on D: drive (120 GB free)
if (-not (Test-Path "D:\tmp")) { New-Item -ItemType Directory -Force -Path "D:\tmp" | Out-Null }
if (-not (Test-Path "D:\torch_extensions")) { New-Item -ItemType Directory -Force -Path "D:\torch_extensions" | Out-Null }

# 2. Redirect temp and compilation scratch away from 10 GB C: drive to D:
$env:TEMP = "D:\tmp"
$env:TMP = "D:\tmp"
$env:TORCH_EXTENSIONS_DIR = "D:\torch_extensions"

# 3. Hardware guards for Pascal sm_61 and 8GB/16GB laptop RAM
$env:TORCH_CUDA_ARCH_LIST = "6.1"
$env:MAX_JOBS = "2"
$env:DISTUTILS_USE_SDK = "1"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"

# 4. Bind CUDA path if installed
if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6") {
    $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    $env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
}

Write-Host "CUDA build environment configured successfully on D: drive."
Write-Host "TEMP / TMP redirected to: $env:TEMP"
Write-Host "Target Architecture: sm_$($env:TORCH_CUDA_ARCH_LIST)"
Write-Host "Max Compiler Jobs: $env:MAX_JOBS"

