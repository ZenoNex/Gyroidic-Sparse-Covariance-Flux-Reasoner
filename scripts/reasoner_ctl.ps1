<#
.SYNOPSIS
    Gyroidic Sparse Covariance Flux Reasoner - Windows 10/11 Service Controller
.DESCRIPTION
    Provides start, graceful stop (with emergency fossilization), status,
    and secure computer startup configuration for Windows 10 & 11.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("Start", "Stop", "Restart", "Status", "EnableAutostart", "DisableAutostart")]
    [string]$Action = "Status"
)

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$PidFile = Join-Path $RepoRoot ".backend.pid"
$LogFile = Join-Path $RepoRoot "server_startup.log"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python.exe"
}

$StartupFolder = [System.IO.Path]::Combine($env:APPDATA, "Microsoft\Windows\Start Menu\Programs\Startup")
$StartupVbs = Join-Path $StartupFolder "GyroidicReasonerAutostart.vbs"
$SilentLauncherVbs = Join-Path $PSScriptRoot "start_reasoner_silent.vbs"

function Get-BackendPid {
    if (Test-Path $PidFile) {
        $raw = (Get-Content -Path $PidFile -Raw -ErrorAction SilentlyContinue).Trim()
        if ($raw -match '^\d+$') {
            return [int]$raw
        }
    }
    return $null
}

function Test-BackendRunning {
    $targetPid = Get-BackendPid
    if ($targetPid) {
        $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        if ($proc) {
            return $true
        }
    }
    # Fallback to HTTP ping
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ping" -Method Get -TimeoutSec 1 -ErrorAction Stop
        if ($resp.status -eq "ok") {
            return $true
        }
    }
    catch {}
    return $false
}

function Start-Reasoner {
    if (Test-BackendRunning) {
        $p = Get-BackendPid
        Write-Host "[STATUS] Reasoner is already running (PID: $p)." -ForegroundColor Yellow
        return
    }

    Write-Host "[START] Launching Gyroidic Flux Reasoner (Non-Interactive)..." -ForegroundColor Cyan
    
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $PythonExe
    $startInfo.Arguments = "-u hybrid_backend.py"
    $startInfo.WorkingDirectory = $RepoRoot
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true

    # Set headless environment variables
    $startInfo.EnvironmentVariables["NON_INTERACTIVE"] = "1"
    $startInfo.EnvironmentVariables["PYTHONPATH"] = "."

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo

    # Redirect logging
    $fileStream = [System.IO.File]::Open($LogFile, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $writer = New-Object System.IO.StreamWriter($fileStream)
    $writer.AutoFlush = $true

    $outHandler = {
        param($sender, $e)
        if ($e.Data) { $writer.WriteLine($e.Data) }
    }

    $process.add_OutputDataReceived($outHandler)
    $process.add_ErrorDataReceived($outHandler)

    $null = $process.Start()
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()

    Set-Content -Path $PidFile -Value $process.Id -Encoding ASCII

    Write-Host "[START] Process started with PID: $($process.Id)" -ForegroundColor Green
    Write-Host "[START] Output streaming to: $LogFile" -ForegroundColor Gray
    Write-Host "[START] Waiting for HTTP readiness at http://127.0.0.1:8000..." -ForegroundColor Cyan

    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ping" -Method Get -TimeoutSec 1 -ErrorAction Stop
            if ($resp.status -eq "ok") {
                Write-Host "[OK] Reasoner online and ready at http://127.0.0.1:8000" -ForegroundColor Green
                return
            }
        }
        catch {}
    }

    Write-Host "[WARN] Process launched, but HTTP endpoint hasn't responded yet. Review $LogFile" -ForegroundColor Yellow
}

function Stop-Reasoner {
    Write-Host "[STOP] Initiating graceful shutdown for Gyroidic Reasoner..." -ForegroundColor Cyan

    $stopped = $false
    try {
        $res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/shutdown" -Method Post -TimeoutSec 3 -ErrorAction Stop
        Write-Host "[STOP] API shutdown signal acknowledged. Awaiting clean fossilization save..." -ForegroundColor Yellow
    }
    catch {
        Write-Host "[STOP] API unreachable; checking PID for shutdown..." -ForegroundColor Gray
    }

    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 1
        if (-not (Test-BackendRunning)) {
            $stopped = $true
            break
        }
    }

    if (-not $stopped) {
        $p = Get-BackendPid
        if ($p) {
            Write-Host "[WARN] Process did not exit in 20s. Enforcing taskkill on PID $p..." -ForegroundColor Yellow
            taskkill /PID $p /F 2>$null
        }
    }

    if (Test-Path $PidFile) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
    Write-Host "[OK] Reasoner stopped." -ForegroundColor Green
}

function Get-ReasonerStatus {
    $running = Test-BackendRunning
    $targetPid = Get-BackendPid
    if ($running) {
        Write-Host "[STATUS] ONLINE" -ForegroundColor Green
        Write-Host "   PID:  $targetPid"
        Write-Host "   HTTP: http://127.0.0.1:8000"
        try {
            $status = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/status" -Method Get -TimeoutSec 2
            Write-Host "   State: $($status.status)"
        }
        catch {
            Write-Host "   State: Starting/Responsive"
        }
    }
    else {
        Write-Host "[STATUS] OFFLINE" -ForegroundColor Red
        if (Test-Path $PidFile) {
            Write-Host "   Stale PID file detected ($targetPid is not running)." -ForegroundColor Yellow
        }
    }

    $autostartActive = Test-Path $StartupVbs
    Write-Host "   Autostart (Startup Folder): $(if ($autostartActive) { 'ENABLED' } else { 'DISABLED' })"
}

function Enable-Autostart {
    # Generate the silent VBS launcher
    $vbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "$RepoRoot"
WshShell.Environment("PROCESS")("NON_INTERACTIVE") = "1"
WshShell.Environment("PROCESS")("PYTHONPATH") = "."
WshShell.Run """$PythonExe"" -u hybrid_backend.py", 0, False
"@
    Set-Content -Path $StartupVbs -Value $vbsContent -Encoding ASCII
    Set-Content -Path $SilentLauncherVbs -Value $vbsContent -Encoding ASCII

    Write-Host "[OK] Windows 10/11 Autostart successfully configured:" -ForegroundColor Green
    Write-Host "     Location: $StartupVbs" -ForegroundColor White
    Write-Host "     The Reasoner will now start silently in the background whenever you log into Windows." -ForegroundColor Gray
    Write-Host "     To turn off anytime, run: .\scripts\reasoner_ctl.ps1 -DisableAutostart" -ForegroundColor Cyan
}

function Disable-Autostart {
    if (Test-Path $StartupVbs) {
        Remove-Item $StartupVbs -Force
        Write-Host "[OK] Windows Startup item removed: $StartupVbs" -ForegroundColor Green
    }
    else {
        Write-Host "[INFO] No autostart item found in Windows Startup folder." -ForegroundColor Gray
    }
    Write-Host "[OK] Autostart disabled on Windows 10/11." -ForegroundColor Green
}

switch ($Action) {
    "Start" { Start-Reasoner }
    "Stop" { Stop-Reasoner }
    "Restart" { Stop-Reasoner; Start-Sleep -Seconds 2; Start-Reasoner }
    "Status" { Get-ReasonerStatus }
    "EnableAutostart" { Enable-Autostart }
    "DisableAutostart" { Disable-Autostart }
}
