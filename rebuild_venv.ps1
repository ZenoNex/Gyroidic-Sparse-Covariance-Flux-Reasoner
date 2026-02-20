# Rebuild VENV script
Write-Host "Removing existing .venv..."
if (Test-Path .venv) {
    Remove-Item -Recurse -Force .venv -ErrorAction Stop
}
Write-Host "Creating new .venv..."
python -m venv .venv
Write-Host "Installing requirements..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Done. Please restart your shell or run '.\.venv\Scripts\Activate.ps1' to activate."
