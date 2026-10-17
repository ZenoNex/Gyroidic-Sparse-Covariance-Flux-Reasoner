Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoRoot = fso.GetParentFolderName(scriptDir)
pythonExe = repoRoot & "\.venv\Scripts\python.exe"

If Not fso.FileExists(pythonExe) Then
    pythonExe = "python.exe"
End If

WshShell.CurrentDirectory = repoRoot
WshShell.Environment("PROCESS")("NON_INTERACTIVE") = "1"
WshShell.Environment("PROCESS")("PYTHONPATH") = "."
WshShell.Run """" & pythonExe & """ -u hybrid_backend.py", 0, False
