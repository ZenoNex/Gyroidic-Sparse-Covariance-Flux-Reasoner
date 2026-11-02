Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner"
WshShell.Environment("PROCESS")("NON_INTERACTIVE") = "1"
WshShell.Environment("PROCESS")("PYTHONPATH") = "."
WshShell.Run """D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\.venv\Scripts\python.exe"" -u hybrid_backend.py", 0, False
