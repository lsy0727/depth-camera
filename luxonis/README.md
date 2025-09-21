# Environment setup (PowerShell)

1. Verify the installed Python version
```
uv python list
```
-> You can view the various Python versions and their installation paths

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/01a41a52-bb14-4729-b02d-c4d7b182b16f" />

2. install python
```
uv python install 3.10.8
```

3. Create a virtual environment
```
uv venv .venv --python 3.10.8
```

4. Activate the virtual environment
```
.venv/Scripts/activate
```

5. Install the required packages for the project
```
uv pip install -r requirements.txt
```

6. Execute
```
uv run collect_data.py
```
