# Environment setup (PowerShell)

1. Verify the installed Python version
```
uv python list
```
-> Python version and installation path are shown

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


# Manual

1. collect_data.py
```
uv run collect_data.py
```
press 'a' : Auto-save frame ON/OFF (Save every 5 frames)

https://github.com/lsy0727/depth-camera/blob/04ec97d85b34c6644cb1f83fa27ec1b7ff86b7d1/luxonis/collect_data.py#L305C5-L305C91

press 'space' : Save a frame

press 'q' or 'esc' : close


2. depth_test.py
```
uv run depth_test.py
```
Print depth in the terminal
