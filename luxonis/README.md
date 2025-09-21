## [install uv](https://github.com/lsy0727/uv_setting)

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
- press 'a' : Auto-save frame ON/OFF (Save every 5 frames)

https://github.com/lsy0727/depth-camera/blob/d7d66afd0ce0134d1f896fb9c8a47240c467ef1b/luxonis/collect_data.py#L305

- press 'space' : Save a frame

- press 'q' or 'esc' : close


2. depth_test.py
```
uv run depth_test.py
```
- Print depth in the terminal

<img width="1631" height="540" alt="image" src="https://github.com/user-attachments/assets/a66e81eb-6298-4bef-9168-daec6e308d3c" />

<img width="729" height="662" alt="image" src="https://github.com/user-attachments/assets/40e39f99-0d82-4982-86e9-029623b5761e" />

