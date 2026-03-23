# KINESYS

Context-aware adaptive gesture control for Windows.

KINESYS turns hand gestures into an application-aware input layer. The same gesture can trigger different actions in Chrome, VS Code, Zoom, or any fallback profile, while also supporting personal gesture training, fatigue-aware smoothing, offline voice feedback, macros, air writing, and a live Streamlit dashboard.

## Core idea

Traditional gesture systems are static: one gesture maps to one action forever. KINESYS adds context, personalization, and runtime adaptation:

- App-context remapping: the same gesture changes behavior based on the active application.
- Two-hand control: left hand acts as Ctrl, Shift, or Alt while the right hand performs the action.
- Personal training: a 5-shot KNN model learns the user's own gesture style.
- Fatigue adaptation: smoothing increases automatically when landmark jitter suggests fatigue.
- Voice feedback: important state changes are announced offline through `pyttsx3`.
- Macro playback: gestures can trigger user-recorded action sequences.
- Write mode: air-writing with EMNIST or a floating touch-free keyboard.
- Live dashboard: Streamlit panel for feed, trainer, analytics, and macro management.

## Repository layout

```text
kinesys/
|- main.py
|- config.py
|- hand_tracker.py
|- cursor_controller.py
|- context_engine.py
|- gesture_trainer.py
|- macro_engine.py
|- fatigue_detector.py
|- analytics.py
|- air_writer.py
|- voice_feedback.py
|- check_setup.py
|- demo_playback.py
|- JUDGE_QA.md
|- PRESENTATION.md
|- profiles/
|- models/
`- ui/
```

## Tested environment

- Windows 11
- Python 3.12.4
- MediaPipe `0.10.14`
- TensorFlow `2.16.2`

The current requirements are pinned to keep native Windows compatibility stable. On Python 3.12, TensorFlow installs as `tensorflow-intel` underneath the `tensorflow` package, which is expected on native Windows.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2. Validate the environment.

```powershell
python check_setup.py
```

3. Add your ngrok token if you want a public dashboard URL.

Create `.env` in the repo root:

```env
NGROK_AUTH_TOKEN=your_token_here
```

4. Train the EMNIST model once if you want air-writing recognition instead of keyboard fallback.

```powershell
python models\train_emnist.py
```

This creates `models/emnist_cnn.h5`. If the model is missing, KINESYS still runs and falls back to the floating keyboard path in write mode.

## Running KINESYS

Start the full runtime:

```powershell
.\.venv\Scripts\python -u main.py
```

The startup sequence:

1. `check_setup.py` validates Python, files, imports, and webcam.
2. OpenCV opens the webcam at `640x480`.
3. MediaPipe Hands starts with support for both hands.
4. The Streamlit dashboard launches on `http://127.0.0.1:8501`.
5. ngrok is started if `NGROK_AUTH_TOKEN` is available.
6. The main loop begins dispatching gestures and updating shared dashboard state.

## Backup demo playback

If the webcam fails on stage, use the prerecorded playback path:

```powershell
.\.venv\Scripts\python demo_playback.py
```

By default, the script looks for:

- `demo_assets/demo_session.mp4`
- `demo_assets/demo_session.json` (optional metadata timeline)

You can also pass a custom file:

```powershell
.\.venv\Scripts\python demo_playback.py --video "D:\path\to\demo.mp4"
```

The playback script can also launch the dashboard and keep `ui/dashboard_state.json` updated so the phone/dashboard view still works during a fallback demo.

## Gesture dictionary

### Right hand

- `INDEX_POINT`: cursor mode
- `PINCH`: left click
- `PEACE_SIGN`: context action
- `TWO_FINGER_SWIPE`: scroll mode
- `THREE_FINGER_LEFT`: browser back / swipe left
- `THREE_FINGER_RIGHT`: browser forward / swipe right
- `FOUR_FINGER_SWIPE`: task/window switch
- `PINCH_ZOOM_IN`: zoom in
- `PINCH_ZOOM_OUT`: zoom out
- `CIRCLE`: macro trigger
- `CLOSED_FIST`: lock input
- `OPEN_PALM`: idle/resume
- `TWO_HANDS_X`: terminate KINESYS

### Left hand modifiers

- `OPEN_PALM_LEFT`: no modifier
- `INDEX_LEFT`: Ctrl
- `PEACE_LEFT`: Shift
- `THREE_FINGERS_LEFT`: Alt

## Profiles

App-specific mappings live in `profiles/`.

Examples:

- Chrome: `PEACE_SIGN -> new_tab`
- VS Code: `PEACE_SIGN -> run_code`
- Zoom: `PEACE_SIGN -> raise_hand`
- Default: `PEACE_SIGN -> switch_window`

If active app detection fails, KINESYS falls back safely to `profiles/default.json`.

## Safety measures

- Gesture actions require `GESTURE_HOLD_FRAMES=3` consecutive frames before firing.
- Classification confidence must exceed `ACTION_CONFIDENCE_THRESHOLD=0.75`.
- `CLOSED_FIST` locks all input immediately.
- `TWO_HANDS_X` terminates the system cleanly.
- `q` remains available as a fallback exit.
- `pyautogui.FAILSAFE = True` keeps top-left-corner escape enabled.
- Fatigue mode raises smoothing and speaks a break reminder.
- Webcam processing stays local; ngrok only exposes the dashboard.

## Current limitations

- GPU TensorFlow is not available on native Windows for this stack; use WSL2 if GPU training is required.
- Air-writing quality depends on the trained EMNIST model and lighting conditions.
- The best live experience still depends on steady framing and a clean background.

## Judge Q&A

See [JUDGE_QA.md](JUDGE_QA.md) for concise answers to likely judging questions.
