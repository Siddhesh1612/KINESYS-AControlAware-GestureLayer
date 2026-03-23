"""Streamlit dashboard for the KINESYS gesture runtime."""

from __future__ import annotations

import base64
import json
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics import AnalyticsTracker
from config import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DASHBOARD_REFRESH_MS,
    DASHBOARD_STATE_AUTHKEY,
    DASHBOARD_STATE_ENV_AUTHKEY,
    DASHBOARD_STATE_ENV_HOST,
    DASHBOARD_STATE_ENV_PORT,
    DASHBOARD_STATE_HOST,
    DASHBOARD_STATE_PORT,
    DASHBOARD_STATE_SNAPSHOT,
    KNN_SAMPLES_REQUIRED,
    MODIFIER_NONE,
    STATE_CURSOR,
    STATE_IDLE,
    STATE_LOCK,
    STATE_MACRO,
    STATE_SCROLL,
    STATE_TERMINATED,
    STATE_WRITE,
    TRAINER_SUPPORTED_GESTURES,
)
from gesture_trainer import GestureTrainer
from macro_engine import MacroEngine


class DashboardStateClient(BaseManager):
    """Client-side manager used by the Streamlit dashboard process."""


DashboardStateClient.register("get_shared_state")

STATE_BADGE_COLORS = {
    STATE_CURSOR: "#0F6E56",
    STATE_WRITE: "#3C3489",
    STATE_SCROLL: "#633806",
    STATE_MACRO: "#1A4A6B",
    STATE_LOCK: "#712B13",
    STATE_IDLE: "#888780",
    STATE_TERMINATED: "#791F1F",
}
STATE_BADGE_DEFAULT = "#6E7781"
EMPTY_TEXT = ""
EMPTY_URL = ""
DEFAULT_MACRO_STEPS = json.dumps(
    [
        {"type": "hotkey", "keys": ["ctrl", "s"]},
        {"type": "sleep", "seconds": 0.2},
        {"type": "press", "key": "f5"},
    ],
    indent=2,
)
LIVE_FEED_HELP = "Live frame, gesture state, and active app from the runtime."
TRAINER_HELP = "Record 5 live samples in front of the camera, then train the personal KNN."
ANALYTICS_HELP = "Session analytics loaded from analytics.json."
MACRO_HELP = "Save, delete, or test gesture-triggered macro definitions."


@st.cache_resource(show_spinner=False)
def connect_shared_state() -> Any | None:
    """Connect to the shared-state bridge started by the KINESYS runtime."""

    state_host = os.getenv(DASHBOARD_STATE_ENV_HOST, DASHBOARD_STATE_HOST)
    state_port = int(os.getenv(DASHBOARD_STATE_ENV_PORT, str(DASHBOARD_STATE_PORT)))
    authkey = os.getenv(DASHBOARD_STATE_ENV_AUTHKEY, DASHBOARD_STATE_AUTHKEY).encode("utf-8")

    try:
        manager = DashboardStateClient(address=(state_host, state_port), authkey=authkey)
        manager.connect()
        return manager.get_shared_state()
    except Exception:
        return None


def load_snapshot_state() -> dict[str, Any]:
    """Load the latest JSON snapshot written by the runtime as a fallback."""

    snapshot_path = Path(DASHBOARD_STATE_SNAPSHOT)
    if not snapshot_path.exists():
        return {}

    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_shared_value(shared_state: Any | None, key: str, default: Any) -> Any:
    """Read one shared-state value with a safe default fallback."""

    if shared_state is None:
        return default
    try:
        return shared_state.get(key, default)
    except Exception:
        return default


def get_state_value(shared_state: Any | None, snapshot_state: dict[str, Any], key: str, default: Any) -> Any:
    """Read one state value from the bridge first, then the JSON snapshot fallback."""

    shared_value = get_shared_value(shared_state, key, default)
    if shared_state is not None:
        return shared_value
    return snapshot_state.get(key, default)


def increment_shared_counter(shared_state: Any, key: str) -> None:
    """Increment one integer command counter stored in the shared state."""

    current_value = int(get_shared_value(shared_state, key, 0))
    shared_state[key] = current_value + 1


def build_state_badge(label: str) -> str:
    """Return HTML for the color-coded gesture-state badge."""

    color = STATE_BADGE_COLORS.get(label, STATE_BADGE_DEFAULT)
    return (
        f"<div style='display:inline-block;padding:0.4rem 0.8rem;border-radius:999px;"
        f"background:{color};color:white;font-weight:700;letter-spacing:0.03em'>{label}</div>"
    )


def load_analytics_tracker() -> AnalyticsTracker:
    """Create a fresh analytics tracker view over the current analytics file."""

    return AnalyticsTracker()


def load_macro_engine() -> MacroEngine:
    """Create a fresh macro engine view over the current macros file."""

    return MacroEngine()


def load_trainer() -> GestureTrainer:
    """Create a fresh trainer view over the current personal model file."""

    return GestureTrainer()


def render_sidebar(shared_state: Any | None, snapshot_state: dict[str, Any]) -> None:
    """Render the dashboard sidebar with live runtime indicators."""

    with st.sidebar:
        st.header("KINESYS")
        st.caption("Context-aware adaptive gesture OS layer")

        fps_value = float(get_state_value(shared_state, snapshot_state, "fps", 0.0))
        fatigue_level = float(get_state_value(shared_state, snapshot_state, "fatigue_level", 0.0))
        modifier_active = get_state_value(shared_state, snapshot_state, "modifier_active", MODIFIER_NONE)
        active_profile = str(get_state_value(shared_state, snapshot_state, "active_profile", "default"))
        ngrok_url = str(get_state_value(shared_state, snapshot_state, "ngrok_url", EMPTY_URL))
        two_hand_mode = bool(get_state_value(shared_state, snapshot_state, "two_hand_mode", False))

        st.metric("FPS", f"{fps_value:.1f}")
        st.progress(min(max(fatigue_level, 0.0), 1.0), text=f"Fatigue: {fatigue_level:.2f}")
        st.write(f"Modifier: `{modifier_active or 'none'}`")
        st.write(f"Profile: `{active_profile}`")
        st.write(f"Two-hand mode: `{'ON' if two_hand_mode else 'OFF'}`")
        if ngrok_url:
            st.code(ngrok_url, language=None)
        else:
            st.caption("Ngrok URL unavailable")
        st.caption("Termination gesture: both hands crossed in an X")


def render_live_panel(shared_state: Any | None, snapshot_state: dict[str, Any]) -> None:
    """Render the live feed, active app, and current gesture badge."""

    st.subheader("Panel 1  Live Feed")
    st.caption(LIVE_FEED_HELP)

    gesture_state = str(get_state_value(shared_state, snapshot_state, "gesture_state", STATE_IDLE))
    active_app = str(get_state_value(shared_state, snapshot_state, "active_app", "Waiting"))
    active_profile = str(get_state_value(shared_state, snapshot_state, "active_profile", "default"))
    confidence = float(get_state_value(shared_state, snapshot_state, "confidence", 0.0))
    modifier_active = get_state_value(shared_state, snapshot_state, "modifier_active", MODIFIER_NONE)
    frame_b64 = str(get_state_value(shared_state, snapshot_state, "frame_b64", EMPTY_TEXT))

    badge_column, app_column = st.columns([1, 2])
    with badge_column:
        st.markdown(build_state_badge(gesture_state), unsafe_allow_html=True)
    with app_column:
        st.write(f"App: `{active_app}`")
        st.write(f"Profile: `{active_profile}`")
        st.write(f"Modifier: `{modifier_active or 'none'}`")
        st.write(f"Confidence: `{confidence:.2f}`")

    if frame_b64:
        try:
            frame_bytes = base64.b64decode(frame_b64)
            st.image(frame_bytes, use_container_width=True)
        except Exception as exc:
            st.warning(f"Failed to decode live frame: {exc}")
    else:
        st.info("Waiting for live webcam frames from main.py.")


def render_trainer_panel(shared_state: Any | None, snapshot_state: dict[str, Any]) -> None:
    """Render the personal gesture training controls."""

    st.subheader("Panel 2  Gesture Trainer")
    st.caption(TRAINER_HELP)

    trainer = load_trainer()
    selected_gesture = st.selectbox("Gesture", TRAINER_SUPPORTED_GESTURES, key="trainer_gesture")
    current_progress = dict(get_state_value(shared_state, snapshot_state, "trainer_progress", {}))
    sample_count = int(current_progress.get(selected_gesture, trainer.get_sample_count(selected_gesture)))
    status_text = str(get_state_value(shared_state, snapshot_state, "trainer_status", "Idle"))
    recording_active = bool(get_state_value(shared_state, snapshot_state, "trainer_recording_active", False))

    if st.button(
        f"Record {KNN_SAMPLES_REQUIRED} Samples",
        use_container_width=True,
        disabled=shared_state is None,
    ):
        shared_state["trainer_target_gesture"] = selected_gesture
        increment_shared_counter(shared_state, "trainer_record_request_id")

    st.progress(min(sample_count / float(KNN_SAMPLES_REQUIRED), 1.0), text=f"{sample_count}/{KNN_SAMPLES_REQUIRED} samples")
    st.write(f"Status: `{status_text}`")
    st.write(f"Recording: `{'ON' if recording_active else 'OFF'}`")

    if st.button("Train Gesture", use_container_width=True, disabled=shared_state is None):
        shared_state["trainer_target_gesture"] = selected_gesture
        increment_shared_counter(shared_state, "trainer_train_request_id")

    trained_gestures = list(get_state_value(shared_state, snapshot_state, "trained_gestures", trainer.list_trained_gestures()))
    if trained_gestures:
        st.write("Trained gestures")
        for gesture_name in trained_gestures:
            gesture_columns = st.columns([3, 1])
            gesture_columns[0].write(gesture_name)
            if gesture_columns[1].button("Delete", key=f"delete_{gesture_name}", use_container_width=True):
                if shared_state is not None:
                    shared_state["trainer_delete_gesture"] = gesture_name
                    increment_shared_counter(shared_state, "trainer_delete_request_id")
    else:
        st.caption("No trained custom gestures yet.")


def render_analytics_panel() -> None:
    """Render the analytics and session statistics panel."""

    st.subheader("Panel 3  Analytics")
    st.caption(ANALYTICS_HELP)

    tracker = load_analytics_tracker()
    counts_by_app = tracker.get_counts_by_app()
    confidence_history = tracker.get_confidence_history()
    session_stats = tracker.get_session_stats()

    stats_columns = st.columns(3)
    stats_columns[0].metric("Total gestures", int(session_stats.get("total_gestures", 0)))
    stats_columns[1].metric("Most used", str(session_stats.get("most_used_gesture") or "-"))
    stats_columns[2].metric("Avg confidence", f"{float(session_stats.get('average_confidence', 0.0)):.2f}")
    st.caption(f"Session duration: {float(session_stats.get('session_duration_seconds', 0.0)):.1f}s")

    if counts_by_app:
        counts_rows: list[dict[str, Any]] = []
        for app_name, gesture_counts in counts_by_app.items():
            for gesture_name, count in gesture_counts.items():
                counts_rows.append(
                    {
                        "app_name": app_name,
                        "gesture_name": gesture_name,
                        "count": count,
                    }
                )
        counts_frame = pd.DataFrame(counts_rows)
        st.bar_chart(counts_frame, x="app_name", y="count", color="gesture_name")
    else:
        st.caption("No analytics events logged yet.")

    if confidence_history:
        confidence_frame = pd.DataFrame(confidence_history)
        st.line_chart(confidence_frame.set_index("timestamp_utc"))
    else:
        st.caption("Confidence history will appear after gestures fire.")


def render_macro_panel(shared_state: Any | None, snapshot_state: dict[str, Any]) -> None:
    """Render macro save, delete, and test controls."""

    st.subheader("Panel 4  Macro Editor")
    st.caption(MACRO_HELP)

    macro_engine = load_macro_engine()
    current_app = str(get_state_value(shared_state, snapshot_state, "active_profile", "default"))

    macro_name = st.text_input("Macro name", key="macro_name")
    macro_app_scope = st.text_input("App scope", value=current_app or "default", key="macro_app_scope")
    steps_text = st.text_area("Macro steps (JSON list)", value=DEFAULT_MACRO_STEPS, height=180, key="macro_steps")

    if st.button("Save Macro", use_container_width=True):
        try:
            parsed_steps = json.loads(steps_text)
            if not isinstance(parsed_steps, list):
                raise ValueError("Macro steps must be a JSON list.")
            macro_engine.save_macro(
                name=macro_name.strip() or "Untitled Macro",
                steps=parsed_steps,
                app_name=macro_app_scope.strip() or "default",
            )
            st.success("Macro saved.")
        except Exception as exc:
            st.error(f"Failed to save macro: {exc}")

    macros = macro_engine.list_macros()
    if not macros:
        st.caption("No macros saved yet.")
        return

    selected_macro_label = st.selectbox(
        "Saved macros",
        [f"{macro.name} [{macro.app_name}]" for macro in macros],
        key="selected_macro_label",
    )
    selected_macro = macros[[f"{macro.name} [{macro.app_name}]" for macro in macros].index(selected_macro_label)]
    st.code(json.dumps(selected_macro.steps, indent=2), language="json")

    macro_columns = st.columns(2)
    if macro_columns[0].button("Delete Selected", use_container_width=True):
        macro_engine.delete_macro(name=selected_macro.name, app_name=selected_macro.app_name)
        st.success("Macro deleted.")
    if macro_columns[1].button("Test Selected", use_container_width=True):
        try:
            import pyautogui

            macro_engine._play_macro_definition(selected_macro, pyautogui)  # type: ignore[attr-defined]
            st.success("Macro playback finished.")
        except Exception as exc:
            st.error(f"Macro test failed: {exc}")


def main() -> None:
    """Run the Streamlit dashboard application."""

    st.set_page_config(
        page_title="KINESYS Dashboard",
        page_icon="K",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st_autorefresh(interval=DASHBOARD_REFRESH_MS, key="kinesys_dashboard_refresh")

    st.title("KINESYS Dashboard")
    st.caption(f"Local dashboard: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    shared_state = connect_shared_state()
    snapshot_state = load_snapshot_state()
    if shared_state is None:
        st.warning("Using dashboard snapshot fallback. Start main.py for the live bridge.")

    render_sidebar(shared_state, snapshot_state)

    top_row = st.columns(2)
    bottom_row = st.columns(2)

    with top_row[0]:
        render_live_panel(shared_state, snapshot_state)
    with top_row[1]:
        render_trainer_panel(shared_state, snapshot_state)
    with bottom_row[0]:
        render_analytics_panel()
    with bottom_row[1]:
        render_macro_panel(shared_state, snapshot_state)


if __name__ == "__main__":
    main()
