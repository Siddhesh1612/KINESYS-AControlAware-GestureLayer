"""Air writing recognition and floating keyboard support for KINESYS."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any

import cv2
import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional runtime dependency
    tf = None

try:
    import tkinter as tk
except Exception:  # pragma: no cover - optional runtime dependency
    tk = None

from config import (
    AIR_WRITER_MIN_DRAW_DISTANCE,
    AIR_WRITER_MIN_PIXELS,
    AIR_WRITER_PADDING,
    AIR_WRITER_STATIONARY_DISTANCE,
    AIR_WRITER_STROKE_THICKNESS,
    CANVAS_SIZE,
    CHAR_SIZE,
    DASHBOARD_REFRESH_MS,
    EMNIST_CLASS_COUNT,
    EMNIST_MODEL_FILE,
    KEYBOARD_BUTTON_PADX,
    KEYBOARD_BUTTON_PADY,
    KEYBOARD_FONT_SIZE,
    KEYBOARD_HOVER_SECONDS,
    KEYBOARD_LAYOUT_ROWS,
    KEYBOARD_UI_POLL_MS,
    KEYBOARD_WINDOW_HEIGHT,
    KEYBOARD_WINDOW_POS_X,
    KEYBOARD_WINDOW_POS_Y,
    KEYBOARD_WINDOW_TITLE,
    KEYBOARD_WINDOW_WIDTH,
    PAUSE_THRESHOLD,
    WRITE_CONFIDENCE_THRESHOLD,
)


LOGGER = logging.getLogger(__name__)

EMPTY_TEXT = ""
KEYBOARD_SOURCE = "keyboard"
AIR_WRITER_SOURCE = "air"
SPACE_TOKEN = "SPACE"
BACKSPACE_TOKEN = "BACKSPACE"
BACKSPACE_OUTPUT = "<BACKSPACE>"
UI_COMMAND_SHOW = "show"
UI_COMMAND_HIDE = "hide"
UI_COMMAND_HOVER = "hover"
UI_COMMAND_SHUTDOWN = "shutdown"
KEYBOARD_BG = "#f2f2f2"
KEYBOARD_FG = "#111111"
KEYBOARD_HOVER_BG = "#0F6E56"
KEYBOARD_HOVER_FG = "#ffffff"
KEYBOARD_PANEL_BG = "#d9d9d9"
ALPHABET_BASE_ORDINAL = ord("A")
EMPTY_CANVAS_VALUE = 0
FILLED_CANVAS_VALUE = 255
FLOATING_KEY_WIDTH = len(BACKSPACE_TOKEN)


@dataclass(slots=True)
class WriteUpdate:
    """One output produced by the write-mode pipeline."""

    character: str | None
    confidence: float
    source: str | None
    canvas_active: bool


class FloatingKeyboardWindow:
    """Render a lightweight non-blocking floating keyboard in its own thread."""

    def __init__(self) -> None:
        """Initialize the UI thread state and cross-thread command queue."""

        self._command_queue: Queue[tuple[str, str | None]] = Queue()
        self._thread: threading.Thread | None = None
        self._buttons: dict[str, Any] = {}
        self._started = False

    def start(self) -> None:
        """Start the UI thread once if Tkinter is available."""

        if tk is None or self._started:
            return

        self._thread = threading.Thread(
            target=self._run,
            name="KinesysKeyboardWindow",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def show(self) -> None:
        """Show the floating keyboard window."""

        if self._started:
            self._command_queue.put((UI_COMMAND_SHOW, None))

    def hide(self) -> None:
        """Hide the floating keyboard window without shutting it down."""

        if self._started:
            self._command_queue.put((UI_COMMAND_HIDE, None))

    def update_hovered_key(self, key: str | None) -> None:
        """Update the highlighted key shown in the floating window."""

        if self._started:
            self._command_queue.put((UI_COMMAND_HOVER, key))

    def shutdown(self) -> None:
        """Request a clean shutdown of the UI thread."""

        if not self._started:
            return

        self._command_queue.put((UI_COMMAND_SHUTDOWN, None))
        if self._thread is not None:
            self._thread.join(timeout=float(DASHBOARD_REFRESH_MS) / 1000.0)
        self._started = False

    def _run(self) -> None:
        """Own the Tkinter event loop and process queued UI commands."""

        try:
            root = tk.Tk()
        except Exception as exc:  # pragma: no cover - UI dependent
            LOGGER.exception("Failed to start floating keyboard window: %s", exc)
            self._started = False
            return

        root.title(KEYBOARD_WINDOW_TITLE)
        root.geometry(
            f"{KEYBOARD_WINDOW_WIDTH}x{KEYBOARD_WINDOW_HEIGHT}"
            f"+{KEYBOARD_WINDOW_POS_X}+{KEYBOARD_WINDOW_POS_Y}"
        )
        root.configure(bg=KEYBOARD_PANEL_BG)
        root.resizable(False, False)
        try:
            root.attributes("-topmost", True)
        except Exception:  # pragma: no cover - platform dependent
            pass

        root.protocol("WM_DELETE_WINDOW", root.withdraw)

        outer_frame = tk.Frame(root, bg=KEYBOARD_PANEL_BG)
        outer_frame.pack(fill="both", expand=True, padx=KEYBOARD_BUTTON_PADX, pady=KEYBOARD_BUTTON_PADY)

        for row in KEYBOARD_LAYOUT_ROWS:
            row_frame = tk.Frame(outer_frame, bg=KEYBOARD_PANEL_BG)
            row_frame.pack(fill="x", expand=True, pady=KEYBOARD_BUTTON_PADY)
            for token in row:
                label = tk.Label(
                    row_frame,
                    text=self._display_token(token),
                    bg=KEYBOARD_BG,
                    fg=KEYBOARD_FG,
                    relief="raised",
                    bd=1,
                    font=("Segoe UI", KEYBOARD_FONT_SIZE, "bold"),
                    width=max(len(token), FLOATING_KEY_WIDTH),
                    padx=KEYBOARD_BUTTON_PADX,
                    pady=KEYBOARD_BUTTON_PADY,
                )
                label.pack(side="left", expand=True, fill="both", padx=KEYBOARD_BUTTON_PADX)
                self._buttons[token] = label

        root.withdraw()

        def process_commands() -> None:
            should_continue = self._process_commands(root)
            if should_continue:
                root.after(KEYBOARD_UI_POLL_MS, process_commands)

        root.after(KEYBOARD_UI_POLL_MS, process_commands)

        try:
            root.mainloop()
        except Exception as exc:  # pragma: no cover - UI dependent
            LOGGER.exception("Floating keyboard loop failed: %s", exc)

    def _process_commands(self, root: Any) -> bool:
        """Drain queued UI commands and apply them on the Tk thread."""

        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break

            if command == UI_COMMAND_SHOW:
                root.deiconify()
            elif command == UI_COMMAND_HIDE:
                root.withdraw()
            elif command == UI_COMMAND_HOVER:
                self._apply_hover_state(payload)
            elif command == UI_COMMAND_SHUTDOWN:
                try:
                    root.quit()
                    root.destroy()
                except Exception:  # pragma: no cover - UI dependent
                    pass
                return False

        return True

    def _apply_hover_state(self, hovered_key: str | None) -> None:
        """Refresh button colors so only the hovered key is highlighted."""

        for token, button in self._buttons.items():
            is_hovered = token == hovered_key
            button.configure(
                bg=KEYBOARD_HOVER_BG if is_hovered else KEYBOARD_BG,
                fg=KEYBOARD_HOVER_FG if is_hovered else KEYBOARD_FG,
            )

    @staticmethod
    def _display_token(token: str) -> str:
        """Return a human-readable keyboard label for one token."""

        if token == SPACE_TOKEN:
            return "Space"
        if token == BACKSPACE_TOKEN:
            return "Backspace"
        return token


class AirWriter:
    """Track fingertip strokes, recognize letters, and provide keyboard fallback."""

    def __init__(self, model_path: str = EMNIST_MODEL_FILE) -> None:
        """Initialize the canvas, model loader, and floating keyboard state."""

        self._model_path = Path(model_path)
        self._model = self._load_model()
        self._canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self._keyboard_window = FloatingKeyboardWindow()
        self._session_active = False
        self._previous_canvas_point: tuple[int, int] | None = None
        self._last_stroke_time = time.perf_counter()
        self._hovered_key: str | None = None
        self._hover_started_at = 0.0

    def start_session(self) -> None:
        """Enable write mode and keep the floating keyboard hidden by default."""

        self._session_active = True
        self.reset()
        self._keyboard_window.start()
        self._keyboard_window.hide()

    def stop_session(self) -> None:
        """Disable write mode and hide the floating keyboard."""

        self._session_active = False
        self._keyboard_window.hide()
        self.reset()

    def reset(self) -> None:
        """Clear the current canvas and hover state."""

        self._canvas.fill(EMPTY_CANVAS_VALUE)
        self._previous_canvas_point = None
        self._last_stroke_time = time.perf_counter()
        self._clear_keyboard_hover()

    def update(
        self,
        index_point: tuple[int, int] | None,
        frame_size: tuple[int, int],
        allow_stroke_drawing: bool = True,
    ) -> WriteUpdate:
        """Process the current fingertip point and return any typed output."""

        canvas_active = self._canvas_has_ink()
        if not self._session_active:
            return WriteUpdate(character=None, confidence=0.0, source=None, canvas_active=canvas_active)

        if index_point is None:
            self._previous_canvas_point = None
            return self._finalize_pause_if_ready()

        now = time.perf_counter()
        canvas_point = self._map_to_canvas(index_point=index_point, frame_size=frame_size)
        movement_distance = self._distance(self._previous_canvas_point, canvas_point)

        if (
            allow_stroke_drawing
            and self._previous_canvas_point is not None
            and movement_distance >= AIR_WRITER_MIN_DRAW_DISTANCE
        ):
            self._draw_stroke(self._previous_canvas_point, canvas_point)
            self._last_stroke_time = now
            self._clear_keyboard_hover()

        self._previous_canvas_point = canvas_point
        canvas_active = self._canvas_has_ink()

        if not allow_stroke_drawing:
            self._previous_canvas_point = None

        if canvas_active:
            self._keyboard_window.update_hovered_key(None)
            if movement_distance <= AIR_WRITER_STATIONARY_DISTANCE:
                return self._finalize_pause_if_ready()
            return WriteUpdate(character=None, confidence=0.0, source=None, canvas_active=True)

        return self._update_keyboard_hover(index_point=index_point, frame_size=frame_size, now=now)

    def shutdown(self) -> None:
        """Stop the floating keyboard thread and release UI resources."""

        self._keyboard_window.shutdown()

    def model_loaded(self) -> bool:
        """Return whether an EMNIST recognition model is available."""

        return self._model is not None

    def _finalize_pause_if_ready(self) -> WriteUpdate:
        """Recognize a drawn character once a pause threshold has elapsed."""

        if not self._canvas_has_ink():
            return WriteUpdate(character=None, confidence=0.0, source=None, canvas_active=False)

        if time.perf_counter() - self._last_stroke_time < PAUSE_THRESHOLD:
            return WriteUpdate(character=None, confidence=0.0, source=None, canvas_active=True)

        character, confidence = self._recognize_canvas()
        self._canvas.fill(EMPTY_CANVAS_VALUE)
        self._previous_canvas_point = None
        if character is None:
            return WriteUpdate(character=None, confidence=confidence, source=None, canvas_active=False)

        return WriteUpdate(
            character=character,
            confidence=confidence,
            source=AIR_WRITER_SOURCE,
            canvas_active=False,
        )

    def _update_keyboard_hover(
        self,
        index_point: tuple[int, int],
        frame_size: tuple[int, int],
        now: float,
    ) -> WriteUpdate:
        """Track hover dwell over the logical keyboard grid and emit key presses."""

        hovered_key = self._map_to_keyboard_key(index_point=index_point, frame_size=frame_size)
        self._keyboard_window.update_hovered_key(hovered_key)

        if hovered_key is None:
            self._clear_keyboard_hover()
            return WriteUpdate(character=None, confidence=0.0, source=None, canvas_active=False)

        if hovered_key != self._hovered_key:
            self._hovered_key = hovered_key
            self._hover_started_at = now
            return WriteUpdate(character=None, confidence=0.0, source=None, canvas_active=False)

        if now - self._hover_started_at < KEYBOARD_HOVER_SECONDS:
            return WriteUpdate(character=None, confidence=0.0, source=None, canvas_active=False)

        self._clear_keyboard_hover()
        return WriteUpdate(
            character=self._keyboard_token_to_output(hovered_key),
            confidence=1.0,
            source=KEYBOARD_SOURCE,
            canvas_active=False,
        )

    def _map_to_canvas(
        self,
        index_point: tuple[int, int],
        frame_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Project a frame coordinate into the square writing canvas."""

        frame_width, frame_height = frame_size
        point_x, point_y = index_point

        mapped_x = int((point_x * CANVAS_SIZE) / float(max(frame_width, 1)))
        mapped_y = int((point_y * CANVAS_SIZE) / float(max(frame_height, 1)))

        return (
            max(0, min(mapped_x, CANVAS_SIZE - 1)),
            max(0, min(mapped_y, CANVAS_SIZE - 1)),
        )

    def _map_to_keyboard_key(
        self,
        index_point: tuple[int, int],
        frame_size: tuple[int, int],
    ) -> str | None:
        """Map a frame point to one logical key in the floating keyboard layout."""

        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            return None

        point_x, point_y = index_point
        normalized_y = point_y / float(frame_height)
        row_index = min(int(normalized_y * len(KEYBOARD_LAYOUT_ROWS)), len(KEYBOARD_LAYOUT_ROWS) - 1)
        row = KEYBOARD_LAYOUT_ROWS[row_index]
        normalized_x = point_x / float(frame_width)
        key_index = min(int(normalized_x * len(row)), len(row) - 1)
        return row[key_index]

    def _draw_stroke(
        self,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
    ) -> None:
        """Draw one fingertip stroke segment on the monochrome canvas."""

        cv2.line(
            self._canvas,
            start_point,
            end_point,
            FILLED_CANVAS_VALUE,
            AIR_WRITER_STROKE_THICKNESS,
        )

    def _recognize_canvas(self) -> tuple[str | None, float]:
        """Recognize the current canvas contents as one uppercase letter."""

        if self._model is None:
            return None, 0.0

        active_points = np.argwhere(self._canvas > EMPTY_CANVAS_VALUE)
        if len(active_points) < AIR_WRITER_MIN_PIXELS:
            return None, 0.0

        y_values = active_points[:, 0]
        x_values = active_points[:, 1]
        y_min = max(int(y_values.min()) - AIR_WRITER_PADDING, 0)
        y_max = min(int(y_values.max()) + AIR_WRITER_PADDING, CANVAS_SIZE - 1)
        x_min = max(int(x_values.min()) - AIR_WRITER_PADDING, 0)
        x_max = min(int(x_values.max()) + AIR_WRITER_PADDING, CANVAS_SIZE - 1)

        cropped = self._canvas[y_min : y_max + 1, x_min : x_max + 1]

        # Pad to square to preserve aspect ratio before resize
        h, w = cropped.shape
        side = max(h, w)
        padded = np.zeros((side, side), dtype=np.uint8)
        y_offset = (side - h) // 2
        x_offset = (side - w) // 2
        padded[y_offset : y_offset + h, x_offset : x_offset + w] = cropped

        resized = cv2.resize(padded, (CHAR_SIZE, CHAR_SIZE), interpolation=cv2.INTER_AREA)

        # EMNIST letters are stored transposed — rotate to match training orientation
        resized = np.transpose(resized)
        resized = np.fliplr(resized)

        normalized = resized.astype("float32") / float(FILLED_CANVAS_VALUE)
        input_tensor = normalized.reshape(1, CHAR_SIZE, CHAR_SIZE, 1)

        try:
            predictions = self._model.predict(input_tensor, verbose=0)
        except Exception as exc:  # pragma: no cover - TensorFlow dependent
            LOGGER.exception("Air-writer prediction failed: %s", exc)
            return None, 0.0

        if len(predictions) == 0:
            return None, 0.0

        scores = predictions[0]
        predicted_index = int(np.argmax(scores))
        confidence = float(scores[predicted_index])

        # EMNIST/letters labels are 1-indexed (1=A … 26=Z)
        zero_based_index = (predicted_index - 1) if len(scores) == EMNIST_CLASS_COUNT + 1 else predicted_index
        zero_based_index = max(0, zero_based_index)

        if confidence < WRITE_CONFIDENCE_THRESHOLD or zero_based_index >= EMNIST_CLASS_COUNT:
            return None, confidence

        return chr(ALPHABET_BASE_ORDINAL + zero_based_index), confidence

    def _load_model(self) -> Any | None:
        """Load the trained EMNIST model when TensorFlow and the model file are available."""

        if tf is None:
            LOGGER.warning("TensorFlow is unavailable; air writing will use keyboard fallback only.")
            return None

        if not self._model_path.exists():
            LOGGER.warning("EMNIST model file not found at %s", self._model_path)
            return None

        try:
            return tf.keras.models.load_model(self._model_path)
        except Exception as exc:  # pragma: no cover - TensorFlow dependent
            LOGGER.exception("Failed to load EMNIST model: %s", exc)
            return None

    def _canvas_has_ink(self) -> bool:
        """Return whether the current canvas contains enough pixels to represent a stroke."""

        return int(np.count_nonzero(self._canvas)) >= AIR_WRITER_MIN_PIXELS

    def _clear_keyboard_hover(self) -> None:
        """Reset the current keyboard dwell state and clear the UI highlight."""

        self._hovered_key = None
        self._hover_started_at = 0.0
        self._keyboard_window.update_hovered_key(None)

    @staticmethod
    def _keyboard_token_to_output(token: str) -> str:
        """Translate a logical keyboard token into emitted output text."""

        if token == SPACE_TOKEN:
            return " "
        if token == BACKSPACE_TOKEN:
            return BACKSPACE_OUTPUT
        return token

    @staticmethod
    def _distance(
        start_point: tuple[int, int] | None,
        end_point: tuple[int, int],
    ) -> float:
        """Return the Euclidean distance between two canvas points."""

        if start_point is None:
            return 0.0
        start_array = np.array(start_point, dtype="float32")
        end_array = np.array(end_point, dtype="float32")
        return float(np.linalg.norm(end_array - start_array))
