"""Gesture macro storage and playback for KINESYS."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

from config import MACROS_FILE


LOGGER = logging.getLogger(__name__)

MACRO_FILE_VERSION = 1
DEFAULT_APP_SCOPE = "default"
DEFAULT_TRIGGER_GESTURE = "CIRCLE"


@dataclass(slots=True)
class MacroDefinition:
    """One persisted macro bound to an app scope and trigger gesture."""

    name: str
    app_name: str
    trigger_gesture: str
    steps: list[dict[str, Any]]


class MacroEngine:
    """Persist, retrieve, and execute user-defined macro sequences."""

    def __init__(self, macros_path: str = MACROS_FILE) -> None:
        """Load macro definitions from disk or initialize an empty store."""

        self._macros_path = Path(macros_path)
        self._macros: list[MacroDefinition] = []
        self._load_macros()

    def list_macros(self, app_name: str | None = None) -> list[MacroDefinition]:
        """Return all macros or only those matching one app scope."""

        if app_name is None:
            return list(self._macros)
        return [macro for macro in self._macros if macro.app_name == app_name]

    def save_macro(
        self,
        name: str,
        steps: list[dict[str, Any]],
        app_name: str = DEFAULT_APP_SCOPE,
        trigger_gesture: str = DEFAULT_TRIGGER_GESTURE,
    ) -> None:
        """Create or replace one macro definition and persist the change."""

        self.delete_macro(name=name, app_name=app_name)
        self._macros.append(
            MacroDefinition(
                name=name,
                app_name=app_name,
                trigger_gesture=trigger_gesture,
                steps=steps,
            )
        )
        self._save_macros()

    def delete_macro(self, name: str, app_name: str | None = None) -> bool:
        """Delete one macro by name, optionally scoped to one app."""

        original_count = len(self._macros)
        self._macros = [
            macro
            for macro in self._macros
            if not (macro.name == name and (app_name is None or macro.app_name == app_name))
        ]
        changed = len(self._macros) != original_count
        if changed:
            self._save_macros()
        return changed

    def play_macro_for_context(
        self,
        trigger_gesture: str,
        active_app: str,
        profile_name: str,
        pyautogui_module: Any,
    ) -> str | None:
        """Execute the best matching macro for the current app/profile context."""

        macro = self._select_macro(trigger_gesture=trigger_gesture, active_app=active_app, profile_name=profile_name)
        if macro is None:
            return None
        self._play_macro_definition(macro=macro, pyautogui_module=pyautogui_module)
        return macro.name

    def _select_macro(
        self,
        trigger_gesture: str,
        active_app: str,
        profile_name: str,
    ) -> MacroDefinition | None:
        """Select the most specific macro for the current app and trigger."""

        exact_match = None
        profile_match = None
        default_match = None

        for macro in self._macros:
            if macro.trigger_gesture != trigger_gesture:
                continue
            if macro.app_name == active_app:
                exact_match = macro
                break
            if macro.app_name == profile_name:
                profile_match = macro
            if macro.app_name == DEFAULT_APP_SCOPE:
                default_match = macro

        return exact_match or profile_match or default_match

    def _play_macro_definition(self, macro: MacroDefinition, pyautogui_module: Any) -> None:
        """Execute one macro definition step-by-step."""

        for step in macro.steps:
            step_type = str(step.get("type", "")).lower()
            try:
                if step_type == "hotkey":
                    pyautogui_module.hotkey(*step.get("keys", []))
                elif step_type == "press":
                    pyautogui_module.press(step.get("key", ""))
                elif step_type == "write":
                    pyautogui_module.write(step.get("text", ""))
                elif step_type == "scroll":
                    pyautogui_module.scroll(int(step.get("amount", 0)))
                elif step_type == "click":
                    pyautogui_module.click(button=step.get("button", "left"))
                elif step_type == "sleep":
                    time.sleep(float(step.get("seconds", 0.0)))
            except Exception as exc:
                LOGGER.exception("Macro step failed for '%s': %s", macro.name, exc)

    def _load_macros(self) -> None:
        """Load persisted macro definitions from disk."""

        if not self._macros_path.exists():
            return

        try:
            with self._macros_path.open("r", encoding="utf-8") as macro_file:
                payload = json.load(macro_file)
        except Exception as exc:
            LOGGER.exception("Failed to load macros file: %s", exc)
            return

        if not isinstance(payload, dict):
            return
        if payload.get("version") != MACRO_FILE_VERSION:
            return

        raw_macros = payload.get("macros", [])
        if not isinstance(raw_macros, list):
            return

        loaded_macros: list[MacroDefinition] = []
        for raw_macro in raw_macros:
            if not isinstance(raw_macro, dict):
                continue
            try:
                loaded_macros.append(MacroDefinition(**raw_macro))
            except Exception:
                continue
        self._macros = loaded_macros

    def _save_macros(self) -> None:
        """Persist the current macro list to disk."""

        payload = {
            "version": MACRO_FILE_VERSION,
            "macros": [
                {
                    "name": macro.name,
                    "app_name": macro.app_name,
                    "trigger_gesture": macro.trigger_gesture,
                    "steps": macro.steps,
                }
                for macro in self._macros
            ],
        }

        try:
            with self._macros_path.open("w", encoding="utf-8") as macro_file:
                json.dump(payload, macro_file, indent=2)
        except Exception as exc:
            LOGGER.exception("Failed to save macros file: %s", exc)
