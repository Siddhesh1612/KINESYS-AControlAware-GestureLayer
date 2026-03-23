"""Runtime environment checks for the KINESYS foundation build."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import (
    APP_NAME,
    EMNIST_MODEL_FILE,
    FOUNDATION_CAMERA_BACKEND,
    FOUNDATION_CAMERA_INDEX,
    FOUNDATION_DEPENDENCIES,
    LOG_FORMAT,
    LOG_LEVEL,
    MAIN_WINDOW_NAME,
    REQUIRED_FOUNDATION_FILES,
    STRICT_DEPENDENCIES,
    STRICT_REQUIRED_FILES,
    SUPPORTED_PYTHON_MAX,
    SUPPORTED_PYTHON_MIN,
    WEBCAM_FOURCC,
    WEBCAM_HEIGHT,
    WEBCAM_WIDTH,
)


LOGGER = logging.getLogger(__name__)

SUCCESS_PREFIX = "[PASS]"
FAIL_PREFIX = "[FAIL]"
INFO_PREFIX = "[INFO]"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


@dataclass(slots=True)
class SetupCheckResult:
    """Represents the result of a single setup validation step."""

    name: str
    passed: bool
    detail: str


def configure_logging() -> None:
    """Configure process-wide logging for setup checks."""

    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)


def check_python_version() -> SetupCheckResult:
    """Validate that the current Python interpreter is supported."""

    current_version = sys.version_info[:2]
    if SUPPORTED_PYTHON_MIN <= current_version <= SUPPORTED_PYTHON_MAX:
        detail = f"Python {sys.version.split()[0]} is supported."
        return SetupCheckResult(name="python", passed=True, detail=detail)

    detail = (
        f"Python {sys.version.split()[0]} is unsupported. "
        f"Use {SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]} "
        f"to {SUPPORTED_PYTHON_MAX[0]}.{SUPPORTED_PYTHON_MAX[1]}."
    )
    return SetupCheckResult(name="python", passed=False, detail=detail)


def check_project_files(required_files: Iterable[Path], strict: bool) -> SetupCheckResult:
    """Ensure the current build's required files are present."""

    missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]
    if strict:
        missing_files.extend(
            str(file_path) for file_path in STRICT_REQUIRED_FILES if not file_path.exists()
        )

    if not missing_files:
        return SetupCheckResult(
            name="files",
            passed=True,
            detail="Required foundation files are present.",
        )

    detail = "Missing required file(s): " + ", ".join(missing_files)
    return SetupCheckResult(name="files", passed=False, detail=detail)


def check_dependencies(strict: bool) -> SetupCheckResult:
    """Ensure the dependencies needed for the current build are importable."""

    dependency_map = dict(FOUNDATION_DEPENDENCIES)
    if strict:
        dependency_map.update(STRICT_DEPENDENCIES)

    missing_packages: list[str] = []
    for module_name, package_name in dependency_map.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.debug("Dependency import failed for %s: %s", module_name, exc)
            missing_packages.append(package_name)

    if not missing_packages:
        return SetupCheckResult(
            name="dependencies",
            passed=True,
            detail="All required dependencies import successfully.",
        )

    detail = "Install missing dependency package(s): " + ", ".join(sorted(set(missing_packages)))
    return SetupCheckResult(name="dependencies", passed=False, detail=detail)


def _get_camera_backend(cv2_module: object) -> int:
    """Resolve the preferred OpenCV backend constant for Windows webcam access."""

    return getattr(cv2_module, FOUNDATION_CAMERA_BACKEND, FOUNDATION_CAMERA_INDEX)


def check_webcam() -> SetupCheckResult:
    """Validate that a webcam can be opened and a frame can be read."""

    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:  # pragma: no cover - environment dependent
        detail = f"OpenCV import failed before webcam test: {exc}"
        return SetupCheckResult(name="webcam", passed=False, detail=detail)

    capture = None
    try:
        backend = _get_camera_backend(cv2)
        capture = cv2.VideoCapture(FOUNDATION_CAMERA_INDEX, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*WEBCAM_FOURCC))

        if not capture.isOpened():
            return SetupCheckResult(
                name="webcam",
                passed=False,
                detail="Unable to open the default webcam.",
            )

        frame_ok, _ = capture.read()
        if not frame_ok:
            return SetupCheckResult(
                name="webcam",
                passed=False,
                detail="Webcam opened but no frame could be read.",
            )

        detail = f"Webcam opened successfully for {APP_NAME} ({MAIN_WINDOW_NAME})."
        return SetupCheckResult(name="webcam", passed=True, detail=detail)
    except Exception as exc:  # pragma: no cover - hardware dependent
        detail = f"Webcam validation failed: {exc}"
        return SetupCheckResult(name="webcam", passed=False, detail=detail)
    finally:
        if capture is not None:
            capture.release()


def print_result(result: SetupCheckResult) -> None:
    """Print a formatted setup result line."""

    prefix = SUCCESS_PREFIX if result.passed else FAIL_PREFIX
    print(f"{prefix} {result.name}: {result.detail}")


def run_checks(strict: bool = False, skip_webcam: bool = False) -> bool:
    """Run all required setup checks and return whether they all pass."""

    configure_logging()
    print(f"{INFO_PREFIX} Starting {APP_NAME} setup validation")
    if strict:
        print(f"{INFO_PREFIX} Strict mode enabled; expecting {EMNIST_MODEL_FILE}")

    results = [
        check_python_version(),
        check_project_files(REQUIRED_FOUNDATION_FILES, strict=strict),
        check_dependencies(strict=strict),
    ]

    if skip_webcam:
        results.append(
            SetupCheckResult(
                name="webcam",
                passed=True,
                detail="Skipped by user request.",
            )
        )
    else:
        results.append(check_webcam())

    overall_success = all(result.passed for result in results)
    for result in results:
        print_result(result)

    if overall_success:
        print(f"{SUCCESS_PREFIX} Setup checks completed successfully.")
    else:
        print(f"{FAIL_PREFIX} Setup checks failed. Resolve the errors above and retry.")
    return overall_success


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for setup validation."""

    parser = argparse.ArgumentParser(description="Validate the KINESYS foundation setup.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Validate future-build dependencies and required model files as well.",
    )
    parser.add_argument(
        "--skip-webcam",
        action="store_true",
        help="Skip the webcam readiness test.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the setup validator as a command-line entry point."""

    args = parse_args()
    return EXIT_SUCCESS if run_checks(strict=args.strict, skip_webcam=args.skip_webcam) else EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
