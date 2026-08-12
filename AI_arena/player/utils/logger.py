"""Logging and stdin signal listening utilities for training loops."""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """Appends every log line to a file and optionally mirrors to stdout.

    Enforces a PID lock so only one training session writes to the log at a time.
    A second process attempting to start will exit immediately with a clear error.
    """

    def __init__(self, log_path: Path, quiet: bool = False) -> None:
        self.quiet = quiet
        self.log_path = log_path
        self._lock_path = log_path.with_suffix(".lock")

        # ── Exclusive instance lock ──────────────────────────────────────────
        my_pid = str(threading.current_thread().native_id or "")
        import os
        my_pid = str(os.getpid())

        if self._lock_path.exists():
            existing_pid = self._lock_path.read_text().strip()
            # Check if that PID is actually still alive
            try:
                os.kill(int(existing_pid), 0)
                # Process still alive — refuse to start
                print(
                    f"\n{'!'*60}\n"
                    f"ERROR: Another training session is already running (PID {existing_pid}).\n"
                    f"Kill it first:  kill -SIGINT {existing_pid}\n"
                    f"Then re-run training.\n"
                    f"{'!'*60}\n"
                )
                raise SystemExit(1)
            except (ProcessLookupError, ValueError):
                # Stale lock — previous session was killed without cleanup
                self._lock_path.unlink(missing_ok=True)

        self._lock_path.write_text(my_pid)

        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        self._file.write(
            f"\n{'='*70}\n"
            f"Training session started at {datetime.now().isoformat()}\n"
            f"{'='*70}\n"
        )
        self._file.flush()

    def log(self, message: str) -> None:
        if not self.quiet:
            print(message)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
        self._lock_path.unlink(missing_ok=True)


class QuitListener:
    """Background listener that watches stdin for a single 'q' keypress.

    Works cross-platform:
      - On Unix, puts terminal into cbreak mode for immediate 'q' detection.
      - On Windows, polls msvcrt.kbhit()/getch().
      - Falls back to line-buffered input if stdin isn't interactive.
    """

    def __init__(self) -> None:
        self._stop_requested = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(
            "Press 'q' at any time to stop training gracefully and save a checkpoint."
        )

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        if not sys.stdin.isatty():
            self._run_line_buffered()
            return

        if sys.platform.startswith("win"):
            self._run_windows()
        else:
            self._run_unix()

    def _run_line_buffered(self) -> None:
        while not self._shutdown.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            if line.strip().lower() == "q":
                self._stop_requested.set()
                return

    def _run_windows(self) -> None:
        import msvcrt

        while not self._shutdown.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    if ch.decode(errors="ignore").lower() == "q":
                        self._stop_requested.set()
                        return
                except Exception:
                    pass
            time.sleep(0.05)

    def _run_unix(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._shutdown.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "q":
                        self._stop_requested.set()
                        break
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
