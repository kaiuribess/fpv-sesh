"""Keep startup errors visible when launched without a console."""
from pathlib import Path
import sys
import traceback


def main():
    try:
        from .ui import main as open_editor
        open_editor()
        return 0
    except Exception:
        root = Path(__file__).resolve().parents[1]
        try:
            (root / "logs").mkdir(exist_ok=True)
            (root / "logs/startup-error.log").write_text(traceback.format_exc(), encoding="utf-8")
            detail = "Details were saved in logs/startup-error.log."
        except OSError:
            detail = "Use a writable local folder for the extracted application."
        message = "FPV Sesh could not open. Run install.cmd to check or repair setup, then launch.cmd again.\n\n" + detail
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "FPV Sesh setup", 0x10)
        elif sys.stderr:
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
