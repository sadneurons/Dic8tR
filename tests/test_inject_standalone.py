"""Standalone text injection test.

Run from project root:  python tests/test_inject_standalone.py

You have 3 seconds to click into a text editor / text field before
injection happens. Tests both direct typing and clipboard paste.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whispr.inject import inject_text, detect_display_server


def main() -> None:
    display = detect_display_server()
    print(f"Display server: {display}")

    # Test 1: Short text via keystroke injection
    print("\n--- Test 1: xdotool type (short text) ---")
    print("Click into a text editor in 3 seconds...")
    time.sleep(3)
    ok = inject_text("Hello from Whispr! Patient reviewed in the Brain Health Clinic.")
    print(f"Result: {'OK' if ok else 'FAILED'}")

    time.sleep(1)

    # Test 2: Newlines and punctuation
    print("\n--- Test 2: Multiline text ---")
    print("Injecting multiline text in 2 seconds...")
    time.sleep(2)
    ok = inject_text("Line one.\nLine two.\n\nNew paragraph with pTau217 and RBANS.")
    print(f"Result: {'OK' if ok else 'FAILED'}")

    time.sleep(1)

    # Test 3: Clipboard paste (force by setting threshold to 0)
    print("\n--- Test 3: Clipboard paste ---")
    print("Injecting via clipboard in 2 seconds...")
    time.sleep(2)
    ok = inject_text(
        "This text was injected via clipboard paste.",
        clipboard_threshold=0,
    )
    print(f"Result: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
