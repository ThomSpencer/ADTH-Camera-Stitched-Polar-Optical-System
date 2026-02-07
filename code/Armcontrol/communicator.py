"""Serial communicator helper for Arduino arm control.

Provides `send_move_command` which sends: "MOVE %d %d %d %d %d %d" over serial.
"""
from typing import Iterable, Optional
import time
import random
from serial import Serial, SerialException # type: ignore

# Module-level persistent Serial instance to avoid reopening (which toggles DTR/reset)
_SER = None
_SER_PORT = None


def _open_serial(port: str, baudrate: int, timeout: float):
    """Open and configure a persistent serial connection.

    Clears DTR after opening to avoid auto-reset on many Arduino boards.
    """
    global _SER, _SER_PORT

    if _SER is not None:
        try:
            if _SER.is_open and _SER_PORT == port:
                return _SER
            else:
                try:
                    _SER.close()
                except Exception:
                    pass
        except Exception:
            # If is_open or close() fails, we'll just try reopening below
            pass

    ser = Serial(port, baudrate=baudrate, timeout=timeout)
    # Clear DTR to prevent the auto-reset on many Arduino boards
    try:
        ser.setDTR(False) # pyright: ignore[reportAttributeAccessIssue] #
    except Exception:
        try:
            ser.dtr = False
        except Exception:
            pass

    # Give device a short moment to settle (bootloader may still trigger on open)
    time.sleep(0.2)

    # Flush any startup lines the board may have printed (e.g. "Alive")
    try:
        ser.reset_input_buffer()
    except Exception:
        try:
            ser.flushInput() # pyright: ignore[reportAttributeAccessIssue] #
        except Exception:
            pass

    _SER = ser
    _SER_PORT = port
    return _SER


def close_serial():
    """Close the persistent serial connection if open."""
    global _SER, _SER_PORT
    if _SER is not None:
        try:
            _SER.close()
        except Exception:
            pass
    _SER = None
    _SER_PORT = None


def send_move_command(port: str = "/dev/ttyACM0", angles: Iterable[int] = (90, 90), baudrate: int = 9600, timeout: float = 2.0) -> Optional[str]:
    """Send a MOVE command to an Arduino over serial and print/return its response.

    This function reuses a single serial connection to avoid toggling DTR and
    resetting the board on each call. If you must target a different `port`,
    the connection will be reopened.
    """
    angles_list = list(angles)
    if len(angles_list) != 2:
        raise ValueError("angles must be an iterable of 2 integers")

    # Validate and coerce angles to integers in range 0-180
    validated = []
    for a in angles_list:
        try:
            ai = int(a)
        except Exception:
            raise ValueError(f"angle value not convertible to int: {a!r}")
        if ai < 0 or ai > 180:
            raise ValueError(f"angle out of range 0-180: {ai}")
        validated.append(ai)

    msg = "MOVE %d %d\n" % tuple(validated)
    print(f"Sending command: {msg.strip()} to port: {port}")

    ser = _open_serial(port, baudrate, timeout)
    try:
        ser.write(msg.encode("utf-8"))
        ser.flush()

        # Read a single response line (b'' on timeout)
        line = ser.readline()
        if not line:
            print("No response (timeout).")
            return None

        resp = line.decode("utf-8", errors="replace").strip()
        print(f"Received response: {resp}")
        return resp
    except Exception as e:
        print(f"Error during serial I/O: {e}")
        return None


if __name__ == "__main__":
    try:
        while True:
            angles = (random.randint(0, 180), random.randint(0, 180))
            try:
                send_move_command(port="/dev/ttyACM0", angles=angles)
            except Exception as e:
                print("Error sending MOVE command:", e)
            input()
    finally:
        close_serial()