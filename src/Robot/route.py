#!/usr/bin/env python3
"""Robot inspection run: drive forward, then take 3 photos by panning the camera.

1. Motors 1-4 (board .233) drive forward for up to DRIVE_SECONDS,
   or stop earlier if the ultrasonic sensor (board .87) sees an obstacle < STOP_CM.
2. Camera (board .207) takes a photo at home, then motor 2 (board .87) pans it
   PAN_STEPS left for a photo, then PAN_STEPS right of home for a photo,
   and finally returns to home.

Motor 2: dir=0 = left, dir=1 = right. Home = position 0.
Photos are saved to src/Robot/runs/<timestamp>/.

Usage:  python3 src/Robot/route.py [--stop-cm 30] [--drive-s 10] [--no-drive]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

CAM = "http://10.12.241.207"
WHEELS = "http://10.12.241.233"
STEPPER = "http://10.12.241.87"

# /dir value that makes each wheel motor drive the robot forward.
# Wheel motor 2 is mounted/wired the other way round, so it needs -1.
WHEEL_FORWARD = {1: 1, 2: -1, 3: 1, 4: 1}

PAN_MOTOR = 2
PAN_STEPS = 400
PAN_SPEED = 300
DIR_LEFT, DIR_RIGHT = 0, 1


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def get(url, timeout=10.0, retries=0):
    """HTTP GET -> bytes. Retries on network errors, raises the last error."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == retries:
                raise
            log(f"  retry {attempt + 1}/{retries} {url} ({e})")
            time.sleep(0.5)


def read_distance(timeout=1.5):
    """Distance in cm, or None if unreadable (timeout, '--', 0)."""
    try:
        text = get(f"{STEPPER}/distance", timeout=timeout).decode().strip()
        cm = float(text.split()[0])
        return cm if cm > 0 else None
    except Exception:
        return None


def stop_wheels():
    """Stop all wheels; keep retrying — this call must not be lost."""
    for attempt in range(10):
        try:
            get(f"{WHEELS}/stop?m=all", timeout=3)
            log("roți OPRITE")
            return True
        except Exception as e:
            log(f"  stop eșuat ({e}), reîncerc {attempt + 1}/10")
    log("!!! NU am putut opri roțile — oprește robotul manual !!!")
    return False


def drive_forward(seconds, stop_cm, sensor_grace_s=2.0):
    """Drive forward until time is up, an obstacle is near, or the sensor goes silent."""
    # Wake the boards up (the first request after idle is slow).
    for base in (STEPPER, WHEELS):
        try:
            get(f"{base}/", timeout=8, retries=1)
        except Exception as e:
            log(f"placa {base} nu răspunde: {e}")
            return "board_unreachable", None

    d = read_distance(timeout=5)
    if d is not None and d < stop_cm:
        log(f"obstacol deja la {d:.1f} cm < {stop_cm} cm — nu pornesc")
        return "obstacle_at_start", d

    for m, val in WHEEL_FORWARD.items():
        get(f"{WHEELS}/dir?m={m}&val={val}", timeout=5, retries=2)

    reason, last_d = "timeout", d
    try:
        get(f"{WHEELS}/start?m=all", timeout=5, retries=2)
        t0 = last_ok = time.monotonic()
        log(f"roți PORNITE (max {seconds}s, stop la < {stop_cm} cm)")
        while True:
            now = time.monotonic()
            if now - t0 >= seconds:
                break
            d = read_distance()
            if d is not None:
                last_d, last_ok = d, time.monotonic()
                if d < stop_cm:
                    reason = "obstacle"
                    log(f"obstacol la {d:.1f} cm")
                    break
            elif time.monotonic() - last_ok > sensor_grace_s:
                reason = "sensor_lost"
                log(f"senzorul nu a răspuns de {sensor_grace_s}s — opresc din siguranță")
                break
            time.sleep(0.05)
        log(f"mers {time.monotonic() - t0:.1f}s, motiv oprire: {reason}")
    finally:
        stop_wheels()
    return reason, last_d


def pan(direction, steps):
    name = "stânga" if direction == DIR_LEFT else "dreapta"
    log(f"motor {PAN_MOTOR}: {steps} pași spre {name}")
    url = f"{STEPPER}/move?motor={PAN_MOTOR}&dir={direction}&speed={PAN_SPEED}&steps={steps}"
    resp = get(url, timeout=90).decode(errors="replace").strip()
    log(f"  -> {resp}")


def rotate_180(path):
    """The camera is mounted upside down; rotate in place with macOS `sips`."""
    try:
        subprocess.run(["sips", "-r", "180", path], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as e:
        log(f"  nu am putut roti {path} ({e}) — rămâne neîntoarsă")


def capture(path, retries=3):
    """Save a JPEG from the camera; retry if the image arrives truncated (weak WiFi)."""
    for attempt in range(1, retries + 1):
        try:
            data = get(f"{CAM}/capture", timeout=45)
            if data[:2] == b"\xff\xd8" and data.rstrip(b"\x00")[-2:] == b"\xff\xd9":
                with open(path, "wb") as f:
                    f.write(data)
                rotate_180(path)
                log(f"poză salvată {path} ({len(data) // 1024} KB)")
                return True
            log(f"  poză incompletă ({len(data)} B), încercarea {attempt}/{retries}")
        except Exception as e:
            log(f"  captură eșuată ({e}), încercarea {attempt}/{retries}")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stop-cm", type=float, default=30.0, help="oprire dacă obstacolul e mai aproape (cm)")
    ap.add_argument("--drive-s", type=float, default=10.0, help="durata maximă de mers înainte (s)")
    ap.add_argument("--no-drive", action="store_true", help="sari peste mers, fă doar pozele")
    args = ap.parse_args()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    summary = {"started": datetime.now().isoformat(timespec="seconds"), "photos": {}}

    if args.no_drive:
        summary["drive"] = "skipped"
    else:
        reason, d = drive_forward(args.drive_s, args.stop_cm)
        summary["drive"] = {"stop_reason": reason, "last_distance_cm": d}
        if reason == "board_unreachable":
            log("anulez rularea")
            return 1

    # Pan position relative to home, so we can always go back.
    position = 0
    try:
        summary["photos"]["home"] = capture(os.path.join(out, "home.jpg"))
        pan(DIR_LEFT, PAN_STEPS)
        position = -PAN_STEPS
        summary["photos"]["left"] = capture(os.path.join(out, "left.jpg"))
        pan(DIR_RIGHT, 2 * PAN_STEPS)
        position = PAN_STEPS
        summary["photos"]["right"] = capture(os.path.join(out, "right.jpg"))
    finally:
        if position:
            pan(DIR_LEFT if position > 0 else DIR_RIGHT, abs(position))
            position = 0
        log("camera înapoi în HOME")

    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"gata -> {out}")
    return 0 if all(summary["photos"].values()) else 2


if __name__ == "__main__":
    sys.exit(main())
