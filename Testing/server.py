"""
VisionX Backend — ESP32 Frame Capture → Video Generator
=========================================================
Flow:
  GET /toggle ON  → Flask starts pulling frames from ESP32
                  → Frames written sequentially into a .avi video file
  GET /toggle OFF → Stop pulling, finalize and save video on laptop

Endpoints:
  GET /toggle  → Toggle system ON / OFF
  GET /status  → Current state + stats
  GET /videos  → List all saved video files
"""

import os
import io
import time
import threading
import requests
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# ⚙️  CONFIG — edit to match your ESP32 setup
# ─────────────────────────────────────────────────────────────
ESP32_IP            = os.getenv("ESP32_IP", "http://192.168.1.XXX")
ESP32_CAPTURE_PATH  = "/capture"          # ESP32 endpoint that returns a JPEG frame
ESP32_POLL_INTERVAL = 0.1                 # seconds between frame fetches (~10 fps)
VIDEO_FPS           = 10.0               # output video FPS
VIDEO_DIR           = "captured_videos"  # folder where .avi files are saved

os.makedirs(VIDEO_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 🔄  SHARED STATE
# ─────────────────────────────────────────────────────────────
state = {
    "active":         False,
    "capture_thread": None,
    "stop_event":     None,
    "frames_written": 0,
    "current_video":  None,
    "total_videos":   0,
}


# ─────────────────────────────────────────────────────────────
# 🛠  HELPERS
# ─────────────────────────────────────────────────────────────

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(tag: str, msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{tag}] {msg}")


def fetch_frame(url: str):
    """
    Fetch a single JPEG frame from the ESP32.
    Returns a decoded OpenCV BGR image (numpy array) or None on failure.
    """
    response = requests.get(url, timeout=5)
    if response.status_code != 200:
        log("ESP32", f"⚠️  HTTP {response.status_code} — skipping frame")
        return None

    # Decode raw JPEG bytes → OpenCV image
    img_array = np.frombuffer(response.content, dtype=np.uint8)
    frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        log("ESP32", "⚠️  Could not decode image — skipping frame")
    return frame


# ─────────────────────────────────────────────────────────────
# 🎥  CAPTURE THREAD
# Continuously pulls JPEG frames from ESP32 and writes them
# into a VideoWriter. Runs until stop_event is set.
# ─────────────────────────────────────────────────────────────

def capture_worker(stop_event: threading.Event, video_path: str):
    url    = f"{ESP32_IP}{ESP32_CAPTURE_PATH}"
    writer = None
    width  = None
    height = None

    log("CAPTURE", f"Starting — pulling frames from {url}")
    log("CAPTURE", f"Output video → {video_path}")

    while not stop_event.is_set():
        try:
            frame = fetch_frame(url)

            if frame is None:
                time.sleep(ESP32_POLL_INTERVAL)
                continue

            h, w = frame.shape[:2]

            # ── Initialize VideoWriter on first valid frame ──────────
            if writer is None:
                width, height = w, h
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))

                if not writer.isOpened():
                    log("CAPTURE", "❌ VideoWriter failed to open. Check codec / path.")
                    return

                log("CAPTURE", f"🔴 VideoWriter ready [{width}x{height} @ {VIDEO_FPS}fps]")

            # ── Resize if ESP32 changes resolution mid-stream ────────
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
            state["frames_written"] += 1

            if state["frames_written"] % 50 == 0:
                log("CAPTURE", f"📹 {state['frames_written']} frames written ...")

        except requests.exceptions.ConnectionError:
            log("CAPTURE", "❌ ESP32 not reachable — retrying ...")
        except requests.exceptions.Timeout:
            log("CAPTURE", "⏱️  Request timed out — retrying ...")
        except Exception as e:
            log("CAPTURE", f"Unexpected error: {e}")

        time.sleep(ESP32_POLL_INTERVAL)

    # ── Finalize video ───────────────────────────────────────────────
    if writer and writer.isOpened():
        writer.release()
        log("CAPTURE", f"✅ Video saved → {video_path}  ({state['frames_written']} frames)")
    else:
        log("CAPTURE", "⚠️  No frames were written — video not saved")

    log("CAPTURE", "🛑 Capture thread stopped")


# ─────────────────────────────────────────────────────────────
# 🌐  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return "VisionX Backend Running 🚀"


@app.route("/toggle", methods=["GET"])
def toggle():
    """Toggle ESP32 frame capture + video recording ON ↔ OFF."""

    if not state["active"]:
        # ── Turn ON ──────────────────────────────────────────────────
        log("TOGGLE", "🟢 Switching ON — starting ESP32 frame capture")

        video_filename          = f"esp32_{timestamp()}.avi"
        video_path              = os.path.join(VIDEO_DIR, video_filename)
        state["current_video"]  = video_path
        state["frames_written"] = 0
        state["active"]         = True

        stop_event = threading.Event()
        state["stop_event"] = stop_event

        capture_thread = threading.Thread(
            target=capture_worker,
            args=(stop_event, video_path),
            daemon=True,
            name="ESP32-Capture"
        )
        capture_thread.start()
        state["capture_thread"] = capture_thread

        return jsonify({
            "success":     True,
            "state":       "ON",
            "video_file":  video_filename,
            "message":     "ESP32 frame capture started — video recording active"
        }), 200

    else:
        # ── Turn OFF ─────────────────────────────────────────────────
        log("TOGGLE", "🔴 Switching OFF — stopping capture + finalizing video")

        state["stop_event"].set()

        if state["capture_thread"]:
            state["capture_thread"].join(timeout=10)

        state["total_videos"]  += 1
        state["active"]         = False
        state["capture_thread"] = None
        state["stop_event"]     = None

        return jsonify({
            "success":        True,
            "state":          "OFF",
            "frames_written": state["frames_written"],
            "video_saved":    state["current_video"],
            "message":        "Capture stopped — video finalized and saved"
        }), 200


@app.route("/status", methods=["GET"])
def status():
    """Return current system state and stats."""
    return jsonify({
        "success":         True,
        "state":           "ON" if state["active"] else "OFF",
        "frames_written":  state["frames_written"],
        "current_video":   state["current_video"],
        "total_videos":    state["total_videos"],
        "esp32_url":       f"{ESP32_IP}{ESP32_CAPTURE_PATH}",
        "poll_interval_s": ESP32_POLL_INTERVAL,
        "output_fps":      VIDEO_FPS,
    }), 200


@app.route("/videos", methods=["GET"])
def list_videos():
    """List all saved video files."""
    try:
        files = sorted(os.listdir(VIDEO_DIR))
        return jsonify({
            "success": True,
            "total":   len(files),
            "videos":  files
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  VisionX Backend — ESP32 → Video Recorder")
    print("=" * 55)
    print(f"  ESP32 URL      : {ESP32_IP}{ESP32_CAPTURE_PATH}")
    print(f"  Poll interval  : {ESP32_POLL_INTERVAL}s  (~{int(1/ESP32_POLL_INTERVAL)} fps)")
    print(f"  Output FPS     : {VIDEO_FPS}")
    print(f"  Videos folder  : {os.path.abspath(VIDEO_DIR)}")
    print("=" * 55)
    print("  Endpoints:")
    print("    GET /toggle  → Start / Stop capture + recording")
    print("    GET /status  → Current state + stats")
    print("    GET /videos  → List all saved videos")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=8000)